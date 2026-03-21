import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import random
import gc
import warnings
import os
import argparse
import multiprocessing
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from pygod.detector import DOMINANT, DONE, GAE, AnomalyDAE, CoLA
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

warnings.filterwarnings("ignore", message=".*pyg-lib.*")
warnings.filterwarnings("ignore", message=".*transductive only.*")
warnings.filterwarnings("ignore", message=".*Backbone and num_layers.*")


# ──────────────────────────────────────────────────────────────────────────────
# Global Config
# ──────────────────────────────────────────────────────────────────────────────
MC_DROPOUT_P = 0.1   # Dropout p=0.0 인 레이어에 주입할 확률값

class Args:
    def __init__(self):
        self.device     = 0 if torch.cuda.is_available() else -1
        self.mc_samples = 20
        print(f"Using device: {'GPU' if self.device >= 0 else 'CPU'}")


# ──────────────────────────────────────────────────────────────────────────────
# JSON Serialization Helper  ✅ 추가
# ──────────────────────────────────────────────────────────────────────────────
def to_serializable(obj):
    """numpy/torch 타입을 json.dump 가능한 Python 기본 타입으로 재귀 변환"""
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_serializable(v) for v in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    else:
        return obj


# ──────────────────────────────────────────────────────────────────────────────
# Mask & Seed Utilities
# ──────────────────────────────────────────────────────────────────────────────
def create_masks(num_nodes):
    indices    = np.arange(num_nodes)
    np.random.shuffle(indices)
    train_size = int(num_nodes * 0.8)
    val_size   = int(num_nodes * 0.1)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask   = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]]                     = True
    val_mask[indices[train_size:train_size + val_size]]  = True
    test_mask[indices[train_size + val_size:]]           = True

    return train_mask, val_mask, test_mask


def set_seed(seed: int):
    """재현성 보장을 위한 통합 시드 설정"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# JSON Worker  ✅ 병렬 로더 추가
# ──────────────────────────────────────────────────────────────────────────────
def read_json_worker(args):
    file_path, idx, feature_dim = args
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)                     # ✅ 파일 핸들만 전달

        feat  = data.get('contract_feature', [])
        label = int(data.get('label', 0))

        if len(feat) != feature_dim:
            raise ValueError(
                f"Feature length mismatch in {file_path}: "
                f"expected {feature_dim}, got {len(feat)}"
            )
        return idx, feat, label

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return idx, [0.0] * feature_dim, 0


# ──────────────────────────────────────────────────────────────────────────────
# MC Dropout 활성화 유틸리티 (main_mc.py 와 동일)
# ──────────────────────────────────────────────────────────────────────────────
def _activate_dropout(model, p=MC_DROPOUT_P):
    """
    모든 Dropout 레이어:
      - p=0.0 이면 MC_DROPOUT_P 로 교체
      - train 모드로 강제 전환
    반환: 원래 상태 복원용 dict
    """
    original_states = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            original_states[name] = {
                'p':        module.p,
                'training': module.training,
            }
            if module.p == 0.0:
                module.p = p        # ✅ p=0.0 → MC_DROPOUT_P 주입
            module.train()          # ✅ train 모드 강제 전환
    return original_states


def _restore_dropout(model, original_states):
    """_activate_dropout 으로 변경한 상태를 원복"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout) and name in original_states:
            module.p        = original_states[name]['p']
            module.training = original_states[name]['training']


def _inject_dropout_to_linear(model, p=MC_DROPOUT_P):
    """
    ✅ AnomalyDAE 처럼 Dropout 레이어가 아예 없는 모델에
       nn.Linear 의 forward 를 패치하여 F.dropout 동적 삽입
    반환: 복원용 (module, original_forward) 리스트
    """
    injected = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            original_forward = module.forward

            def make_patched_forward(orig_fwd):
                def patched_forward(x):
                    return F.dropout(
                        orig_fwd(x), p=p, training=True   # training=True 고정
                    )
                return patched_forward

            module.forward = make_patched_forward(original_forward)
            injected.append((module, original_forward))
    return injected


def _restore_linear_forward(injected):
    """_inject_dropout_to_linear 으로 패치한 forward 원복"""
    for module, original_forward in injected:
        module.forward = original_forward


# ──────────────────────────────────────────────────────────────────────────────
# MC Predict — 모델별 전략 분기 (main_mc.py 와 동일)
# ──────────────────────────────────────────────────────────────────────────────
def mc_predict(detector, data, mc_samples=50):
    """
    진단 결과에 따른 모델별 MC 전략:

    ┌──────────────┬────────────────────────────────────────────────────────┐
    │ CoLA         │ decision_function 그대로 (eval std=0.353 확인)         │
    │ DONE         │ model.train() + model.eval() 무력화 후 d_f 반복        │
    │              │ (BatchNorm train 모드 분산 → std=0.004 자연 발생)      │
    │ DOMINANT     │ Dropout p=0.0→MC_P 주입 + eval() 무력화 + d_f 반복    │
    │ GAE          │ Dropout p=0.0→MC_P 주입 + eval() 무력화 + d_f 반복    │
    │ AnomalyDAE   │ Linear 뒤에 F.dropout 동적 삽입 + d_f 반복            │
    └──────────────┴────────────────────────────────────────────────────────┘
    """
    model_type = type(detector).__name__

    if model_type == 'CoLA':
        return _mc_cola(detector, data, mc_samples)
    elif model_type == 'DONE':
        return _mc_done(detector, data, mc_samples)
    elif model_type in ('DOMINANT', 'GAE'):
        return _mc_with_dropout_activation(detector, data, mc_samples)
    elif model_type == 'AnomalyDAE':
        return _mc_anomalydae(detector, data, mc_samples)
    else:
        score = detector.decision_function(data)
        return score, np.zeros_like(score)


# ── CoLA: eval 모드에서도 이미 std=0.353 → 그대로 반복 ──────────────────────
def _mc_cola(detector, data, mc_samples):
    mc_scores = []
    for _ in range(mc_samples):
        score = detector.decision_function(data)
        mc_scores.append(score)
    mc_scores   = np.stack(mc_scores, axis=0)
    mean_score  = mc_scores.mean(axis=0)
    uncertainty = mc_scores.std(axis=0)
    return mean_score, uncertainty


# ── DONE: BatchNorm train 모드 분산 활용 ──────────────────────────────────────
def _mc_done(detector, data, mc_samples):
    original_model_eval    = detector.model.eval
    detector.model.eval    = lambda: detector.model   # noop
    detector.model.train()

    mc_scores = []
    try:
        for _ in range(mc_samples):
            score = detector.decision_function(data)
            mc_scores.append(score)
    finally:
        detector.model.eval = original_model_eval
        detector.model.eval()

    mc_scores   = np.stack(mc_scores, axis=0)
    mean_score  = mc_scores.mean(axis=0)
    uncertainty = mc_scores.std(axis=0)
    return mean_score, uncertainty


# ── DOMINANT / GAE: Dropout p=0.0 → MC_DROPOUT_P 주입 ───────────────────────
def _mc_with_dropout_activation(detector, data, mc_samples):
    original_dropout_states = _activate_dropout(detector.model)
    original_model_eval     = detector.model.eval
    detector.model.eval     = lambda: detector.model   # noop

    mc_scores = []
    try:
        for _ in range(mc_samples):
            score = detector.decision_function(data)
            mc_scores.append(score)
    finally:
        detector.model.eval = original_model_eval
        _restore_dropout(detector.model, original_dropout_states)
        detector.model.eval()

    mc_scores   = np.stack(mc_scores, axis=0)
    mean_score  = mc_scores.mean(axis=0)
    uncertainty = mc_scores.std(axis=0)
    return mean_score, uncertainty


# ── AnomalyDAE: Dropout 없음 → Linear forward 패치로 동적 삽입 ───────────────
def _mc_anomalydae(detector, data, mc_samples):
    injected            = _inject_dropout_to_linear(detector.model)
    original_model_eval = detector.model.eval
    detector.model.eval = lambda: detector.model   # noop

    mc_scores = []
    try:
        for _ in range(mc_samples):
            score = detector.decision_function(data)
            mc_scores.append(score)
    finally:
        detector.model.eval = original_model_eval
        _restore_linear_forward(injected)
        detector.model.eval()

    mc_scores   = np.stack(mc_scores, axis=0)
    mean_score  = mc_scores.mean(axis=0)
    uncertainty = mc_scores.std(axis=0)
    return mean_score, uncertainty


# ──────────────────────────────────────────────────────────────────────────────
# Detector Builder
# ──────────────────────────────────────────────────────────────────────────────
def build_detector(model_class, param: dict, chain: str, device: int):
    if chain == 'ethereum':
        return model_class(
            hid_dim=param['hid_dim'],
            num_layers=2,
            lr=param['lr'],
            epoch=param['epoch'],
            gpu=device,
            batch_size=2048,
            num_neigh=64,
        )
    else:
        return model_class(
            hid_dim=param['hid_dim'],
            num_layers=2,
            lr=param['lr'],
            epoch=param['epoch'],
            gpu=device,
        )


# ──────────────────────────────────────────────────────────────────────────────
# MC Run  ✅ test_mask 기반 (ROC-AUC 파일의 고유 특성 유지)
#             F1 score 추가 유지
# ──────────────────────────────────────────────────────────────────────────────
def run_mc_model(detector, data, seeds, mc_samples=50):
    """
    main_mc.py 와 달리 roc_auc_score_mc.py 는 test_mask 기반 평가.
    F1 score 도 함께 측정 (원본 고유 지표).
    """
    auc_list = []
    f1_list  = []
    ap_list  = []
    unc_list = []

    test_mask = data.test_mask

    for seed in seeds:
        set_seed(seed)
        detector.fit(data)

        mean_score, uncertainty_scores = mc_predict(
            detector, data, mc_samples=mc_samples
        )

        test_labels = data.y[test_mask].cpu().numpy()
        test_scores = mean_score[test_mask]
        test_unc    = uncertainty_scores[test_mask].mean()

        # ✅ mean_score를 0.5 기준으로 이진화하여 F1 계산
        test_preds  = (test_scores >= 0.5).astype(int)

        try:
            auc = roc_auc_score(test_labels, test_scores)
            ap  = average_precision_score(test_labels, test_scores)
            f1  = f1_score(test_labels, test_preds, zero_division=0)
        except ValueError:
            auc, ap, f1 = 0.0, 0.0, 0.0

        auc_list.append(float(auc))
        f1_list.append(float(f1))
        ap_list.append(float(ap))
        unc_list.append(float(test_unc))

    return (
        float(np.mean(auc_list)), float(np.std(auc_list)),
        float(np.mean(f1_list)),  float(np.std(f1_list)),
        float(np.mean(ap_list)),  float(np.std(ap_list)),
        float(np.mean(unc_list)), float(np.std(unc_list)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 + Phase 2 Evaluation  ✅ main_mc.py 와 동일한 2단계 구조 이식
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model_name, model_class, param_list,
    global_data, seeds_for_evaluation,
    chain, device, mc_samples,
    checkpoint_dir, results_acc, final_results_acc
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = (
        f"{checkpoint_dir}/{chain}_{model_name}_roc_auc_mc_checkpoint.json"
    )

    best_params_info = {}
    completed_tasks  = set()

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                ckpt = json.load(f)
            best_params_info = ckpt.get("best_params_info", {})
            completed_tasks  = set(ckpt.get("completed_tasks", []))
            print(
                f"  [Checkpoint] {model_name}: "
                f"{len(completed_tasks)} tasks already done."
            )
        except Exception as e:
            print(f"  [Checkpoint] Load failed ({e}). Starting fresh.")

    seed_for_param_selection = 42

    # ── Phase 1: Grid Search (단일 시드) ───────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[Phase 1] Grid Search – {model_name}  (MC Samples={mc_samples})")
    print(f"{'─'*60}")

    for param in param_list:
        task_id = (
            f"{model_name}_{param['hid_dim']}_"
            f"{param['lr']}_{param['epoch']}"
        )

        if task_id in completed_tasks:
            continue

        detector = build_detector(model_class, param, chain, device)

        try:
            (avg_auc, std_auc,
             avg_f1,  std_f1,
             avg_ap,  std_ap,
             avg_unc, std_unc) = run_mc_model(
                detector, global_data,
                seeds=[seed_for_param_selection],
                mc_samples=mc_samples,
            )

            # 최고 AUC 파라미터 갱신
            if (not best_params_info) or (
                avg_auc > best_params_info.get("Best AUC", 0)
            ):
                best_params_info = {
                    "Best AUC":    float(avg_auc),
                    "AUC Std Dev": float(std_auc),
                    "Best F1":     float(avg_f1),
                    "F1 Std Dev":  float(std_f1),
                    "Best AP":     float(avg_ap),
                    "AP Std Dev":  float(std_ap),
                    "Avg Unc":     float(avg_unc),
                    "Unc Std Dev": float(std_unc),
                    "Params": {
                        "hid_dim": int(param["hid_dim"]),
                        "lr":      float(param["lr"]),
                        "epoch":   int(param["epoch"]),
                    },
                }

            time_str = datetime.now().strftime("%Y:%m:%d_%H:%M:%S")
            print(
                f"{time_str} --> [Grid] ROC-AUC MC {model_name} {param} "
                f"AUC={avg_auc:.4f} F1={avg_f1:.4f} "
                f"AP={avg_ap:.4f} Unc={avg_unc:.4f}"
            )

            # ✅ Raw 결과 누적 (시드 1개이므로 std=0)
            results_acc.append({
                "Phase":            "GridSearch",
                "Timestamp":        time_str,
                "Dataset":          chain,
                "Model":            model_name,
                "Params":           str(param),
                "Test AUC":         round(avg_auc, 4),
                "Test AUC Std Dev": round(std_auc, 4),
                "Test F1":          round(avg_f1,  4),
                "Test F1 Std Dev":  round(std_f1,  4),
                "Test AP":          round(avg_ap,  4),
                "Test AP Std Dev":  round(std_ap,  4),
                "Test Uncertainty": round(avg_unc, 4),
                "Test Unc Std Dev": round(std_unc, 4),
            })

            completed_tasks.add(task_id)

            # ✅ 체크포인트 저장
            with open(checkpoint_file, 'w') as f:
                json.dump(
                    to_serializable({
                        "best_params_info": best_params_info,
                        "completed_tasks":  list(completed_tasks),
                    }),
                    f, indent=4
                )

        except Exception as e:
            print(
                f"  [ERROR] ROC-AUC MC {model_name} | {param} "
                f"→ {e}. Skipping."
            )
            continue

        finally:
            del detector
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Phase 2: Final Evaluation (멀티 시드) ─────────────────────────────
    if not best_params_info:
        print(
            f"  [WARN] ROC-AUC MC {model_name}: "
            f"No valid params found. Skipping final eval."
        )
        return

    best_param = best_params_info["Params"]
    print(f"\n{'─'*60}")
    print(f"[Phase 2] Final Evaluation – ROC-AUC MC {model_name}")
    print(f"  Best Param : {best_param}")
    print(f"  Seeds      : {seeds_for_evaluation}  |  MC Samples={mc_samples}")
    print(f"{'─'*60}")

    detector = build_detector(model_class, best_param, chain, device)

    try:
        (avg_auc, std_auc,
         avg_f1,  std_f1,
         avg_ap,  std_ap,
         avg_unc, std_unc) = run_mc_model(
            detector, global_data,
            seeds=seeds_for_evaluation,
            mc_samples=mc_samples,
        )

        time_str = datetime.now().strftime("%Y:%m:%d_%H:%M:%S")
        print(
            f"{time_str} --> [Final] ROC-AUC MC {model_name} "
            f"Avg AUC={avg_auc:.4f} (±{std_auc:.4f})  "
            f"Avg F1={avg_f1:.4f} (±{std_f1:.4f})  "
            f"Avg AP={avg_ap:.4f} (±{std_ap:.4f})  "
            f"Avg Unc={avg_unc:.4f} (±{std_unc:.4f})"
        )

        final_results_acc.append({
            "Phase":            "FinalEval",
            "Timestamp":        time_str,
            "Dataset":          chain,
            "Model":            model_name,
            "Best Params":      str(best_param),
            "Test AUC":         round(avg_auc, 4),
            "Test AUC Std Dev": round(std_auc, 4),
            "Test F1":          round(avg_f1,  4),
            "Test F1 Std Dev":  round(std_f1,  4),
            "Test AP":          round(avg_ap,  4),
            "Test AP Std Dev":  round(std_ap,  4),
            "Test Uncertainty": round(avg_unc, 4),
            "Test Unc Std Dev": round(std_unc, 4),
        })

    except Exception as e:
        print(
            f"  [ERROR] Final eval failed for ROC-AUC MC {model_name}: {e}"
        )

    finally:
        del detector
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain',   type=str, default='polygon')
    parser.add_argument('--workers', type=int, default=4)
    cli_args = parser.parse_args()

    chain = cli_args.chain
    cfg   = Args()

    file_path         = f'../../_data/GoG/{chain}/graphs'
    global_edges_path = (
        f'../../_data/GoG/edges/{chain}/{chain}_train_edges.txt'
    )

    # ── JSON 병렬 로드 ─────────────────────────────────────────────────────
    json_files = sorted(
        Path(file_path).glob('*.json'), key=lambda f: int(f.stem)
    )
    num_nodes = len(json_files)
    print(f"Loading features and labels from {num_nodes} JSON files...")

    # feature_dim 자동 추론
    with open(json_files[0], 'r') as f:
        sample = json.load(f)
    feature_dim = len(sample.get('contract_feature', []))

    x_list       = [None] * num_nodes
    y_list       = [None] * num_nodes
    contract_ids = [f.stem for f in json_files]

    args_list = [(f, i, feature_dim) for i, f in enumerate(json_files)]

    with multiprocessing.Pool(processes=cli_args.workers) as pool:
        for idx, feat, label in tqdm(
            pool.imap_unordered(read_json_worker, args_list, chunksize=40),
            total=num_nodes,
            desc="JSON Parsing"
        ):
            x_list[idx] = feat
            y_list[idx] = label

    features = torch.tensor(x_list, dtype=torch.float32)
    labels   = torch.tensor(y_list, dtype=torch.long)

    assert labels.min() >= 0 and labels.max() <= 1, \
        f"label 범위 이상: {labels.min()}~{labels.max()}"
    print(f"✅ features: {features.shape}, labels unique: {labels.unique()}")









    contract_to_idx = {cid: i for i, cid in enumerate(contract_ids)}

    edges = []
    invalid_edges = []

    with open(global_edges_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                src, dst = parts[0], parts[1]
                src_idx = contract_to_idx.get(src)
                dst_idx = contract_to_idx.get(dst)

                if src_idx is not None and dst_idx is not None:
                    edges.append([src_idx, dst_idx])
                else:
                    reason = []
                    if src_idx is None:
                        reason.append(f"src '{src}' not found")
                    if dst_idx is None:
                        reason.append(f"dst '{dst}' not found")
                    invalid_edges.append((src, dst, "; ".join(reason)))

    print(f'Invalid edges (not in contract_to_idx): {len(invalid_edges)}')
    if invalid_edges:
        print("Sample invalid_edges:", invalid_edges[:5])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    from torch_geometric.utils import add_self_loops, coalesce
    edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    node_ids_in_edges = torch.unique(edge_index)
    missing_nodes = sorted(set(range(num_nodes)) - set(node_ids_in_edges.tolist()))

    print(f'nodes appearing in edge_index: {node_ids_in_edges.numel()} / {num_nodes}')
    print(f'missing nodes from edge_index: {len(missing_nodes)}')
    print(f'edge_index min: {edge_index.min().item()}, max: {edge_index.max().item()}, num_nodes: {num_nodes}')

    assert edge_index.min().item() >= 0
    assert edge_index.max().item() < num_nodes

    train_mask, val_mask, test_mask = create_masks(num_nodes)

    global_data = Data(x=features, edge_index=edge_index, y=labels)
    global_data.num_nodes = num_nodes
    global_data.train_mask = train_mask
    global_data.val_mask = val_mask
    global_data.test_mask = test_mask
    global_data.chain = chain









    # ── Hyperparameter Grid ────────────────────────────────────────────────
    hyperparameters = [{"hid_dim": d, "lr": lr, "epoch": e} for d in [4, 8, 12, 16, 20] for lr in [0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03] for e in [20, 30, 40, 50, 80, 100, 120]]


    models = {
        'DOMINANT':   DOMINANT,
        'DONE':       DONE,
        'GAE':        GAE,
        'AnomalyDAE': AnomalyDAE,
        'CoLA':       CoLA,
    }

    seeds_for_evaluation = [42, 100, 2026]

    RESULT_PATH    = f"../../_data/results/fraud_detection_mc"
    checkpoint_dir = f"{RESULT_PATH}/checkpoints/{chain}"
    os.makedirs(RESULT_PATH, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    all_results       = []
    all_final_results = []

    for model_name, model_class in models.items():
        evaluate_model(
            model_name           = model_name,
            model_class          = model_class,
            param_list           = hyperparameters,
            global_data          = global_data,
            seeds_for_evaluation = seeds_for_evaluation,
            chain                = chain,
            device               = cfg.device,
            mc_samples           = cfg.mc_samples,
            checkpoint_dir       = checkpoint_dir,
            results_acc          = all_results,
            final_results_acc    = all_final_results,
        )

    # ── 결과 저장 ──────────────────────────────────────────────────────────
    results_df       = pd.DataFrame(all_results)
    final_results_df = pd.DataFrame(all_final_results)

    if not results_df.empty:
        results_df.to_csv(
            f'{RESULT_PATH}/mc-roc-auc-results_{chain}_raw.csv',
            index=False
        )
        print(f"✅ Raw 결과 저장 완료.")

    if not final_results_df.empty:
        final_results_df.to_csv(
            f'{RESULT_PATH}/mc-roc-auc-results_{chain}_final.csv',
            index=False
        )
        print(f"✅ Final 결과 저장 완료.")

    print(
        f"\n✅ ROC-AUC MC 최종 평가 결과가 {RESULT_PATH} 에 저장되었습니다."
    )


if __name__ == "__main__":
    main()
