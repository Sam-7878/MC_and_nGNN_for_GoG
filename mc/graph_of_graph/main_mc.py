import torch
import torch.nn as nn
import numpy as np
import random
from pygod.detector import DOMINANT, DONE, GAE, AnomalyDAE, CoLA
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
import json
import os
import gc
from pathlib import Path
from torch_geometric.utils import coalesce
import argparse
import multiprocessing
from datetime import datetime
from tqdm import tqdm
import pandas as pd

warnings.filterwarnings("ignore", message=".*pyg-lib.*")
warnings.filterwarnings("ignore", message=".*transductive only.*")
warnings.filterwarnings("ignore", message=".*Backbone and num_layers.*")


# ──────────────────────────────────────────────────────────────────────────────
# Global Config
# ──────────────────────────────────────────────────────────────────────────────
MC_DROPOUT_P = 0.1   # Dropout이 p=0.0인 경우 주입할 확률값

class Args:
    def __init__(self):
        self.device = 0 if torch.cuda.is_available() else -1
        self.mc_samples = 20  # MC 샘플링 횟수 (평가 시 반복 횟수)
        print(f"Using device: {'GPU' if self.device >= 0 else 'CPU'}")


# ──────────────────────────────────────────────────────────────────────────────
# JSON Serialization Helper
# ──────────────────────────────────────────────────────────────────────────────
def to_serializable(obj):
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
    indices = np.arange(num_nodes)
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────────────────────
# JSON Worker
# ──────────────────────────────────────────────────────────────────────────────
def read_json_worker(args):
    file_path, idx, feature_dim = args
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

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
# MC Dropout 활성화 유틸리티
# ──────────────────────────────────────────────────────────────────────────────
def _activate_dropout(model, p=MC_DROPOUT_P):
    """
    모든 Dropout 레이어를:
      1) p=0.0 이면 p를 MC_DROPOUT_P 로 교체
      2) train 모드로 전환 (eval 모드에서는 Dropout 비활성화되므로)
    반환: 원래 (p, training) 상태 복원용 dict
    """
    original_states = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            original_states[name] = {
                'p':        module.p,
                'training': module.training,
            }
            if module.p == 0.0:
                module.p = p          # ✅ p=0.0 → p=0.1 강제 주입
            module.train()            # ✅ train 모드로 강제 전환
    return original_states


def _restore_dropout(model, original_states):
    """_activate_dropout 으로 변경한 상태를 원복"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout) and name in original_states:
            module.p        = original_states[name]['p']
            module.training = original_states[name]['training']


def _inject_dropout_to_linear(model, p=MC_DROPOUT_P):
    """
    ✅ AnomalyDAE처럼 Dropout 레이어가 아예 없는 모델에
       nn.Linear 직후마다 Dropout 을 동적으로 삽입
    반환: 복원용 원래 forward 함수들
    """
    injected = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            original_forward = module.forward

            def make_patched_forward(orig_fwd):
                def patched_forward(x):
                    return nn.functional.dropout(
                        orig_fwd(x), p=p, training=True  # training=True 고정
                    )
                return patched_forward

            module.forward = make_patched_forward(original_forward)
            injected.append((module, original_forward))

    return injected   # 복원용


def _restore_linear_forward(injected):
    """_inject_dropout_to_linear 으로 패치한 forward 원복"""
    for module, original_forward in injected:
        module.forward = original_forward


# ──────────────────────────────────────────────────────────────────────────────
# MC Predict  ← 모델별 전략 분기
# ──────────────────────────────────────────────────────────────────────────────
def mc_predict(detector, data, mc_samples=10):
    """
    진단 결과에 따른 모델별 MC 전략:

    ┌──────────────┬────────────────────────────────────────────────────────┐
    │ CoLA         │ decision_function 그대로 (이미 eval std=0.353 확인)    │
    │ DONE         │ model.train() 후 decision_function                     │
    │              │ (BatchNorm train 모드 분산 → std=0.004 자연 발생)      │
    │ DOMINANT     │ Dropout p=0.0→0.1 주입 + model.train() + d_f 호출     │
    │ GAE          │ Dropout p=0.0→0.1 주입 + model.train() + d_f 호출     │
    │ AnomalyDAE   │ Linear 뒤에 Dropout 동적 삽입 + d_f 호출              │
    └──────────────┴────────────────────────────────────────────────────────┘
    """
    model_type = type(detector).__name__

    # ── 전략 분기 ────────────────────────────────────────────────────────────
    if model_type == 'CoLA':
        return _mc_cola(detector, data, mc_samples)

    elif model_type == 'DONE':
        return _mc_done(detector, data, mc_samples)

    elif model_type in ('DOMINANT', 'GAE'):
        return _mc_with_dropout_activation(detector, data, mc_samples)

    elif model_type == 'AnomalyDAE':
        return _mc_anomalydae(detector, data, mc_samples)

    else:
        # 알 수 없는 모델 → 기본 방식 (불확실성 0 반환)
        score = detector.decision_function(data)
        return score, np.zeros_like(score)


# ── CoLA: 이미 정상 작동 (eval 모드에서도 std=0.353) ─────────────────────────
def _mc_cola(detector, data, mc_samples):
    """
    CoLA는 decision_function 자체가 랜덤성 있음 → 그대로 반복 호출
    (내부적으로 subgraph sampling 등의 stochastic 요소 존재)
    """
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
    """
    DONE은 Dropout이 없지만 BatchNorm1d가 train 모드에서
    running stats 대신 batch stats를 사용 → 자연스러운 분산 발생
    (진단에서 std=0.003584 확인)
    """
    mc_scores = []

    # model.eval() 무력화: decision_function 내부의 model.eval() 차단
    original_model_eval        = detector.model.eval
    detector.model.eval        = lambda: detector.model   # noop
    detector.model.train()     # BatchNorm을 train 모드로 전환

    try:
        for _ in range(mc_samples):
            score = detector.decision_function(data)
            mc_scores.append(score)
    finally:
        detector.model.eval = original_model_eval
        detector.model.eval()    # 복구

    mc_scores   = np.stack(mc_scores, axis=0)
    mean_score  = mc_scores.mean(axis=0)
    uncertainty = mc_scores.std(axis=0)
    return mean_score, uncertainty


# ── DOMINANT / GAE: Dropout p=0.0 → p=0.1 주입 ───────────────────────────────
def _mc_with_dropout_activation(detector, data, mc_samples):
    """
    Dropout 레이어는 있지만 p=0.0 인 경우:
      1) p=0.0 → MC_DROPOUT_P(=0.1)로 강제 변경
      2) model.eval() 무력화
      3) decision_function 반복 호출
      4) p / training 상태 원복
    """
    # 1. Dropout p 주입 + train 모드 전환
    original_dropout_states    = _activate_dropout(detector.model)

    # 2. model.eval() 무력화
    original_model_eval        = detector.model.eval
    detector.model.eval        = lambda: detector.model   # noop

    mc_scores = []
    try:
        for _ in range(mc_samples):
            score = detector.decision_function(data)
            mc_scores.append(score)
    finally:
        # 3. 반드시 복구
        detector.model.eval = original_model_eval
        _restore_dropout(detector.model, original_dropout_states)
        detector.model.eval()    # 정상 eval 복구

    mc_scores   = np.stack(mc_scores, axis=0)
    mean_score  = mc_scores.mean(axis=0)
    uncertainty = mc_scores.std(axis=0)
    return mean_score, uncertainty


# ── AnomalyDAE: Dropout 자체 없음 → Linear 뒤에 동적 삽입 ───────────────────
def _mc_anomalydae(detector, data, mc_samples):
    """
    AnomalyDAE는 Dropout도 BatchNorm도 없어 완전 deterministic.
    nn.Linear의 forward를 패치하여 functional.dropout을 동적으로 삽입.
    training=True 고정으로 eval 모드에서도 Dropout 활성화.
    """
    # 1. Linear forward 패치
    injected = _inject_dropout_to_linear(detector.model)

    # 2. model.eval() 무력화 (혹시 몰라 추가)
    original_model_eval   = detector.model.eval
    detector.model.eval   = lambda: detector.model   # noop

    mc_scores = []
    try:
        for _ in range(mc_samples):
            score = detector.decision_function(data)
            mc_scores.append(score)
    finally:
        # 3. 반드시 복구
        detector.model.eval = original_model_eval
        _restore_linear_forward(injected)
        detector.model.eval()

    mc_scores   = np.stack(mc_scores, axis=0)
    mean_score  = mc_scores.mean(axis=0)
    uncertainty = mc_scores.std(axis=0)
    return mean_score, uncertainty


# ──────────────────────────────────────────────────────────────────────────────
# MC Run
# ──────────────────────────────────────────────────────────────────────────────
def run_mc_model(detector, data, seeds, mc_samples=10):
    auc_list = []
    ap_list  = []
    unc_list = []

    val_mask = data.val_mask

    for seed in seeds:
        set_seed(seed)
        detector.fit(data)

        mean_score, uncertainty_scores = mc_predict(
            detector, data, mc_samples=mc_samples
        )

        val_labels = data.y[val_mask].cpu().numpy()
        val_scores = mean_score[val_mask]
        val_unc    = uncertainty_scores[val_mask].mean()

        try:
            auc = roc_auc_score(val_labels, val_scores)
            ap  = average_precision_score(val_labels, val_scores)
        except ValueError:
            auc, ap = 0.0, 0.0

        auc_list.append(float(auc))
        ap_list.append(float(ap))
        unc_list.append(float(val_unc))

    return (
        float(np.mean(auc_list)), float(np.std(auc_list)),
        float(np.mean(ap_list)),  float(np.std(ap_list)),
        float(np.mean(unc_list)), float(np.std(unc_list)),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Detector Builder
# ──────────────────────────────────────────────────────────────────────────────
def build_detector(model_class, param: dict, chain: str, device: int):
    if chain == 'ethereum':
        return model_class(
            hid_dim=param['hid_dim'],
            num_layers=2,
            epoch=param['epoch'],
            lr=param['lr'],
            gpu=device,
            batch_size=2048,
            num_neigh=12
        )
    else:
        return model_class(
            hid_dim=param['hid_dim'],
            num_layers=2,
            epoch=param['epoch'],
            lr=param['lr'],
            gpu=device,
        )


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 + Phase 2 Evaluation
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_model(
    model_name, model_class, param_list,
    global_data, seeds_for_evaluation,
    chain, device, mc_samples,
    checkpoint_dir, results_acc, final_results_acc
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = f"{checkpoint_dir}/{chain}_{model_name}_mc_checkpoint.json"

    best_params_info = {}
    completed_tasks  = set()

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                ckpt = json.load(f)
            best_params_info = ckpt.get("best_params_info", {})
            completed_tasks  = set(ckpt.get("completed_tasks", []))
            print(f"  [Checkpoint] {model_name}: {len(completed_tasks)} tasks already done.")
        except Exception as e:
            print(f"  [Checkpoint] Load failed ({e}). Starting fresh.")

    seed_for_param_selection = 42

    # ── Phase 1: Grid Search ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"[Phase 1] Grid Search – {model_name}  (MC Samples={mc_samples})")
    print(f"{'─'*60}")

    for param in param_list:
        task_id = f"{model_name}_{param['hid_dim']}_{param['lr']}_{param['epoch']}"

        if task_id in completed_tasks:
            continue

        detector = build_detector(model_class, param, chain, device)

        try:
            avg_auc, std_auc, avg_ap, std_ap, avg_unc, std_unc = run_mc_model(
                detector, global_data,
                seeds=[seed_for_param_selection],
                mc_samples=mc_samples
            )

            if (not best_params_info) or (avg_auc > best_params_info.get("Best AUC", 0)):
                best_params_info = {
                    "Best AUC":    float(avg_auc),
                    "AUC Std Dev": float(std_auc),
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
                f"{time_str} --> [Grid] {model_name} {param} "
                f"AUC={avg_auc:.4f} AP={avg_ap:.4f} Unc={avg_unc:.4f}"
            )

            results_acc.append({
                "Phase":           "GridSearch",
                "Timestamp":       time_str,
                "Dataset":         chain,
                "Model":           model_name,
                "Params":          str(param),
                "Val AUC":         round(avg_auc, 4),
                "Val AUC Std Dev": round(std_auc, 4),
                "Val AP":          round(avg_ap,  4),
                "Val AP Std Dev":  round(std_ap,  4),
                "Val Uncertainty": round(avg_unc, 4),
                "Val Unc Std Dev": round(std_unc, 4),
            })

            completed_tasks.add(task_id)

            with open(checkpoint_file, 'w') as f:
                json.dump(
                    to_serializable({
                        "best_params_info": best_params_info,
                        "completed_tasks":  list(completed_tasks)
                    }),
                    f, indent=4
                )

        except Exception as e:
            print(f"  [ERROR] {model_name} | {param} → {e}. Skipping.")
            continue

        finally:
            del detector
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Phase 2: Final Evaluation ───────────────────────────────────────────
    if not best_params_info:
        print(f"  [WARN] {model_name}: No valid params found. Skipping final eval.")
        return

    best_param = best_params_info["Params"]
    print(f"\n{'─'*60}")
    print(f"[Phase 2] Final Evaluation – {model_name}")
    print(f"  Best Param: {best_param}")
    print(f"  Seeds: {seeds_for_evaluation}  |  MC Samples={mc_samples}")
    print(f"{'─'*60}")

    detector = build_detector(model_class, best_param, chain, device)

    try:
        avg_auc, std_auc, avg_ap, std_ap, avg_unc, std_unc = run_mc_model(
            detector, global_data,
            seeds=seeds_for_evaluation,
            mc_samples=mc_samples
        )

        time_str = datetime.now().strftime("%Y:%m:%d_%H:%M:%S")
        print(
            f"{time_str} --> [Final] {model_name} "
            f"Avg AUC={avg_auc:.4f} (±{std_auc:.4f})  "
            f"Avg AP={avg_ap:.4f} (±{std_ap:.4f})  "
            f"Avg Unc={avg_unc:.4f} (±{std_unc:.4f})"
        )

        final_results_acc.append({
            "Phase":           "FinalEval",
            "Timestamp":       time_str,
            "Dataset":         chain,
            "Model":           model_name,
            "Best Params":     str(best_param),
            "Val AUC":         round(avg_auc, 4),
            "Val AUC Std Dev": round(std_auc, 4),
            "Val AP":          round(avg_ap,  4),
            "Val AP Std Dev":  round(std_ap,  4),
            "Val Uncertainty": round(avg_unc, 4),
            "Val Unc Std Dev": round(std_unc, 4),
        })

    except Exception as e:
        print(f"  [ERROR] Final eval failed for {model_name}: {e}")

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

    cfg    = Args()
    chain  = cli_args.chain
    device = cfg.device

    file_path         = f'../../_data/GoG/{chain}/graphs'
    global_edges_path = f'../../_data/GoG/edges/{chain}/{chain}_train_edges.txt'

    json_files = sorted(
        Path(file_path).glob('*.json'), key=lambda f: int(f.stem)
    )
    num_nodes = len(json_files)
    print(f"Loading features and labels from {num_nodes} JSON files...")

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
    with open(global_edges_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                src, dst = parts[0], parts[1]
                if src in contract_to_idx and dst in contract_to_idx:
                    edges.append([contract_to_idx[src], contract_to_idx[dst]])

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    if edge_index.numel() > 0:
        edge_index = coalesce(edge_index)

    train_mask, val_mask, test_mask = create_masks(num_nodes)

    global_data            = Data(x=features, edge_index=edge_index, y=labels)
    global_data.train_mask = train_mask
    global_data.val_mask   = val_mask
    global_data.test_mask  = test_mask
    global_data.chain      = chain

    hyperparameters = [
        {"hid_dim": d, "lr": lr, "epoch": e}
        for d  in [8, 16, 32]
        for lr in [0.005, 0.01, 0.1]
        for e  in [50, 100, 150]
    ]

    seeds_for_evaluation = [42, 100, 2026]

    models = {
        'DOMINANT':   DOMINANT,
        'DONE':       DONE,
        'GAE':        GAE,
        'AnomalyDAE': AnomalyDAE,
        'CoLA':       CoLA,
    }

    RESULT_PATH    = f"../../_data/results/fraud_detection_mc"
    checkpoint_dir = f"{RESULT_PATH}/checkpoints"
    os.makedirs(RESULT_PATH, exist_ok=True)

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
            device               = device,
            mc_samples           = cfg.mc_samples,
            checkpoint_dir       = checkpoint_dir,
            results_acc          = all_results,
            final_results_acc    = all_final_results,
        )

    results_df       = pd.DataFrame(all_results)
    final_results_df = pd.DataFrame(all_final_results)

    if not results_df.empty:
        results_df.to_csv(
            f'{RESULT_PATH}/mc-main-results_{chain}_raw.csv', index=False
        )
    if not final_results_df.empty:
        final_results_df.to_csv(
            f'{RESULT_PATH}/mc-main-results_{chain}_final.csv', index=False
        )

    print(f"\n✅ MC 최종 평가 결과가 {RESULT_PATH} 에 저장되었습니다.")


if __name__ == "__main__":
    main()
