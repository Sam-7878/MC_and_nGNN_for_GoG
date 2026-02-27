import os
import torch
import numpy as np
import pandas as pd
import random
from pygod.detector import DOMINANT, DONE, GAE, AnomalyDAE, CoLA
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from utils import hierarchical_graph_reader
import warnings
import json
from pathlib import Path
import argparse

warnings.filterwarnings("ignore", message=".*pyg-lib.*")
warnings.filterwarnings("ignore", message=".*transductive only.*")
warnings.filterwarnings("ignore", message=".*Backbone and num_layers.*") 

def create_masks(num_nodes):
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    train_size = int(num_nodes * 0.8)
    val_size = int(num_nodes * 0.1)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    return train_mask, val_mask, test_mask

def eval_roc_auc(label, score):
    roc_auc = roc_auc_score(y_true=label, y_score=score)
    # PyGOD 특성상 이상 점수가 반대로 나오는 경우 보정
    if roc_auc < 0.5:
        score = [1 - s for s in score]
        roc_auc = roc_auc_score(y_true=label, y_score=score)
    return roc_auc

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_model(detector, data, seeds, mask=None):
    auc_scores = []
    ap_scores = []
    f1_scores = []
    
    for seed in seeds:
        set_seed(seed)
        
        # 모델 학습 (PyGOD는 비지도 학습이므로 전체 그래프 구조를 입력)
        detector.fit(data)

        # 예측: pred는 임계값이 적용된 이진 분류 라벨(0 정상, 1 이상), score는 연속적인 이상 점수
        pred, score, _, _ = detector.predict(data, return_pred=True, return_score=True, return_prob=True, return_conf=True)
        
        # Mask가 제공된 경우 (Val 또는 Test) 해당 노드만 추출하여 평가
        if mask is not None:
            y_true = data.y[mask].cpu().numpy()
            y_score = score[mask].cpu().numpy()
            y_pred = pred[mask].cpu().numpy()
        else:
            y_true = data.y.cpu().numpy()
            y_score = score.cpu().numpy()
            y_pred = pred.cpu().numpy()
        
        # 평가 지표 계산
        auc_score = eval_roc_auc(y_true, y_score)
        ap_score = average_precision_score(y_true, y_score)
        # 소수 클래스(이상 거래) 탐지 성능 확인을 위한 F1-score
        f1 = f1_score(y_true, y_pred, zero_division=0) 

        auc_scores.append(auc_score)
        ap_scores.append(ap_score)
        f1_scores.append(f1)

    return (np.mean(auc_scores), np.std(auc_scores), 
            np.mean(ap_scores), np.std(ap_scores),
            np.mean(f1_scores), np.std(f1_scores))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=str, default='polygon')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
    return parser.parse_args()

def main():
    args = get_args()
    chain = args.chain

    # =========================================================================
    # ✅ 메모리 최적화: GraphDatasetGenerator 대신 JSON 초경량 파싱 (Lazy Loading) 적용
    # =========================================================================
    graphs_dir = Path(f"../../_data/GoG/{chain}/graphs/")
    num_nodes = len(list(graphs_dir.glob("*.json")))
    
    x_list = []
    y_list = []
    
    print(f"Loading features and labels from {num_nodes} JSON files (Lightweight Lazy Mode)...")
    
    for idx in range(num_nodes):
        with open(graphs_dir / f"{idx}.json", 'r') as f:
            data = json.load(f)
            # 수많은 로컬 엣지 데이터는 무시하고, Global 노드 피처(contract_feature)와 정답(label)만 추출
            x_list.append(data.get('contract_feature', []))
            y_list.append(data.get('label', 0))
            
    x = torch.tensor(x_list, dtype=torch.float)
    y = torch.tensor(y_list, dtype=torch.long)
    # =========================================================================

    hierarchical_graph = hierarchical_graph_reader(
        f"../../_data/GoG/{chain}/edges/global_edges.csv"
    )

    edge_index = torch.LongTensor(list(hierarchical_graph.edges)).t().contiguous()

    # 모든 노드에 self-loop 추가 (고립 노드 포함)
    self_loops = torch.arange(num_nodes, dtype=torch.long)
    self_edge_index = torch.stack([self_loops, self_loops], dim=0)
    edge_index = torch.cat([edge_index, self_edge_index], dim=1)

    # 중복/정렬 정리
    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    global_data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=num_nodes,
    )

    print("num_nodes:", global_data.num_nodes)
    print("edge_index min/max:", int(global_data.edge_index.min()), int(global_data.edge_index.max()))

    # Mask 생성 및 할당
    train_mask, val_mask, test_mask = create_masks(global_data.num_nodes)
    global_data.train_mask = train_mask
    global_data.val_mask = val_mask
    global_data.test_mask = test_mask

    model_params = {
        "DOMINANT": [{"hid_dim": d, "lr": lr, "epoch": e} for d in [16, 32] for lr in [0.01, 0.005] for e in [50, 100]],
        "DONE": [{"hid_dim": d, "lr": lr, "epoch": e} for d in [16, 32] for lr in [0.01, 0.005] for e in [50, 100]],
        "GAE": [{"hid_dim": d, "lr": lr, "epoch": e} for d in [16, 32] for lr in [0.01, 0.005] for e in [50, 100]],
        "AnomalyDAE": [{"hid_dim": d, "lr": lr, "epoch": e} for d in [16, 32] for lr in [0.01, 0.005] for e in [50, 100]],
        "CoLA": [{"hid_dim": d, "lr": lr, "epoch": e} for d in [16, 32] for lr in [0.01, 0.005] for e in [50, 100]],
    }

    MODEL_MAP = {
        "DOMINANT": DOMINANT,
        "DONE": DONE,
        "GAE": GAE,
        "AnomalyDAE": AnomalyDAE,
        "CoLA": CoLA,
    }

    def build_detector(model_name: str, param: dict):
        ModelCls = MODEL_MAP[model_name]
        return ModelCls(
            hid_dim=param["hid_dim"],
            num_layers=2,
            epoch=param["epoch"],
            lr=param["lr"],
            gpu=args.gpu,
        )

    seed_for_param_selection = 42
    best_model_params = {}

    print("\n--- Hyperparameter Search (using Validation Set) ---")
    for model_name, param_list in model_params.items():
        for param in param_list:
            detector = build_detector(model_name, param)
            
            # 파라미터 튜닝 시에는 val_mask를 사용
            avg_auc, std_auc, avg_ap, std_ap, avg_f1, std_f1 = run_model(
                detector, global_data, [seed_for_param_selection], mask=global_data.val_mask
            )

            # AUC를 기준으로 최고 성능 모델 선정 (필요 시 F1으로 변경 가능)
            if (model_name not in best_model_params) or (
                avg_auc > best_model_params[model_name].get("Best AUC", 0)
            ):
                best_model_params[model_name] = {
                    "Best AUC": avg_auc,
                    "AUC Std Dev": std_auc,
                    "Best AP": avg_ap,
                    "AP Std Dev": std_ap,
                    "Best F1": avg_f1,
                    "F1 Std Dev": std_f1,
                    "Params": param,
                }

            print(
                f"Tested {model_name} with {param}: "
                f"Val AUC={avg_auc:.4f}, Val F1={avg_f1:.4f}, Val AP={avg_ap:.4f}"
            )

    seeds_for_evaluation = [42, 43, 44]
    final_results = []

    print("\n--- Final Evaluation (using Test Set) ---")
    for model_name, stats in best_model_params.items():
        param = stats["Params"]
        detector = build_detector(model_name, param)
        
        # 최종 평가는 test_mask를 사용
        avg_auc, std_auc, avg_ap, std_ap, avg_f1, std_f1 = run_model(
            detector, global_data, seeds_for_evaluation, mask=global_data.test_mask
        )
        
        print(
            f"[{model_name}] Test AUC: {avg_auc:.4f}±{std_auc:.4f} | "
            f"Test F1: {avg_f1:.4f}±{std_f1:.4f} | Test AP: {avg_ap:.4f}±{std_ap:.4f}"
        )
        
        # CSV 저장을 위한 데이터 정리
        final_results.append({
            "Dataset": chain,
            "Model": model_name,
            "Best Params": str(param),
            "Test AUC": round(avg_auc, 4),
            "Test F1": round(avg_f1, 4),
            "Test AP": round(avg_ap, 4),
            "Uncertainty": "N/A" # 베이스라인은 불확실성 지표가 없으므로 N/A 처리
        })

    # 결과를 DataFrame으로 변환 후 CSV로 저장 (Batch Script 연동 목적)
    results_df = pd.DataFrame(final_results)

    RESULT_PATH = f"../../_data/results/fraud_detection"
    os.makedirs(RESULT_PATH, exist_ok=True)
    csv_path = f"{RESULT_PATH}/baseline_{chain}_log.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n✅ 최종 베이스라인 평가 결과가 {csv_path} 에 저장되었습니다.")

if __name__ == "__main__":
    main()