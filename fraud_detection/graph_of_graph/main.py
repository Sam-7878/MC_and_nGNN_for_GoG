import torch
import numpy as np
import random
from pygod.detector import DOMINANT, DONE, GAE, AnomalyDAE, CoLA
from pygod.metric import eval_roc_auc
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, average_precision_score
from utils import hierarchical_graph_reader
import warnings
import json
from pathlib import Path
from torch_geometric.utils import coalesce
import argparse

# 귀찮은 경고 숨김 처리
warnings.filterwarnings("ignore", message=".*pyg-lib.*")
warnings.filterwarnings("ignore", message=".*transductive only.*")
warnings.filterwarnings("ignore", message=".*Backbone and num_layers.*")

class Args:
    def __init__(self):
        # PyGOD는 gpu 인자를 int로 받습니다: 0,1,2... / CPU는 -1
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'GPU' if self.device >= 0 else 'CPU'}")


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


def run_model(detector, data, seeds):
    auc_scores = []
    ap_scores = []
    
    for seed in seeds:
        set_seed(seed)
        detector.fit(data)

        _, score, _, _ = detector.predict(data, return_pred=True, return_score=True, return_prob=True, return_conf=True)
        
        auc_score = eval_roc_auc(data.y, score)
        ap_score = average_precision_score(data.y.cpu().numpy(), score.cpu().numpy())

        auc_scores.append(auc_score)
        ap_scores.append(ap_score)

    return np.mean(auc_scores), np.std(auc_scores), np.mean(ap_scores), np.std(ap_scores)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=str, default='bsc')
    parser.add_argument('--device', type=int, default=0, help='GPU device index (e.g., 0, 1) or -1 for CPU')
    return parser.parse_args()


def main():
    args = get_args()
    chain = args.chain
    
    # =========================================================================
    # ✅ 메모리 폭발 원인 해결: GraphDatasetGenerator 대신 JSON 초경량 파싱 적용
    # =========================================================================
    graphs_dir = Path(f"../../_data/GoG/{chain}/graphs/")
    
    # JSON 파일 개수 파악
    num_nodes = len(list(graphs_dir.glob("*.json")))
    
    x_list = []
    y_list = []
    
    print(f"Loading features and labels from {num_nodes} JSON files (Lightweight Lazy Mode)...")
    
    # 0번부터 순차적으로 필요한 부분만 메모리에 올리고 버림 (OOM 완벽 차단)
    for idx in range(num_nodes):
        with open(graphs_dir / f"{idx}.json", 'r') as f:
            data = json.load(f)
            # 수십~수백만 개의 로컬 edge와 feature 데이터는 버리고 전역 요약본(contract_feature)만 추출
            x_list.append(data.get('contract_feature', []))
            y_list.append(data.get('label', 0))
            
    x = torch.tensor(x_list, dtype=torch.float)
    y = torch.tensor(y_list, dtype=torch.long)
    
    print(f"Feature matrix shape: {x.shape}")
    print(f"Label vector shape: {y.shape}")
    # =========================================================================

    hierarchical_graph = hierarchical_graph_reader(
        f"../../_data/GoG/{chain}/edges/global_edges.csv"
    )

    edge_index = torch.LongTensor(list(hierarchical_graph.edges)).t().contiguous()

    # 모든 노드에 self-loop 추가
    self_loops = torch.arange(num_nodes, dtype=torch.long)
    self_edge_index = torch.stack([self_loops, self_loops], dim=0)
    edge_index = torch.cat([edge_index, self_edge_index], dim=1)

    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    global_data = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        num_nodes=num_nodes,
    )

    print("Total Global Nodes:", global_data.num_nodes)
    print("Edge Index min/max:", int(global_data.edge_index.min()), int(global_data.edge_index.max()))

    train_mask, val_mask, test_mask = create_masks(global_data.num_nodes)
    global_data.train_mask = train_mask
    global_data.val_mask = val_mask
    global_data.test_mask = test_mask

    model_params = {
        "DOMINANT": [{"hid_dim": d, "lr": lr, "epoch": e}
                     for d in [16, 32, 64] for lr in [0.01, 0.005, 0.1] for e in [50, 100, 150]],
        "DONE": [{"hid_dim": d, "lr": lr, "epoch": e}
                 for d in [16, 32, 64] for lr in [0.01, 0.005, 0.1] for e in [50, 100, 150]],
        "GAE": [{"hid_dim": d, "lr": lr, "epoch": e}
                for d in [16, 32, 64] for lr in [0.01, 0.005, 0.1] for e in [50, 100, 150]],
        "AnomalyDAE": [{"hid_dim": d, "lr": lr, "epoch": e}
                       for d in [16, 32, 64] for lr in [0.01, 0.005, 0.1] for e in [50, 100, 150]],
        "CoLA": [{"hid_dim": d, "lr": lr, "epoch": e}
                 for d in [16, 32, 64] for lr in [0.01, 0.005, 0.1] for e in [50, 100, 150]],
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
            gpu=args.device,
        )

    seed_for_param_selection = 42
    best_model_params = {}

    for model_name, param_list in model_params.items():
        for param in param_list:
            detector = build_detector(model_name, param)
            avg_auc, std_auc, avg_ap, std_ap = run_model(
                detector, global_data, [seed_for_param_selection]
            )

            if (model_name not in best_model_params) or (
                avg_auc > best_model_params[model_name].get("Best AUC", 0)
            ):
                best_model_params[model_name] = {
                    "Best AUC": avg_auc,
                    "AUC Std Dev": std_auc,
                    "Best AP": avg_ap,
                    "AP Std Dev": std_ap,
                    "Params": param,
                }

            print(
                f"Tested {model_name} with {param}: "
                f"Avg AUC={avg_auc:.4f}, Std AUC={std_auc:.4f}, "
                f"Avg AP={avg_ap:.4f}, Std AP={std_ap:.4f}"
            )

    seeds_for_evaluation = [42, 43, 44]
    for model_name, stats in best_model_params.items():
        param = stats["Params"]
        detector = build_detector(model_name, param)
        avg_auc, std_auc, avg_ap, std_ap = run_model(
            detector, global_data, seeds_for_evaluation
        )
        print(
            f"Final Evaluation for {model_name}: "
            f"Avg AUC={avg_auc:.4f}, Std AUC={std_auc:.4f}, "
            f"Avg AP={avg_ap:.4f}, Std AP={std_ap:.4f}"
        )


if __name__ == "__main__":
    main()