import json

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from utils import hierarchical_graph_reader
from pygod.detector import DOMINANT, DONE, GAE, AnomalyDAE, CoLA
from pygod.metric import eval_roc_auc
from sklearn.metrics import average_precision_score, roc_auc_score
import os
import pandas as pd
import random
import warnings
import multiprocessing
from tqdm import tqdm
from datetime import datetime

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

def load_labels(filepath, column_name='label'):
    try:
        labels = pd.read_csv(filepath)[column_name].values
        return torch.tensor(labels, dtype=torch.long)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        exit()
    except KeyError:
        print(f"Error: Column {column_name} does not exist in the file.")
        exit()

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=str, default='bsc')
    parser.add_argument('--device', type=int, default=0, help='GPU index to use')
    parser.add_argument('--workers', type=int, default=-1, help='Number of multiprocessing workers')
    return parser.parse_args()


# =========================================================================
# ✅ 멀티프로세싱 워커: Numpy 기반으로 읽고 평균(Mean) 계산 후 반환
# =========================================================================
def load_embedding_worker(args):
    idx, embedding_file = args
    try:
        if os.path.exists(embedding_file):
            arr = np.load(embedding_file)
            if arr.size == 0:
                return idx, None
            # 노드 단위 평균을 계산 (Numpy 연산이 가장 빠름)
            mean_emb = arr.mean(axis=0, keepdims=True)
            return idx, mean_emb
        else:
            return idx, None
    except Exception as e:
        print(f"Error reading {embedding_file}: {e}")
        return idx, None


def main():
    args = get_args()
    chain = args.chain
    
    filepath = f'../../_data/dataset/features/{chain}_basic_metrics_processed.csv'
    y = load_labels(filepath)
    num_nodes = len(y)
    
    embedding_path = f'../../_data/dataset/Deepwalk/{chain}'
    
    # 미리 크기가 정해진 리스트 생성
    graph_embeddings = [None] * num_nodes
    tasks = [(idx, os.path.join(embedding_path, f'{idx}.npy')) for idx in range(num_nodes)]
    
    n_workers = args.workers if args.workers > 0 else max(2, multiprocessing.cpu_count() // 2)
    print(f"Loading {num_nodes} .npy files using {n_workers} CPU cores...")

    valid_dim = None
    missing_indices = []

    # =========================================================================
    # ✅ 멀티프로세싱 병렬 로딩 적용
    # =========================================================================
    with multiprocessing.Pool(processes=n_workers) as pool:
        for idx, mean_emb in tqdm(pool.imap_unordered(load_embedding_worker, tasks, chunksize=40), total=num_nodes, desc="Loading DeepWalk embeddings"):
            if mean_emb is not None:
                graph_embeddings[idx] = torch.tensor(mean_emb, dtype=torch.float32)
                if valid_dim is None:
                    valid_dim = mean_emb.shape[1]
            else:
                missing_indices.append(idx)

    if valid_dim is None:
        raise ValueError("No valid embeddings were processed. Please check the embedding path.")

    # 결측치(Missing files) 처리 - 이전 코드의 Size Mismatch 에러 방지
    for idx in missing_indices:
        print(f"Embedding file not found or empty for index: {idx}, padding with zeros.")
        graph_embeddings[idx] = torch.zeros((1, valid_dim), dtype=torch.float32)

    x = torch.cat(graph_embeddings, dim=0)
    print(f"Feature matrix shape: {x.shape}")
    print(f"Label vector shape: {y.shape}")
    # =========================================================================

    hierarchical_graph = hierarchical_graph_reader(f'../../_data/GoG/{chain}/edges/global_edges.csv')
    edge_index = torch.LongTensor(list(hierarchical_graph.edges)).t().contiguous()

    # PyG 안정성을 위한 self-loop 추가 및 정렬
    self_loops = torch.arange(num_nodes, dtype=torch.long)
    self_edge_index = torch.stack([self_loops, self_loops], dim=0)
    edge_index = torch.cat([edge_index, self_edge_index], dim=1)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    global_data = Data(x=x, edge_index=edge_index, y=y)
    
    train_mask, val_mask, test_mask = create_masks(global_data.num_nodes)
    global_data.train_mask = train_mask
    global_data.val_mask = val_mask
    global_data.test_mask = test_mask

    # Parameters to test
    model_params = {
        "DOMINANT": [{"hid_dim": d, "lr": lr, "epoch": e} for d in [8, 16, 32] for lr in [0.005, 0.01, 0.1] for e in [50, 100, 150]],
        "DONE": [{"hid_dim": d, "lr": lr, "epoch": e} for d in [8, 16, 32] for lr in [0.005, 0.01, 0.1] for e in [50, 100, 150]],
        "GAE": [{"hid_dim": d, "lr": lr, "epoch": e} for d in [8, 16, 32] for lr in [0.005, 0.01, 0.1] for e in [50, 100, 150]],
        "AnomalyDAE": [{"hid_dim": d, "lr": lr, "epoch": e} for d in [8, 16, 32] for lr in [0.005, 0.01,  0.1] for e in [50, 100, 150]],
        "CoLA": [{"hid_dim": d, "lr": lr, "epoch": e} for d in [8, 16, 32] for lr in [0.005, 0.01, 0.1] for e in [50, 100, 150]],
    }

    # eval() 대신 안전한 모델 매핑 사용
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
    completed_tasks = set()

    results = []
    final_results = []
    
    checkpoint_dir = f"../../_data/results/fraud_detection"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = f"{checkpoint_dir}/{chain}_main-deepwalk_checkpoint.json"

    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                ckpt_data = json.load(f)
                best_model_params = ckpt_data.get("best_model_params", {})
                completed_tasks = set(ckpt_data.get("completed_tasks", []))
            print(f"\n[Checkpoint Loaded] Resuming from {len(completed_tasks)} completed steps...")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")

    for model_name, param_list in model_params.items():
        for param in param_list:
            task_id = f"{model_name}_{param['hid_dim']}_{param['lr']}_{param['epoch']}"
            
            if task_id in completed_tasks:
                continue

            detector = build_detector(model_name, param)

            avg_auc, std_auc, avg_ap, std_ap = run_model(detector, global_data, [seed_for_param_selection])
            if model_name not in best_model_params or avg_auc > best_model_params[model_name].get('Best AUC', 0):
                best_model_params[model_name] = {
                    "Best AUC": avg_auc,
                    "AUC Std Dev": std_auc,
                    "Best AP": avg_ap,
                    "AP Std Dev": std_ap,
                    "Params": param
                }

            # 날짜-시각 문자열 (파일명에 안전하도록 초단위까지만)
            time_str = datetime.now().strftime("%Y:%m:%d_%H:%M:%S")
            print(f"{time_str} --> Tested {model_name} with {param}: Avg AUC={avg_auc:.4f}, Std AUC={std_auc:.4f}, Avg AP={avg_ap:.4f}, Std AP={std_ap:.4f}")

            # CSV 저장을 위한 데이터 정리
            results.append({
                "Timestamp": time_str,
                "Dataset": chain,
                "Model": model_name,
                "Best Params": str(param),
                "Val AUC": round(avg_auc, 4),
                "Val AUC Std Dev": round(std_auc, 4),
                "Val AP": round(avg_ap, 4),
                "Val AP Std Dev": round(std_ap, 4),
                # "Uncertainty": "N/A" # 베이스라인은 불확실성 지표가 없으므로 N/A 처리
            })

            completed_tasks.add(task_id)
            with open(checkpoint_file, 'w') as f:
                json.dump({
                    "best_model_params": best_model_params,
                    "completed_tasks": list(completed_tasks)
                }, f, indent=4)

    seeds_for_evaluation = [42, 43, 44]
    for model_name, stats in best_model_params.items():
        param = stats['Params']
        detector = build_detector(model_name, param)
        avg_auc, std_auc, avg_ap, std_ap = run_model(detector, global_data, seeds_for_evaluation)
        print(model_name)
        print(stats)
        # 날짜-시각 문자열 (파일명에 안전하도록 초단위까지만)
        time_str = datetime.now().strftime("%Y:%m:%d_%H:%M:%S")
        print(f'{time_str} --> Final Evaluation for {model_name}: Avg AUC={avg_auc:.4f}, Std AUC={std_auc:.4f}, Avg AP={avg_ap:.4f}, Std AP={std_ap:.4f}')
        
        # CSV 저장을 위한 데이터 정리
        final_results.append({
            "Timestamp": time_str,
            "Dataset": chain,
            "Model": model_name,
            "Best Params": str(param),
            "Val AUC": round(avg_auc, 4),
            "Val AUC Std Dev": round(std_auc, 4),
            "Val AP": round(avg_ap, 4),
            "Val AP Std Dev": round(std_ap, 4),
            # "Uncertainty": "N/A" # 베이스라인은 불확실성 지표가 없으므로 N/A 처리
        })

    RESULT_PATH = f"../../_data/results/fraud_detection"
    # 결과를 DataFrame으로 변환 후 CSV로 저장 (Batch Script 연동 목적)
    results_df = pd.DataFrame(results)
    final_results_df = pd.DataFrame(final_results)
    results_df.to_csv(f"{RESULT_PATH}/main-deepwalk-results_{chain}_log.csv", index=False)
    final_results_df.to_csv(f"{RESULT_PATH}/main-deepwalk-final_{chain}_log.csv", index=False)
    print(f"\n✅ 최종 베이스라인 평가 결과가 {RESULT_PATH} 에 저장되었습니다.")


if __name__ == "__main__":
    main()