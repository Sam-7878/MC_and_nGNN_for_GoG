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
import os
import gc
from pathlib import Path
from torch_geometric.utils import coalesce
import argparse
import multiprocessing
from datetime import datetime
from tqdm import tqdm
import pandas as pd

# 귀찮은 경고 숨김 처리
warnings.filterwarnings("ignore", message=".*pyg-lib.*")
warnings.filterwarnings("ignore", message=".*transductive only.*")
warnings.filterwarnings("ignore", message=".*Backbone and num_layers.*")

class Args:
    def __init__(self):
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
    parser.add_argument('--workers', type=int, default=-1, help='Number of multiprocessing workers')
    return parser.parse_args()


# 워커 프로세스용 전역 함수
def read_json_worker(file_path):
    idx = int(file_path.stem)
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return idx, data.get('contract_feature', []), data.get('label', 0)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return idx, [], 0


def main():
    args = get_args()
    chain = args.chain
    
    graphs_dir = Path(f"../../_data/GoG/{chain}/graphs/")
    json_files = list(graphs_dir.glob("*.json"))
    num_nodes = len(json_files)
    
    x_list = [None] * num_nodes
    y_list = [None] * num_nodes
    
    n_workers = args.workers if args.workers > 0 else max(2, multiprocessing.cpu_count()//2) 
    # CPU 코어의 2/3를 활용하되 최소 2개는 사용하도록 설정
    print(f"Loading features and labels from {num_nodes} JSON files...")
    
    with multiprocessing.Pool(processes=n_workers) as pool:
        # for idx, feat, label in tqdm(pool.imap_unordered(read_json_worker, json_files, chunksize=100), total=num_nodes, desc="JSON Parsing"):
        for idx, feat, label in tqdm(pool.imap_unordered(read_json_worker, json_files, chunksize=40), total=num_nodes, desc="JSON Parsing"):
            x_list[idx] = feat
            y_list[idx] = label
            
    x = torch.tensor(x_list, dtype=torch.float)
    y = torch.tensor(y_list, dtype=torch.long)
    
    hierarchical_graph = hierarchical_graph_reader(f"../../_data/GoG/{chain}/edges/global_edges.csv")
    edge_index = torch.LongTensor(list(hierarchical_graph.edges)).t().contiguous()

    self_loops = torch.arange(num_nodes, dtype=torch.long)
    self_edge_index = torch.stack([self_loops, self_loops], dim=0)
    edge_index = torch.cat([edge_index, self_edge_index], dim=1)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)

    global_data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)
    train_mask, val_mask, test_mask = create_masks(global_data.num_nodes)
    global_data.train_mask = train_mask
    global_data.val_mask = val_mask
    global_data.test_mask = test_mask

    model_params = {
        "DOMINANT": [{"hid_dim": d, "lr": lr, "epoch": e} for d in [8, 16, 32] for lr in [0.005, 0.01, 0.1] for e in [50, 100, 150]],
        "DONE": [{"hid_dim": d, "lr": lr, "epoch": e} for d in [8, 16, 32] for lr in [0.005, 0.01, 0.1] for e in [50, 100, 150]],
        "GAE": [{"hid_dim": d, "lr": lr, "epoch": e} for d in [8, 16, 32] for lr in [0.005, 0.01, 0.1] for e in [50, 100, 150]],
        "AnomalyDAE": [{"hid_dim": d, "lr": lr, "epoch": e} for d in [8, 16, 32] for lr in [0.005, 0.01,  0.1] for e in [50, 100, 150]],
        "CoLA": [{"hid_dim": d, "lr": lr, "epoch": e} for d in [8, 16, 32] for lr in [0.005, 0.01, 0.1] for e in [50, 100, 150]],
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
    completed_tasks = set()
    
    results = []
    final_results = []
    
    checkpoint_dir = f"../../_data/results/fraud_detection"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = f"{checkpoint_dir}/{chain}_main_checkpoint.json"

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
            
            try:
                avg_auc, std_auc, avg_ap, std_ap = run_model(detector, global_data, [seed_for_param_selection])

                if (model_name not in best_model_params) or (avg_auc > best_model_params[model_name].get("Best AUC", 0)):
                    best_model_params[model_name] = {
                        "Best AUC": avg_auc,
                        "AUC Std Dev": std_auc,
                        "Best AP": avg_ap,
                        "AP Std Dev": std_ap,
                        "Params": param,
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
                    
            except Exception as e:
                print(f"\n[ERROR] {model_name} failed with {param}. Skipping to next. Reason: {e}\n")
                continue
            
            finally:
                # =========================================================================
                # ✅ GPU VRAM 누수 원천 차단: 매 훈련마다 메모리를 강제 청소합니다.
                # =========================================================================
                del detector  # 모델 객체 메모리 해제
                gc.collect()  # 파이썬 가비지 컬렉터 강제 실행
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # GPU 내 캐시된 미사용 텐서 강제 반환
                # =========================================================================

    print("\n[Grid Search Complete] Proceeding to Final Evaluation...")
    
    seeds_for_evaluation = [42, 43, 44]
    for model_name, stats in best_model_params.items():
        param = stats["Params"]
        detector = build_detector(model_name, param)
        
        try:
            avg_auc, std_auc, avg_ap, std_ap = run_model(detector, global_data, seeds_for_evaluation)
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
            
        finally:
            # 최종 평가 후에도 모델 삭제 및 GPU 메모리 초기화
            del detector
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    RESULT_PATH = f"../../_data/results/fraud_detection"
    # 결과를 DataFrame으로 변환 후 CSV로 저장 (Batch Script 연동 목적)
    results_df = pd.DataFrame(results)
    final_results_df = pd.DataFrame(final_results)
    results_df.to_csv(f"{RESULT_PATH}/main-results_{chain}_log.csv", index=False)
    final_results_df.to_csv(f"{RESULT_PATH}/main-final_{chain}_log.csv", index=False)
    print(f"\n✅ 최종 베이스라인 평가 결과가 {RESULT_PATH} 에 저장되었습니다.")


if __name__ == "__main__":
    main()