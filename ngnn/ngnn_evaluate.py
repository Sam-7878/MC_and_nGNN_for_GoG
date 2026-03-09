import torch
import numpy as np
import pandas as pd
import os
import gc
import warnings
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from torch.utils.data import DataLoader
from hierarchical_gnn import GlobalGNN

# 구현하신 Dataset 및 Model 임포트
from hierarchical_dataset import HierarchicalDataset, hierarchical_collate_fn
from hierarchical_dataset_mc import HierarchicalDatasetMC
from hierarchical_gnn import * # 업로드하신 nGNN 모델 클래스 임포트

warnings.filterwarnings("ignore")

def evaluate_ngnn(model, dataloader, device, use_mc=False, mc_samples=10):
    model.eval()
    all_scores, all_preds, all_labels = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            local_batch = batch['local_batch'].to(device)
            global_edge_index = batch['global_edge_index'].to(device)
            # node feature나 edge attr가 있다면 모델 입력에 맞게 조정하세요
            
            labels = batch['labels'].numpy()
            
            if use_mc:
                # 💡 모델 레벨 MC-Dropout 적용 (hierarchical_gnn.py의 메서드 활용)
                # 입력 인자는 구현하신 모델의 forward 시그니처에 맞게 수정 필요
                logits_stack = model.forward_with_mcdropout(
                    local_batch, global_edge_index, n_samples=mc_samples
                )
                # (n_samples, batch_size, num_classes) -> 평균 계산
                mean_logits = logits_stack.mean(dim=0)
                probs = torch.softmax(mean_logits, dim=1)[:, 1] # 클래스 1의 확률
            else:
                # 💡 Base nGNN 일반 예측
                logits = model(local_batch, global_edge_index)
                probs = torch.softmax(logits, dim=1)[:, 1]
                
            scores = probs.cpu().numpy()
            preds = (scores >= 0.5).astype(int)
            
            all_scores.extend(scores)
            all_preds.extend(preds)
            all_labels.extend(labels)
            
    auc = roc_auc_score(all_labels, all_scores)
    ap = average_precision_score(all_labels, all_scores)
    f1 = f1_score(all_labels, all_preds)
    
    return auc, ap, f1

def main(chain='polygon', use_mc=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Evaluation on {chain.upper()} (MC Mode: {use_mc})")
    
    data_dir = f"../../_data/GoG/{chain}/local_graphs"
    contract_graph_path = f"../../_data/GoG/{chain}/{chain}_hybrid_graph.pt"
    
    # 1. Dataset 로딩 분기 (데이터 레벨 MC)
    if use_mc:
        dataset = HierarchicalDatasetMC(data_dir, contract_graph_path, split='test', mc_edge_dropout=0.1)
        version_name = "MC nGNN (Data+Model)"
    else:
        dataset = HierarchicalDataset(data_dir, contract_graph_path, split='test')
        version_name = "Base nGNN"

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=hierarchical_collate_fn)
    
    # 2. 모델 초기화 및 가중치 로드
    # 💡 실제 구현하신 메인 클래스명으로 변경하고 학습된 가중치(.pt)를 로드하세요.
    model = GlobalGNN(node_dim=..., edge_dim=..., hidden_dim=64).to(device)
    model.load_state_dict(torch.load(f'../../_data/models/best_ngnn_{chain}.pt'))
    
    # [!] 아래는 실행을 위한 더미(Dummy) 처리입니다. 실제 모델 코드로 주석을 해제하고 교체하세요.
    # class DummyModel:
    #     def eval(self): pass
    #     def __call__(self, *args): return torch.randn(len(args[0].batch.unique()), 2)
    #     def forward_with_mcdropout(self, *args, **kwargs): 
    #         return torch.stack([torch.randn(len(args[0].batch.unique()), 2) for _ in range(kwargs.get('n_samples', 10))])
    # model = DummyModel()
    
    # 3. 평가 수행
    auc, ap, f1 = evaluate_ngnn(model, dataloader, device, use_mc=use_mc)
    
    # 4. 결과 저장
    result = [{
        "Timestamp": datetime.now().strftime("%Y:%m:%d_%H:%M:%S"),
        "Dataset": chain,
        "Model": version_name,
        "Test AUC": round(auc, 4),
        "Test AP": round(ap, 4),
        "Test F1": round(f1, 4)
    }]
    
    RESULT_PATH = f"../../_data/results/fraud_detection_ngnn"
    os.makedirs(RESULT_PATH, exist_ok=True)
    csv_file = f'{RESULT_PATH}/ngnn_comparison_{chain}.csv'
    
    df = pd.DataFrame(result)
    if not os.path.exists(csv_file):
        df.to_csv(csv_file, index=False)
    else:
        df.to_csv(csv_file, mode='a', header=False, index=False)
        
    print(f"✅ Results for {version_name}: AUC={auc:.4f}, AP={ap:.4f}, F1={f1:.4f}")
    print(f"📁 Saved to {csv_file}\n")

if __name__ == "__main__":
    # 두 버전 연속 평가 실행
    main(chain='polygon', use_mc=False)
    main(chain='polygon', use_mc=True)