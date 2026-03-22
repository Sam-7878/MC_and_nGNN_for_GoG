## 검증 스크립트: 캐시된 그래프 데이터에 NaN이 포함되어 있는지 확인
from gog_fraud.data.io.dataset import DatasetConfig, FraudDataset
import torch

cfg = DatasetConfig(
    transactions_root="../_data/dataset/transactions",
    labels_path="../_data/dataset/labels.csv",
    global_graph_root="../_data/dataset/global_graph",
    cache_root="../_data/dataset/.cache/graphs",
    chain="polygon",
    load_global_graph=False,
)
ds = FraudDataset(cfg).load()

nan_count = 0
for tg in ds.transaction_graphs:
    g = tg.graph
    if torch.isnan(g.x).any() or (g.edge_attr is not None and torch.isnan(g.edge_attr).any()):
        nan_count += 1

print(f"Total graphs   : {len(ds.transaction_graphs)}")
print(f"NaN graphs     : {nan_count}")
print(f"Label coverage : {len(ds.labels)}")
print(f"Overlap        : {len(set(tg.contract_id for tg in ds.transaction_graphs) & set(ds.labels.keys()))}")

# 첫 5개 상세 출력
