# example_usage.py

import sys
from pathlib import Path
from torch.utils.data import DataLoader

# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
ROOT = Path(__file__).resolve().parent.parent
# ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
from data.dataset import HierarchicalDataset
from data.collate import hierarchical_collate_fn

dataset = HierarchicalDataset(
    data_dir="../../_data/GoG/polygon/graphs",
    contract_graph_path="../../_data/GoG/polygon/polygon_hybrid_graph.pt",
    split="train",
    split_seed=42,
    edge_dropout=0.0,
)

print(dataset.get_metadata())

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=hierarchical_collate_fn,
)

batch = next(iter(loader))

print("local_batch.x.shape         =", batch["local_batch"].x.shape)
print("local_batch.edge_index.shape=", batch["local_batch"].edge_index.shape)
print("local_batch.edge_attr.shape =", batch["local_batch"].edge_attr.shape)
print("contract_ids.shape          =", batch["contract_ids"].shape)
print("labels.shape                =", batch["labels"].shape)
print("global_edge_index.shape     =", batch["global_edge_index"].shape)
print("global_features.shape       =", batch["global_features"].shape)
