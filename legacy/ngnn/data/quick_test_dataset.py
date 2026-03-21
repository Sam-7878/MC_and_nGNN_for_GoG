# quick_test_dataset.py

import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
ROOT = Path(__file__).resolve().parent.parent
# ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from data.dataset import HierarchicalDataset

dataset = HierarchicalDataset(
    data_dir="../../_data/GoG/polygon/graphs",
    contract_graph_path="../../_data/GoG/polygon/polygon_hybrid_graph.pt",
    split="train",
)

print("len(dataset):", len(dataset))
print("node_dim:", dataset.node_dim)
print("edge_dim:", dataset.edge_dim)
print("global_feat_dim:", dataset.global_feat_dim)
print("global_edge_index.shape:", dataset.global_edge_index.shape)
print("global_features.shape:", dataset.global_features.shape)

sample = dataset[0]
print("sample keys:", sample.keys())
print("contract_id:", sample["contract_id"])
print("contract_name:", sample["contract_name"])
print("label:", sample["label"])
print("local_graph:", sample["local_graph"])
print("x finite:", sample["local_graph"].x.isfinite().all().item())
print("edge_attr finite:", sample["local_graph"].edge_attr.isfinite().all().item())
print("global_features finite:", dataset.global_features.isfinite().all().item())
