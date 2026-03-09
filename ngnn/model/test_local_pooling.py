import sys
from pathlib import Path
from torch.utils.data import DataLoader

# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
ROOT = Path(__file__).resolve().parent.parent
# ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))
# test_local_pooling.py

from data.dataset import HierarchicalDataset
from data.collate import hierarchical_collate_fn
from model.local_encoder import LocalEncoder
from model.pooling import ContractPooling


def main():
    dataset = HierarchicalDataset(
        data_dir="../../_data/GoG/polygon/graphs",
        contract_graph_path="../../_data/GoG/polygon/polygon_hybrid_graph.pt",
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=hierarchical_collate_fn,
    )

    batch = next(iter(loader))
    local_batch = batch["local_batch"]

    encoder = LocalEncoder(
        node_dim=dataset.node_dim,
        edge_dim=dataset.edge_dim,
        hidden_dim=64,
        num_layers=2,
        heads=4,
        dropout=0.3,
    )

    pooling = ContractPooling(
        hidden_dim=64,
        mode="mean",
    )

    node_emb = encoder(
        x=local_batch.x,
        edge_index=local_batch.edge_index,
        edge_attr=local_batch.edge_attr,
        batch=local_batch.batch,
    )

    contract_emb = pooling(node_emb, local_batch.batch)

    print("local_batch.x.shape         =", local_batch.x.shape)
    print("local_batch.edge_index.shape=", local_batch.edge_index.shape)
    print("local_batch.edge_attr.shape =", local_batch.edge_attr.shape)
    print("node_emb.shape              =", node_emb.shape)
    print("contract_emb.shape          =", contract_emb.shape)
    print("labels.shape                =", batch["labels"].shape)

    assert node_emb.dim() == 2
    assert contract_emb.dim() == 2
    assert contract_emb.size(0) == batch["labels"].size(0)
    assert contract_emb.size(1) == 64

    print("✅ LocalEncoder + ContractPooling MVP test passed.")


if __name__ == "__main__":
    main()
