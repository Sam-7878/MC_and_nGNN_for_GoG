# test_hierarchical_gnn.py

from xml.parsers.expat import model

import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
ROOT = Path(__file__).resolve().parent.parent
# ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))


from data.dataset import HierarchicalDataset
from data.collate import hierarchical_collate_fn
from model.hierarchical_gnn import HierarchicalGNN


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HIDDEN_DIM = 64
    BATCH_SIZE = 8

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    dataset = HierarchicalDataset(
        data_dir="../../../_data/GoG/polygon/graphs",
        contract_graph_path="../../../_data/GoG/polygon/polygon_hybrid_graph.pt",
        split="train",
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=hierarchical_collate_fn,
    )

    batch = next(iter(loader))
    batch = {
        k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    # PyG Batch object는 별도로 이동
    batch["local_batch"] = batch["local_batch"].to(DEVICE)

    B = batch["contract_ids"].size(0)
    N = batch["global_features"].size(0)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = HierarchicalGNN(
        node_dim=dataset.node_dim,
        edge_dim=dataset.edge_dim,
        hidden_dim=HIDDEN_DIM,
        global_feat_dim=dataset.global_feat_dim,
        num_classes=2,
        num_local_layers=2,
        num_global_layers=2,
        local_heads=4,
        global_heads=2,
        dropout=0.3,
        pooling_mode="mean",
        use_global_gnn=True,
    ).to(DEVICE)

    # ------------------------------------------------------------------
    # Model summary
    # ------------------------------------------------------------------
    info = model.get_model_info()
    print("=" * 50)
    print("Model Info")
    print("=" * 50)
    for k, v in info.items():
        print(f"  {k:<25} = {v}")
    print()

    # ------------------------------------------------------------------
    # Forward pass (eval)
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        logits = model(batch)

    print("=" * 50)
    print("Forward Pass (eval)")
    print("=" * 50)
    print(f"  B (batch size)           = {B}")
    print(f"  N (total contracts)      = {N}")
    print(f"  logits.shape             = {logits.shape}")
    print(f"  logits sample:\n{logits}")
    print()

    # ------------------------------------------------------------------
    # Forward pass (train mode → dropout active)
    # ------------------------------------------------------------------
    model.train()
    logits_train = model(batch)

    print("=" * 50)
    print("Forward Pass (train)")
    print("=" * 50)
    print(f"  logits_train.shape       = {logits_train.shape}")
    print()

    # ------------------------------------------------------------------
    # Loss computation
    # ------------------------------------------------------------------
    labels = batch["labels"]
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits_train, labels)

    print("=" * 50)
    print("Loss")
    print("=" * 50)
    print(f"  labels                   = {labels}")
    print(f"  loss                     = {loss.item():.4f}")
    print()

    # ------------------------------------------------------------------
    # Backward pass (gradient flow check)
    # ------------------------------------------------------------------
    loss.backward()

    grad_ok = all(
        p.grad is not None
        for p in model.parameters()
        if p.requires_grad
    )

    print("=" * 50)
    print("Gradient Check")
    print("=" * 50)
    print(f"  All gradients computed   = {grad_ok}")
    print()

    # ------------------------------------------------------------------
    # encode() 검증 (pre-classifier embedding)
    # ------------------------------------------------------------------
    model.eval()
    with torch.no_grad():
        emb = model.encode(batch)

    print("=" * 50)
    print("encode() (pre-classifier embedding)")
    print("=" * 50)
    print(f"  emb.shape                = {emb.shape}")
    print()

    # ------------------------------------------------------------------
    # Ablation: use_global_gnn=False
    # ------------------------------------------------------------------
    model_no_global = HierarchicalGNN(
        node_dim=dataset.node_dim,
        edge_dim=dataset.edge_dim,
        hidden_dim=HIDDEN_DIM,
        global_feat_dim=dataset.global_feat_dim,
        num_classes=2,
        use_global_gnn=False,
    ).to(DEVICE)

    model_no_global.eval()
    with torch.no_grad():
        logits_no_global = model_no_global(batch)

    print("=" * 50)
    print("Ablation: use_global_gnn=False")
    print("=" * 50)
    print(f"  logits_no_global.shape   = {logits_no_global.shape}")
    print()

    # ------------------------------------------------------------------
    # Final assertions
    # ------------------------------------------------------------------
    assert logits.shape == (B, 2), \
        f"Expected logits shape ({B}, 2), got {logits.shape}"
    assert logits_train.shape == (B, 2)
    assert emb.shape[0] == B
    assert logits_no_global.shape == (B, 2)
    assert grad_ok, "Some parameters have no gradient!"

    print("✅ HierarchicalGNN MVP test PASSED.")
    print(f"   B={B}, N={N}, H={HIDDEN_DIM}")


if __name__ == "__main__":
    main()
