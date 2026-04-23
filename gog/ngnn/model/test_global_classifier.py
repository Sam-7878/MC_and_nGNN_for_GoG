import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# test_global_classifier.py

# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
ROOT = Path(__file__).resolve().parent.parent
# ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from data.dataset import HierarchicalDataset
from data.collate import hierarchical_collate_fn
from model.local_encoder import LocalEncoder
from model.pooling import ContractPooling
from model.global_encoder import GlobalEncoder
from model.classifier import FraudClassifier


def main():
    HIDDEN_DIM = 64
    DEVICE = "cpu"

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
        batch_size=8,
        shuffle=False,
        collate_fn=hierarchical_collate_fn,
    )

    batch = next(iter(loader))
    local_batch = batch["local_batch"].to(DEVICE)
    contract_ids = batch["contract_ids"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    global_edge_index = batch["global_edge_index"].to(DEVICE)
    global_features = batch["global_features"].to(DEVICE)

    N_contracts = global_features.size(0)
    B = contract_ids.size(0)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    local_encoder = LocalEncoder(
        node_dim=dataset.node_dim,
        edge_dim=dataset.edge_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
        heads=4,
        dropout=0.3,
    ).to(DEVICE)

    pooling = ContractPooling(
        hidden_dim=HIDDEN_DIM,
        mode="mean",
    ).to(DEVICE)

    # feat_proj: fuse contract_emb + global_features[contract_ids]
    feat_proj = torch.nn.Linear(
        HIDDEN_DIM + dataset.global_feat_dim,
        HIDDEN_DIM,
    ).to(DEVICE)

    global_encoder = GlobalEncoder(
        hidden_dim=HIDDEN_DIM,
        num_layers=2,
        heads=2,
        dropout=0.3,
    ).to(DEVICE)

    classifier = FraudClassifier(
        hidden_dim=HIDDEN_DIM,
        num_classes=2,
        dropout=0.3,
    ).to(DEVICE)

    # ------------------------------------------------------------------
    # Forward pass (manual hierarchical_gnn.py 로직 미리 검증)
    # ------------------------------------------------------------------

    # Step 1. Local encoding
    tx_emb = local_encoder(
        x=local_batch.x,
        edge_index=local_batch.edge_index,
        edge_attr=local_batch.edge_attr,
        batch=local_batch.batch,
    )
    print(f"tx_emb.shape              = {tx_emb.shape}")   # [total tx nodes, H]

    # Step 2. Contract pooling
    contract_emb = pooling(tx_emb, local_batch.batch)
    print(f"contract_emb.shape        = {contract_emb.shape}")  # [B, H]

    # Step 3. Fuse with global static features
    global_feat_batch = global_features[contract_ids]           # [B, Fg]
    fused = torch.cat([contract_emb, global_feat_batch], dim=-1)  # [B, H+Fg]
    fused = feat_proj(fused)                                    # [B, H]
    print(f"fused.shape               = {fused.shape}")         # [B, H]

    # Step 4. Inject batch contract embeddings into full global_x
    global_x = global_features.new_zeros((N_contracts, HIDDEN_DIM))  # [N, H]
    global_x[contract_ids] = fused.detach()

    # Step 5. Global encoding over full contract graph
    global_out = global_encoder(global_x, global_edge_index)   # [N, H]
    print(f"global_out.shape          = {global_out.shape}")    # [N, H]

    # Step 6. Slice batch contracts and classify
    batch_out = global_out[contract_ids]                        # [B, H]
    logits = classifier(batch_out)                             # [B, 2]
    print(f"batch_out.shape           = {batch_out.shape}")     # [B, H]
    print(f"logits.shape              = {logits.shape}")        # [B, 2]

    # ------------------------------------------------------------------
    # Assertions
    # ------------------------------------------------------------------
    assert tx_emb.dim() == 2
    assert contract_emb.size(0) == B
    assert fused.size(0) == B
    assert global_out.size(0) == N_contracts
    assert global_out.size(1) == HIDDEN_DIM
    assert batch_out.size(0) == B
    assert logits.size(0) == B
    assert logits.size(1) == 2

    print(f"\n✅ GlobalEncoder + FraudClassifier MVP test passed.")
    print(f"   B={B}, N={N_contracts}, H={HIDDEN_DIM}")


if __name__ == "__main__":
    main()
