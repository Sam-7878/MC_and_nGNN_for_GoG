# model/hierarchical_gnn.py

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.classifier import FraudClassifier
from model.global_encoder import GlobalEncoder
from model.local_encoder import LocalEncoder
from model.pooling import ContractPooling


class HierarchicalGNN(nn.Module):
    """
    GoG + nGNN MVP: Full Hierarchical GNN.

    Assembles LocalEncoder → ContractPooling → GlobalEncoder → FraudClassifier
    into a single end-to-end module.

    Design decisions:
    - Single forward() interface: model(batch) → logits [B, num_classes]
    - Full-graph transductive GlobalEncoder:
        always operates on all N contract nodes,
        caller slices [contract_ids] after.
    - global_feat_proj provides valid initial embeddings for ALL N contracts
        (not just the batch), so GlobalEncoder sees meaningful context
        even for contracts not in the current batch.
    - Phase 1~4 모두 이 파일을 수정하지 않습니다.
        edge_dropout → dataset.py 에서 제어
        MC-Dropout   → mc/mc_dropout.py 에서 제어

    Args:
        node_dim         : local node feature dim (4 in current GoG JSON)
        edge_dim         : local edge feature dim (1, synthesized in dataset.py)
        hidden_dim       : shared embedding dim across all modules
        global_feat_dim  : static contract feature dim (7 in current GoG JSON)
        num_classes      : number of fraud classes (2)
        num_local_layers : GATConv layers in LocalEncoder
        num_global_layers: GATConv layers in GlobalEncoder
        local_heads      : attention heads in LocalEncoder GATConv
        global_heads     : attention heads in GlobalEncoder GATConv
        dropout          : shared dropout rate
        pooling_mode     : "mean" | "attention"
        use_global_gnn   : False → skip GlobalEncoder (ablation용)

    Forward:
        batch: dict = {
            "local_batch"       : PyG Batch,
            "contract_ids"      : LongTensor [B],
            "labels"            : LongTensor [B],
            "global_edge_index" : LongTensor [2, E],
            "global_features"   : FloatTensor [N, Fg],
        }

    Returns:
        logits: FloatTensor [B, num_classes]
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        global_feat_dim: int,
        num_classes: int = 2,
        num_local_layers: int = 2,
        num_global_layers: int = 2,
        local_heads: int = 4,
        global_heads: int = 2,
        dropout: float = 0.3,
        pooling_mode: str = "mean",
        use_global_gnn: bool = True,
    ):
        super().__init__()

        # ------------------------------------------------------------------
        # Validate
        # ------------------------------------------------------------------
        self._validate_args(
            node_dim, edge_dim, hidden_dim, global_feat_dim, num_classes, dropout
        )

        # ------------------------------------------------------------------
        # Save hyperparameters (for checkpoint / inspection)
        # ------------------------------------------------------------------
        self.hidden_dim = hidden_dim
        self.global_feat_dim = global_feat_dim
        self.num_classes = num_classes
        self.use_global_gnn = use_global_gnn

        # ------------------------------------------------------------------
        # Module 1. Local encoder
        # ------------------------------------------------------------------
        self.local_encoder = LocalEncoder(
            node_dim=node_dim,
            edge_dim=edge_dim,
            hidden_dim=hidden_dim,
            num_layers=num_local_layers,
            heads=local_heads,
            dropout=dropout,
            use_residual=True,
        )

        # ------------------------------------------------------------------
        # Module 2. Contract pooling
        # ------------------------------------------------------------------
        self.pooling = ContractPooling(
            hidden_dim=hidden_dim,
            mode=pooling_mode,
            dropout=dropout,
        )

        # ------------------------------------------------------------------
        # Module 3a. Feature fusion projection
        #
        # Fuses pooled local embedding + static contract feature
        # for the B contracts in the current batch.
        #
        # Input : cat(contract_emb [B, H], global_feat [B, Fg]) → [B, H+Fg]
        # Output: fused [B, H]
        # ------------------------------------------------------------------
        self.feat_proj = nn.Sequential(
            nn.Linear(hidden_dim + global_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ------------------------------------------------------------------
        # Module 3b. Global feature projection
        #
        # Projects ALL N contract static features to hidden_dim.
        # This ensures non-batch contracts have a meaningful initial
        # embedding for GlobalEncoder (not just zeros).
        #
        # Input : global_features [N, Fg]
        # Output: global_x [N, H]
        # ------------------------------------------------------------------
        self.global_feat_proj = nn.Sequential(
            nn.Linear(global_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ------------------------------------------------------------------
        # Module 4. Global encoder (optional for ablation)
        # ------------------------------------------------------------------
        if self.use_global_gnn:
            self.global_encoder = GlobalEncoder(
                hidden_dim=hidden_dim,
                num_layers=num_global_layers,
                heads=global_heads,
                dropout=dropout,
                use_residual=True,
            )
        else:
            self.global_encoder = None

        # ------------------------------------------------------------------
        # Module 5. Classifier
        # ------------------------------------------------------------------
        self.classifier = FraudClassifier(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
        )

        # ------------------------------------------------------------------
        # Weight initialization
        # ------------------------------------------------------------------
        self._init_weights()

    # ----------------------------------------------------------------------
    # Forward
    # ----------------------------------------------------------------------
    def forward(self, batch: Dict) -> torch.Tensor:
        """
        End-to-end forward pass.

        Args:
            batch: dict with keys
                "local_batch"       : PyG Batch
                "contract_ids"      : LongTensor [B]
                "global_edge_index" : LongTensor [2, E]
                "global_features"   : FloatTensor [N, Fg]

        Returns:
            logits: FloatTensor [B, num_classes]
        """
        local_batch = batch["local_batch"]
        contract_ids = batch["contract_ids"]
        global_edge_index = batch["global_edge_index"]
        global_features = batch["global_features"]    # [N, Fg]

        # ------------------------------------------------------------------
        # Step 1. Local encoding
        # → tx_emb: [Σtx_nodes, H]
        # ------------------------------------------------------------------
        tx_emb = self.local_encoder(
            x=local_batch.x,
            edge_index=local_batch.edge_index,
            edge_attr=local_batch.edge_attr,
            batch=local_batch.batch,
        )

        # ------------------------------------------------------------------
        # Step 2. Pooling: tx-level → contract-level
        # → contract_emb: [B, H]
        # ------------------------------------------------------------------
        contract_emb = self.pooling(tx_emb, local_batch.batch)

        # ------------------------------------------------------------------
        # Step 3. Fuse with static contract features (batch contracts only)
        # → fused: [B, H]
        # ------------------------------------------------------------------
        global_feat_batch = global_features[contract_ids]            # [B, Fg]
        fused = torch.cat([contract_emb, global_feat_batch], dim=-1)  # [B, H+Fg]
        fused = self.feat_proj(fused)                                  # [B, H]

        # ------------------------------------------------------------------
        # Step 4. Build full global_x [N, H]
        #         a. Project ALL contract static features as base
        #         b. Overwrite batch-contract positions with fused embeddings
        #
        # Why not zeros as base?
        # → Non-batch contracts should still carry meaningful signal
        #   so that GlobalEncoder can propagate useful information
        #   from/to batch contracts via their neighbors.
        # ------------------------------------------------------------------
        global_x = self.global_feat_proj(global_features)     # [N, H]

        # Clone before in-place modification to preserve autograd safety
        global_x = global_x.clone()
        global_x[contract_ids] = fused                        # inject batch

        # ------------------------------------------------------------------
        # Step 5. Global encoding over full contract graph
        # → global_out: [N, H]
        # ------------------------------------------------------------------
        if self.use_global_gnn and self.global_encoder is not None:
            global_out = self.global_encoder(global_x, global_edge_index)
        else:
            # Ablation: skip global GNN, use projected features directly
            global_out = global_x

        # ------------------------------------------------------------------
        # Step 6. Slice batch contracts
        # → batch_out: [B, H]
        # ------------------------------------------------------------------
        batch_out = global_out[contract_ids]

        # ------------------------------------------------------------------
        # Step 7. Classify
        # → logits: [B, num_classes]
        # ------------------------------------------------------------------
        logits = self.classifier(batch_out)

        return logits

    # ----------------------------------------------------------------------
    # Utility: embedding extraction (for MC uncertainty, visualization)
    # ----------------------------------------------------------------------
    def encode(self, batch: Dict) -> torch.Tensor:
        """
        Returns pre-classifier embedding for a batch.
        Useful for MC-Dropout uncertainty and UMAP visualization.

        Returns:
            embedding: FloatTensor [B, hidden_dim // 2]
        """
        local_batch = batch["local_batch"]
        contract_ids = batch["contract_ids"]
        global_edge_index = batch["global_edge_index"]
        global_features = batch["global_features"]

        tx_emb = self.local_encoder(
            x=local_batch.x,
            edge_index=local_batch.edge_index,
            edge_attr=local_batch.edge_attr,
            batch=local_batch.batch,
        )
        contract_emb = self.pooling(tx_emb, local_batch.batch)

        global_feat_batch = global_features[contract_ids]
        fused = self.feat_proj(
            torch.cat([contract_emb, global_feat_batch], dim=-1)
        )

        global_x = self.global_feat_proj(global_features).clone()
        global_x[contract_ids] = fused

        if self.use_global_gnn and self.global_encoder is not None:
            global_out = self.global_encoder(global_x, global_edge_index)
        else:
            global_out = global_x

        batch_out = global_out[contract_ids]
        return self.classifier.encode(batch_out)

    # ----------------------------------------------------------------------
    # Model summary
    # ----------------------------------------------------------------------
    def get_model_info(self) -> Dict:
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "hidden_dim": self.hidden_dim,
            "global_feat_dim": self.global_feat_dim,
            "num_classes": self.num_classes,
            "use_global_gnn": self.use_global_gnn,
            "total_params": total_params,
            "trainable_params": trainable_params,
        }

    # ----------------------------------------------------------------------
    # Internals
    # ----------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _validate_args(
        node_dim, edge_dim, hidden_dim, global_feat_dim, num_classes, dropout
    ):
        assert node_dim > 0, f"node_dim must be > 0, got {node_dim}"
        assert edge_dim > 0, f"edge_dim must be > 0, got {edge_dim}"
        assert hidden_dim > 0, f"hidden_dim must be > 0, got {hidden_dim}"
        assert global_feat_dim >= 0, f"global_feat_dim must be >= 0, got {global_feat_dim}"
        assert num_classes >= 2, f"num_classes must be >= 2, got {num_classes}"
        assert 0.0 <= dropout < 1.0, f"dropout must be in [0, 1), got {dropout}"
