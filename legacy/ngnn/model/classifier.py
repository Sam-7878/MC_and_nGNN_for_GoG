# model/classifier.py

from __future__ import annotations

import torch
import torch.nn as nn


class FraudClassifier(nn.Module):
    """
    Contract-level fraud classifier head for GoG + nGNN MVP.

    Input:
        x: FloatTensor [B, hidden_dim]
           ← output of GlobalEncoder sliced at [contract_ids]

    Output:
        logits: FloatTensor [B, num_classes]
                (raw logits, NOT softmax/sigmoid)

    Notes:
    - Returns raw logits → caller applies CrossEntropyLoss.
    - Two-layer MLP with BatchNorm for training stability.
    - Dropout is placed BEFORE the final projection to regularize
      during both training and MC-Dropout inference.
    - Embedding extraction via encode() method is exposed for
      downstream tasks (MC uncertainty, visualization, etc).

    Design:
        Linear(H → H//2) → BN → ReLU → Dropout → Linear(H//2 → C)
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_classes <= 1:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        mid_dim = max(hidden_dim // 2, num_classes)

        self.pre_classifier = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.fc_out = nn.Linear(mid_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns intermediate embedding before final projection.
        Useful for:
        - MC-Dropout uncertainty estimation
        - T-SNE / UMAP visualization
        - Downstream tasks

        Args:
            x: FloatTensor [B, hidden_dim]

        Returns:
            embedding: FloatTensor [B, mid_dim]
        """
        return self.pre_classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor [B, hidden_dim]

        Returns:
            logits: FloatTensor [B, num_classes]
        """
        if x is None:
            raise ValueError("FraudClassifier.forward received x=None")
        if x.dim() != 2:
            raise ValueError(
                f"FraudClassifier expects 2D input [B, H], got shape {x.shape}"
            )
        if x.size(1) != self.hidden_dim:
            raise ValueError(
                f"FraudClassifier input dim mismatch: "
                f"expected {self.hidden_dim}, got {x.size(1)}"
            )

        emb = self.pre_classifier(x)       # [B, mid_dim]
        logits = self.fc_out(emb)          # [B, num_classes]
        return logits
