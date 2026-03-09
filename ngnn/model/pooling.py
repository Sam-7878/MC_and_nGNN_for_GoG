# model/pooling.py

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.utils import softmax


class MeanPooling(nn.Module):
    """
    Simple mean pooling over nodes in each local graph.

    Input:
        x    : FloatTensor [N, hidden_dim]
        batch: LongTensor [N]

    Output:
        pooled: FloatTensor [B, hidden_dim]
    """

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
        if batch is None:
            return x.mean(dim=0, keepdim=True)
        return global_mean_pool(x, batch)


class AttentionPooling(nn.Module):
    """
    Attention-based pooling over nodes in each local graph.

    Input:
        x    : FloatTensor [N, hidden_dim]
        batch: LongTensor [N]

    Output:
        pooled: FloatTensor [B, hidden_dim]

    Notes:
    - Computes node importance scores within each graph.
    - Uses graph-wise softmax over nodes that share the same batch id.
    """

    def __init__(
        self,
        hidden_dim: int,
        attn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        attn_hidden_dim = attn_hidden_dim or hidden_dim

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, attn_hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(attn_hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
        scores = self.attn(x)   # [N, 1]

        if batch is None:
            weights = torch.softmax(scores, dim=0)   # [N, 1]
            pooled = (x * weights).sum(dim=0, keepdim=True)
            return pooled

        weights = softmax(scores, batch)             # [N, 1], graph-wise softmax
        pooled = global_add_pool(x * weights, batch) # [B, H]
        return pooled


class ContractPooling(nn.Module):
    """
    Wrapper pooling module for local graph -> contract embedding.

    Supported modes:
        - "mean"
        - "attention"
    """

    def __init__(
        self,
        hidden_dim: int,
        mode: str = "mean",
        attn_hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.mode = mode.lower()

        if self.mode == "mean":
            self.pool = MeanPooling()
        elif self.mode == "attention":
            self.pool = AttentionPooling(
                hidden_dim=hidden_dim,
                attn_hidden_dim=attn_hidden_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(
                f"Unsupported pooling mode: {mode}. "
                f"Expected one of ['mean', 'attention']."
            )

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
        return self.pool(x, batch)
