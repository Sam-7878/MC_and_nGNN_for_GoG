# model/global_encoder.py

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATConv


class GlobalEncoder(nn.Module):
    """
    Contract-level / global-graph encoder for GoG + nGNN MVP.

    Design decision: Full-graph Transductive.
        - Always receives the FULL contract graph (all N contracts).
        - Performs message passing over the entire edge_index.
        - Caller is responsible for slicing output[contract_ids]
          to get only the batch-relevant embeddings.

    Input:
        x          : FloatTensor [N, hidden_dim]     ← full contract feature matrix
        edge_index : LongTensor  [2, E]              ← full contract graph edges

    Output:
        refined_embeddings: FloatTensor [N, hidden_dim]
        → Caller slices → [B, hidden_dim]

    Notes:
    - If the global graph has no edges (empty graph during early MVP),
      the encoder safely returns the input x with residual projection.
    - Unlike LocalEncoder which processes many small tx-graphs in one batch,
      GlobalEncoder processes one large contract graph.
    - Heads are set to 1 by default at global level to avoid over-smoothing
      in the contract graph which tends to be sparse.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        heads: int = 2,
        dropout: float = 0.3,
        use_residual: bool = True,
    ):
        super().__init__()

        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.use_residual = use_residual

        # Message passing blocks
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=False,        # concat=False → output dim = hidden_dim (not heads * hidden_dim)
                    dropout=dropout,
                    edge_dim=None,       # Global encoder: no explicit edge features in contract graph
                    add_self_loops=True,
                )
            )
            self.norms.append(BatchNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Optional projection for when residual dim matches
        self.residual_proj = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x          : FloatTensor [N, hidden_dim]
            edge_index : LongTensor  [2, E]

        Returns:
            refined_embeddings: FloatTensor [N, hidden_dim]
        """
        if x is None:
            raise ValueError("GlobalEncoder.forward received x=None")
        if edge_index is None:
            raise ValueError("GlobalEncoder.forward received edge_index=None")

        # ------------------------------------------------------------
        # Safety: if no edges exist (empty contract graph fallback)
        # ------------------------------------------------------------
        if edge_index.numel() == 0:
            # No inter-contract edges → return input as-is
            # This is valid during early MVP where global graph is not yet built
            return x

        # ------------------------------------------------------------
        # GNN layers over full contract graph
        # ------------------------------------------------------------
        for conv, norm in zip(self.convs, self.norms):
            residual = x

            x = conv(x, edge_index)   # [N, hidden_dim]
            x = norm(x)
            x = F.elu(x)
            x = self.dropout(x)

            if self.use_residual:
                x = x + self.residual_proj(residual)

        return x
