# model/local_encoder.py

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import BatchNorm, GATConv


class LocalEncoder(nn.Module):
    """
    Transaction-level / local-graph encoder for GoG + nGNN MVP.

    Input:
        x         : FloatTensor [N, node_dim]
        edge_index: LongTensor [2, E]
        edge_attr : FloatTensor [E, edge_dim] or None
        batch     : LongTensor [N] (optional, not used in message passing)

    Output:
        node_embeddings: FloatTensor [N, hidden_dim]

    Notes:
    - Current raw GoG JSON has node features but no explicit edge features.
      In dataset.py we synthesize edge_attr with shape [E, 1].
    - This encoder uses GATConv with edge_attr support.
    - If a graph has no edges, it safely falls back to encoded node features only.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        use_residual: bool = True,
    ):
        super().__init__()

        if node_dim <= 0:
            raise ValueError(f"node_dim must be positive, got {node_dim}")
        if edge_dim <= 0:
            raise ValueError(f"edge_dim must be positive, got {edge_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")

        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.heads = heads
        self.use_residual = use_residual

        # Input projection
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_dim, hidden_dim)

        # Message passing blocks
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(
                GATConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=hidden_dim,
                    add_self_loops=True,
                )
            )
            self.norms.append(BatchNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x         : [N, node_dim]
            edge_index: [2, E]
            edge_attr : [E, edge_dim] or None
            batch     : [N] (unused here, kept for API consistency)

        Returns:
            node_embeddings: [N, hidden_dim]
        """
        if x is None:
            raise ValueError("LocalEncoder.forward received x=None")
        if edge_index is None:
            raise ValueError("LocalEncoder.forward received edge_index=None")

        # ------------------------------------------------------------
        # Step 1. Input projection
        # ------------------------------------------------------------
        x = self.node_encoder(x)          # [N, H]
        x = F.relu(x)
        x = self.dropout(x)

        # ------------------------------------------------------------
        # Step 2. Edge feature handling
        # ------------------------------------------------------------
        # Current MVP expects edge_attr to exist because dataset.py creates it.
        # Still, we make it robust in case edge_attr is missing.
        if edge_attr is None:
            num_edges = edge_index.size(1)
            if num_edges > 0:
                edge_attr = torch.ones(
                    (num_edges, self.edge_dim),
                    dtype=x.dtype,
                    device=x.device,
                )
            else:
                edge_attr = None

        if edge_attr is not None and edge_attr.numel() > 0:
            edge_attr = self.edge_encoder(edge_attr)   # [E, H]

        # ------------------------------------------------------------
        # Step 3. If no edges, return projected node features
        # ------------------------------------------------------------
        if edge_index.numel() == 0:
            return x

        # ------------------------------------------------------------
        # Step 4. GNN layers
        # ------------------------------------------------------------
        for conv, norm in zip(self.convs, self.norms):
            residual = x

            x = conv(x, edge_index, edge_attr=edge_attr)   # [N, H]
            x = norm(x)
            x = F.elu(x)
            x = self.dropout(x)

            if self.use_residual and x.shape == residual.shape:
                x = x + residual

        return x
