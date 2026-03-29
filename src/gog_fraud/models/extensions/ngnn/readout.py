# src/gog_fraud/models/extensions/ngnn/readout.py

import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from src.gog_fraud.models.extensions.ngnn.interfaces import NestedReadout

class StandardNestedReadout(NestedReadout):
    """
    Pools the Subgraph-level embeddings into a Graph-level embedding.
    """
    def __init__(self, pooling: str = "mean"):
        super().__init__()
        self.pooling = pooling

    def forward(self, subgraph_embs: torch.Tensor, subgraph_to_graph_batch: torch.Tensor) -> torch.Tensor:
        """
        subgraph_embs: [total_subgraphs_in_batch, hidden_dim]
        subgraph_to_graph_batch: [total_subgraphs_in_batch] mapping to parent graph idx.
        """
        if self.pooling == "mean":
            return global_mean_pool(subgraph_embs, subgraph_to_graph_batch)
        elif self.pooling == "max":
            return global_max_pool(subgraph_embs, subgraph_to_graph_batch)
        elif self.pooling == "sum":
            return global_add_pool(subgraph_embs, subgraph_to_graph_batch)
        else:
            raise ValueError(f"Unknown readout pooling method: {self.pooling}")
