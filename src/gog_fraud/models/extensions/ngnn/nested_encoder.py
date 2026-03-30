# src/gog_fraud/models/extensions/ngnn/nested_encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import SAGEConv, GCNConv, GATv2Conv, global_mean_pool, global_max_pool, global_add_pool

from gog_fraud.models.extensions.ngnn.interfaces import NestedEncoder

class StandardNestedEncoder(NestedEncoder):
    """
    Standard GNN encoder run over the extracted nested subgraphs.
    This module expects a PyG Batch consisting of multiple disconnected
    rooted subgraphs. 
    It pools node embeddings WITHIN each subgraph to form subgraph embeddings.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        conv_type: str = "sage",
        subgraph_pooling: str = "main_root"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.subgraph_pooling = subgraph_pooling
        
        if in_dim is None:
            self.input_proj = nn.LazyLinear(hidden_dim)
        else:
            self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            if conv_type == "sage":
                conv = SAGEConv(hidden_dim, hidden_dim)
            elif conv_type == "gcn":
                conv = GCNConv(hidden_dim, hidden_dim)
            elif conv_type == "gat":
                conv = GATv2Conv(hidden_dim, hidden_dim, heads=1, concat=False)
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")
                
            self.convs.append(conv)
            self.norms.append(nn.BatchNorm1d(hidden_dim))

    def forward(self, nested_batch: Batch) -> torch.Tensor:
        """
        nested_batch is a PyG Batch containing many subgraphs.
        nested_batch.batch maps node -> subgraph index.
        """
        x = nested_batch.x
        edge_index = nested_batch.edge_index
        batch = nested_batch.batch

        # 1. Base Projection
        if x.dim() == 1 or x.size(1) == 0:
            x = torch.ones((x.size(0), 1), device=x.device, dtype=torch.float)
            
        x = self.input_proj(x)

        # 2. Message Passing
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 3. Subgraph Pooling
        if self.subgraph_pooling == "main_root":
            # Just take the representation of the root node of each subgraph
            # Assumes root_indicator is set by the extractor
            if hasattr(nested_batch, 'root_indicator'):
                root_mask = nested_batch.root_indicator.bool()
                subgraph_embs = x[root_mask]
            else:
                # Fallback to mean pool
                subgraph_embs = global_mean_pool(x, batch)
        elif self.subgraph_pooling == "mean":
            subgraph_embs = global_mean_pool(x, batch)
        elif self.subgraph_pooling == "max":
            subgraph_embs = global_max_pool(x, batch)
        elif self.subgraph_pooling == "sum":
            subgraph_embs = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.subgraph_pooling}")
            
        return subgraph_embs
