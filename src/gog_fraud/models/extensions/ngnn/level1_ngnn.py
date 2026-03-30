# src/gog_fraud/models/extensions/ngnn/level1_ngnn.py

import torch
import torch.nn as nn

from gog_fraud.models.level1.level1_gnn import MLPHead, _cfg_get, _nested_get
from gog_fraud.models.extensions.ngnn.nested_encoder import StandardNestedEncoder
from gog_fraud.models.extensions.ngnn.readout import StandardNestedReadout

class Level1nGNN(nn.Module):
    """
    nGNN drop-in replacement for Level1GNN.
    It orchestrates extraction (if not precomputed), nested encoding, 
    nested readout, and classification.
    """
    def __init__(
        self,
        in_dim: int = 16,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.2,
        conv_type: str = "sage",
        subgraph_pooling: str = "main_root",
        nested_readout: str = "mean",
        head_hidden_dim: int = None
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout

        self.nested_encoder = StandardNestedEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            conv_type=conv_type,
            subgraph_pooling=subgraph_pooling
        )

        self.nested_readout = StandardNestedReadout(
            pooling=nested_readout
        )

        self.classifier = MLPHead(
            in_dim=hidden_dim,
            hidden_dim=head_hidden_dim or hidden_dim,
            out_dim=num_classes,
            dropout=dropout
        )

    @classmethod
    def from_config(cls, cfg: dict):
        level1_cfg = (
            _nested_get(cfg, "model", "level1")
            or _nested_get(cfg, "level1")
            or _nested_get(cfg, "model")
            or cfg
        )

        in_dim = _cfg_get(level1_cfg, "in_dim", None)
        hidden_dim = _cfg_get(level1_cfg, "hidden_dim", 128)
        num_layers = _cfg_get(level1_cfg, "num_layers", 2)
        num_classes = _cfg_get(level1_cfg, "num_classes", 2)
        dropout = _cfg_get(level1_cfg, "dropout", 0.2)
        conv_type = _cfg_get(level1_cfg, "conv_type", "sage")
        head_hidden_dim = _cfg_get(level1_cfg, "head_hidden_dim", None)

        ngnn_cfg = _cfg_get(level1_cfg, "ngnn", {})
        subgraph_pooling = _cfg_get(ngnn_cfg, "subgraph_pooling", "main_root")
        nested_readout = _cfg_get(ngnn_cfg, "nested_readout", "mean")

        return cls(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            conv_type=conv_type,
            subgraph_pooling=subgraph_pooling,
            nested_readout=nested_readout,
            head_hidden_dim=head_hidden_dim
        )

    def _resolve_batch_indices(self, data):
        """
        Calculates globally unique subgraph IDs and maps them to parent graph IDs.
        Assumes data is a merged batch of precomputed `Data` objects containing
        `subgraph_idx` (0-indexed per parent graph) and `batch` (mapping node to parent graph).
        """
        device = data.x.device
        node_to_parent = getattr(data, 'batch', None)
        
        if node_to_parent is None:
            node_to_parent = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

        node_to_local_subgraph = getattr(data, 'subgraph_idx', None)

        if node_to_local_subgraph is None:
            # Fallback: if data is a Batch of subgraphs WITHOUT parent mapping,
            # or data is just a standard graph with no nested structure.
            # In standard PyG Batch from nested list, `data.batch` is node -> subgraph 
            # and `data.ptr` exists. We handle standard batching fallback here.
            
            # Simple fallback: each node is in its own subgraph
            # We assume it hasn't been transformed
            return node_to_parent, node_to_parent, node_to_parent
            
        # Offset subgraph_ids by computing cumulative maxes
        # Find max subgraph idx per parent
        max_sub_per_parent = torch.zeros(node_to_parent.max().item() + 1, dtype=torch.long, device=device)
        # using scatter max
        max_sub_per_parent.scatter_reduce_(0, node_to_parent, node_to_local_subgraph, reduce="amax", include_self=False)
        max_sub_per_parent = max_sub_per_parent + 1 # Number of subgraphs per parent
        
        offsets = torch.cat([torch.tensor([0], device=device), torch.cumsum(max_sub_per_parent[:-1], dim=0)])
        
        global_subgraph_idx = node_to_local_subgraph + offsets[node_to_parent]
        
        # We also need a mapping from global_subgraph_idx -> parent graph
        # For each global subgraph idx, what is the parent?
        # A simple array is sufficient, since we know num_total_subgraphs 
        total_subgraphs = offsets[-1] + max_sub_per_parent[-1]
        
        # Debug
        if node_to_parent.max() > 1000 or total_subgraphs > 100000:
            print(f"[DEBUG Resolve] node_to_parent min/max: {node_to_parent.min().item()}/{node_to_parent.max().item()}")
            print(f"[DEBUG Resolve] node_to_local_subgraph max: {node_to_local_subgraph.max().item()}")
            print(f"[DEBUG Resolve] offsets max: {offsets.max().item()}")
            print(f"[DEBUG Resolve] total_subgraphs: {total_subgraphs}")

        subgraph_to_parent = torch.zeros(total_subgraphs, dtype=torch.long, device=device)
        subgraph_to_parent.scatter_(0, global_subgraph_idx, node_to_parent)
        
        return global_subgraph_idx, subgraph_to_parent, node_to_parent

    def encode_graph(self, data) -> torch.Tensor:
        global_subgraph_idx, subgraph_to_parent, node_to_parent = self._resolve_batch_indices(data)
        
        # Debug
        if node_to_parent.max() > 1000:
             print(f"[DEBUG Encode] subgraph_to_parent min/max: {subgraph_to_parent.min().item()}/{subgraph_to_parent.max().item()}")

        # Because nested_encoder currently expects data.batch to be node -> subgraph mapping:
        orig_batch = getattr(data, 'batch', None)
        data.batch = global_subgraph_idx
        
        subgraph_embs = self.nested_encoder(data)
        
        # Debug
        if node_to_parent.max() > 1000:
            print(f"[DEBUG Encode] subgraph_embs shape: {subgraph_embs.shape}")
            print(f"[DEBUG Encode] subgraph_to_parent length: {len(subgraph_to_parent)}")

        # Restore orig_batch 
        data.batch = orig_batch

        graph_embs = self.nested_readout(subgraph_embs, subgraph_to_parent)
        return graph_embs

    def embed(self, data) -> torch.Tensor:
        return self.encode_graph(data)

    def extract_graph_embedding(self, data) -> torch.Tensor:
        return self.encode_graph(data)

    def forward(self, data, return_embedding=False):
        graph_embs = self.encode_graph(data)
        logits = self.classifier(graph_embs)
        if return_embedding:
            return logits, graph_embs
        return logits

    @torch.no_grad()
    def predict_proba(self, data) -> torch.Tensor:
        logits = self.forward(data)
        if self.num_classes == 1:
            prob_pos = torch.sigmoid(logits)
            return torch.cat([1.0 - prob_pos, prob_pos], dim=-1)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict(self, data) -> torch.Tensor:
        probs = self.predict_proba(data)
        return probs.argmax(dim=-1)

    def reset_parameters(self):
        # Omitted for brevity: call reset_parameters on children
        pass
