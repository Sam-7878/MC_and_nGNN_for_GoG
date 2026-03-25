# src/gog_fraud/models/level1/level1_gnn.py

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv,
    SAGEConv,
    GATv2Conv,
    global_mean_pool,
    global_add_pool,
    global_max_pool,
)


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _nested_get(cfg: Any, *keys, default=None):
    cur = cfg
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Level1GNN(nn.Module):
    """
    Graph-level classifier for Level1 fraud detection.

    Expected input:
      - PyG Data or Batch
      - fields:
          x: [num_nodes, num_node_features] (optional)
          edge_index: [2, num_edges] (required)
          batch: [num_nodes] (optional; auto-created if absent)

    Output:
      - logits: [num_graphs, num_classes]

    Useful methods:
      - forward(data): logits
      - encode_graph(data): graph embedding
      - predict_proba(data): probabilities
      - from_config(cfg): build from config
    """

    def __init__(
        self,
        in_dim: int = 32,
        out_dim: int = 32,
        input_dim: Optional[int] = None,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_classes: int = 2,
        dropout: float = 0.2,
        conv_type: str = "sage",
        pooling: str = "mean",
        use_batchnorm: bool = True,
        head_hidden_dim: Optional[int] = None,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.in_dim = in_dim
        self.out_dim = out_dim
        # self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.conv_type = conv_type.lower()
        self.pooling = pooling.lower()
        self.use_batchnorm = use_batchnorm
        self.head_hidden_dim = head_hidden_dim or hidden_dim

        # x가 없거나 in_dim을 모를 수 있으므로 LazyLinear 사용
        if in_dim is None:
            self.input_proj = nn.LazyLinear(hidden_dim)
        else:
            self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(self._build_conv(hidden_dim, hidden_dim, self.conv_type))
            if self.use_batchnorm:
                self.norms.append(nn.BatchNorm1d(hidden_dim))

        self.classifier = MLPHead(
            in_dim=hidden_dim,
            hidden_dim=self.head_hidden_dim,
            out_dim=num_classes,
            dropout=dropout,
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "Level1GNN":
        """
        가능한 한 다양한 config 구조를 흡수하도록 작성.
        우선순위:
          1) cfg.model.level1
          2) cfg.level1
          3) cfg.model
          4) cfg 자체
        """
        level1_cfg = (
            _nested_get(cfg, "model", "level1")
            or _nested_get(cfg, "level1")
            or _nested_get(cfg, "model")
            or cfg
        )

        input_dim = (
            _cfg_get(level1_cfg, "input_dim", None)
            or _cfg_get(level1_cfg, "in_dim", None)
            or _cfg_get(level1_cfg, "num_node_features", None)
            or _cfg_get(level1_cfg, "node_feat_dim", None)
        )

        hidden_dim = (
            _cfg_get(level1_cfg, "hidden_dim", None)
            or _cfg_get(level1_cfg, "hidden_channels", None)
            or 128
        )

        num_layers = (
            _cfg_get(level1_cfg, "num_layers", None)
            or _cfg_get(level1_cfg, "gnn_layers", None)
            or 3
        )

        num_classes = (
            _cfg_get(level1_cfg, "num_classes", None)
            or _cfg_get(level1_cfg, "out_dim", None)
            or _cfg_get(level1_cfg, "output_dim", None)
            or 2
        )

        dropout = _cfg_get(level1_cfg, "dropout", 0.2)
        conv_type = _cfg_get(level1_cfg, "conv_type", "sage")
        pooling = (
            _cfg_get(level1_cfg, "pooling", None)
            or _cfg_get(level1_cfg, "readout", None)
            or "mean"
        )
        use_batchnorm = _cfg_get(level1_cfg, "use_batchnorm", True)
        head_hidden_dim = _cfg_get(level1_cfg, "head_hidden_dim", None)

        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
            conv_type=conv_type,
            pooling=pooling,
            use_batchnorm=use_batchnorm,
            head_hidden_dim=head_hidden_dim,
        )

    def _build_conv(self, in_dim: int, out_dim: int, conv_type: str) -> nn.Module:
        if conv_type == "sage":
            return SAGEConv(in_dim, out_dim)
        if conv_type == "gcn":
            return GCNConv(in_dim, out_dim)
        if conv_type == "gat":
            return GATv2Conv(in_dim, out_dim, heads=1, concat=False)

        raise ValueError(
            f"Unsupported conv_type='{conv_type}'. "
            f"Choose from ['sage', 'gcn', 'gat']."
        )

    def _get_x(self, data) -> torch.Tensor:
        x = getattr(data, "x", None)
        if x is None:
            num_nodes = self._num_nodes(data)
            device = self._infer_device(data)
            x = torch.ones((num_nodes, 1), dtype=torch.float, device=device)
        return x.float()

    def _get_edge_index(self, data) -> torch.Tensor:
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            raise ValueError("Input data must have 'edge_index'.")
        return edge_index

    def _get_batch(self, data, num_nodes: int) -> torch.Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            device = self._infer_device(data)
            batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        return batch

    def _num_nodes(self, data) -> int:
        if getattr(data, "x", None) is not None:
            return data.x.size(0)
        if hasattr(data, "num_nodes") and data.num_nodes is not None:
            return int(data.num_nodes)
        if getattr(data, "edge_index", None) is not None and data.edge_index.numel() > 0:
            return int(data.edge_index.max().item()) + 1
        raise ValueError("Cannot infer number of nodes from data.")

    def _infer_device(self, data) -> torch.device:
        if getattr(data, "x", None) is not None:
            return data.x.device
        if getattr(data, "edge_index", None) is not None:
            return data.edge_index.device
        return torch.device("cpu")

    def _pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            return global_mean_pool(x, batch)
        if self.pooling == "sum":
            return global_add_pool(x, batch)
        if self.pooling == "max":
            return global_max_pool(x, batch)
        raise ValueError(
            f"Unsupported pooling='{self.pooling}'. Choose from ['mean', 'sum', 'max']."
        )

    def encode_nodes(self, data) -> torch.Tensor:
        x = self._get_x(data)
        edge_index = self._get_edge_index(data)

        x = self.input_proj(x)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_batchnorm:
                x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def encode_graph(self, data) -> torch.Tensor:
        node_emb = self.encode_nodes(data)
        batch = self._get_batch(data, num_nodes=node_emb.size(0))
        graph_emb = self._pool(node_emb, batch)
        return graph_emb

    # alias
    def embed(self, data) -> torch.Tensor:
        return self.encode_graph(data)

    def extract_graph_embedding(self, data) -> torch.Tensor:
        return self.encode_graph(data)

    def forward(self, data, return_embedding: bool = False):
        graph_emb = self.encode_graph(data)
        logits = self.classifier(graph_emb)
        if return_embedding:
            return logits, graph_emb
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
        if hasattr(self.input_proj, "reset_parameters"):
            self.input_proj.reset_parameters()

        for conv in self.convs:
            if hasattr(conv, "reset_parameters"):
                conv.reset_parameters()

        for norm in self.norms:
            if hasattr(norm, "reset_parameters"):
                norm.reset_parameters()

        for module in self.classifier.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
