from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GINConv, global_max_pool, global_mean_pool

from gog_fraud.common.types import Level1Output


@dataclass
class Level1ModelConfig:
    in_dim: int
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    readout: str = "meanmax"   # "mean", "max", "meanmax"
    struct_dim: int = 0
    struct_hidden_dim: int = 64
    out_dim: int = 1


class MLPBlock(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StructuralEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Level1GNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.dropout = dropout
        self.layers = nn.ModuleList()

        self.layers.append(
            GINConv(
                MLPBlock(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim)
            )
        )

        for _ in range(num_layers - 1):
            self.layers.append(
                GINConv(
                    MLPBlock(in_dim=hidden_dim, hidden_dim=hidden_dim, out_dim=hidden_dim)
                )
            )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.layers:
            h = conv(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GraphReadout(nn.Module):
    def __init__(self, mode: str = "meanmax"):
        super().__init__()
        if mode not in {"mean", "max", "meanmax"}:
            raise ValueError(f"Unsupported readout mode: {mode}")
        self.mode = mode

    def forward(self, node_repr: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
        if self.mode == "mean":
            return global_mean_pool(node_repr, batch_idx)
        if self.mode == "max":
            return global_max_pool(node_repr, batch_idx)

        mean_pool = global_mean_pool(node_repr, batch_idx)
        max_pool = global_max_pool(node_repr, batch_idx)
        return torch.cat([mean_pool, max_pool], dim=-1)


class Level1FraudHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Level1Model(nn.Module):
    """
    Phase-1 Level 1 모델.
    - 그래프 내부 구조를 encode
    - graph-level embedding 생성
    - fraud score 예측
    - optional structural feature fusion 지원
    """

    def __init__(self, cfg: Level1ModelConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = Level1GNNEncoder(
            in_dim=cfg.in_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
        self.readout = GraphReadout(mode=cfg.readout)

        readout_dim = cfg.hidden_dim
        if cfg.readout == "meanmax":
            readout_dim = cfg.hidden_dim * 2

        self.struct_encoder: Optional[StructuralEncoder] = None
        struct_out_dim = 0
        if cfg.struct_dim > 0:
            self.struct_encoder = StructuralEncoder(
                in_dim=cfg.struct_dim,
                hidden_dim=cfg.struct_hidden_dim,
            )
            struct_out_dim = cfg.struct_hidden_dim

        self.output_dim = readout_dim + struct_out_dim
        self.head = Level1FraudHead(
            in_dim=self.output_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.out_dim,
        )

    def _resolve_batch_vector(self, batch) -> torch.Tensor:
        if hasattr(batch, "batch") and batch.batch is not None:
            return batch.batch
        return torch.zeros(batch.x.size(0), dtype=torch.long, device=batch.x.device)

    def _resolve_graph_id(self, batch, num_graphs: int) -> torch.Tensor:
        graph_id = getattr(batch, "graph_id", None)
        if graph_id is None:
            return torch.arange(num_graphs, device=batch.x.device)
        return graph_id.view(-1)

    def _extract_graph_struct_features(self, batch, num_graphs: int) -> Optional[torch.Tensor]:
        for attr_name in ("struct_feat", "graph_attr", "struct_x"):
            if hasattr(batch, attr_name):
                feat = getattr(batch, attr_name)
                if feat is None:
                    continue

                if feat.dim() == 1:
                    feat = feat.view(num_graphs, -1)
                elif feat.dim() == 2 and feat.size(0) != num_graphs:
                    if feat.numel() % num_graphs != 0:
                        raise ValueError(
                            f"{attr_name} cannot be reshaped to [num_graphs, -1]. "
                            f"feat.shape={tuple(feat.shape)}, num_graphs={num_graphs}"
                        )
                    feat = feat.view(num_graphs, -1)

                return feat.float()

        return None

    def forward(self, batch) -> Level1Output:
        if not hasattr(batch, "x") or not hasattr(batch, "edge_index"):
            raise ValueError("Level1Model expects batch.x and batch.edge_index")

        batch_idx = self._resolve_batch_vector(batch)
        num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 1

        node_repr = self.encoder(batch.x, batch.edge_index)
        graph_repr = self.readout(node_repr, batch_idx)

        if self.struct_encoder is not None:
            struct_feat = self._extract_graph_struct_features(batch, num_graphs)
            if struct_feat is None:
                raise ValueError(
                    "Model config expects structural features, but batch has none. "
                    "Provide one of: struct_feat, graph_attr, struct_x"
                )
            struct_repr = self.struct_encoder(struct_feat.to(graph_repr.device))
            graph_repr = torch.cat([graph_repr, struct_repr], dim=-1)

        logits = self.head(graph_repr)
        score = torch.sigmoid(logits)

        label = getattr(batch, "y", None)
        if label is not None:
            label = label.view(-1, 1).float()

        return Level1Output(
            graph_id=self._resolve_graph_id(batch, num_graphs),
            embedding=graph_repr,
            logits=logits,
            score=score,
            label=label,
            aux={
                "num_graphs": num_graphs,
                "output_dim": self.output_dim,
            },
        )

