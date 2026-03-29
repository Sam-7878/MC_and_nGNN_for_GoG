from typing import Optional
from dataclasses import asdict, dataclass, fields, is_dataclass
from dataclasses import dataclass as _dc, field as _field
from types import SimpleNamespace
from typing import Any, Mapping, Dict
import inspect

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool, global_mean_pool

from gog_fraud.common.types import Level1Output


# ──────────────────────────────────────────────
# Output type for Level 2
# ──────────────────────────────────────────────



def _cfg_to_plain_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}

    if isinstance(cfg, dict):
        return dict(cfg)

    if is_dataclass(cfg):
        return asdict(cfg)

    if hasattr(cfg, "items"):
        try:
            return dict(cfg.items())
        except Exception:
            pass

    if hasattr(cfg, "__dict__"):
        return {
            k: v
            for k, v in vars(cfg).items()
            if not k.startswith("_")
        }

    raise TypeError(f"Unsupported config type: {type(cfg)}")


def _filter_kwargs_for_cls_init(cls, data: dict) -> dict:
    try:
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in params.values()
        )
        if accepts_var_kw:
            return dict(data)

        allowed = set(params.keys()) - {"self"}
        return {k: v for k, v in data.items() if k in allowed}
    except Exception:
        return dict(data)


@_dc
class Level2Output:
    graph_id:  torch.Tensor
    embedding: torch.Tensor
    logits:    torch.Tensor
    score:     torch.Tensor
    label:     Optional[torch.Tensor] = None
    aux:       Dict[str, Any] = _field(default_factory=dict)


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────

@dataclass
class Level2ModelConfig:
    in_dim:          int  = 16     # Level 1 emb dim + 1 (score)
    hidden_dim:      int  = 128
    num_layers:      int  = 2
    num_heads:       int  = 4
    dropout:         float = 0.2
    edge_dim:        int  = 0      # 0 = no edge feature
    readout:         str  = "meanmax"
    out_dim:         int  = 1

    @classmethod
    def from_config(cls, cfg: Any) -> "Level2ModelConfig":
        if isinstance(cfg, cls):
            return cfg

        data = _cfg_to_plain_dict(cfg)

        # 흔한 alias 보정
        if "hidden_dim" in data and "hid_dim" not in data:
            data["hid_dim"] = data["hidden_dim"]
        if "num_layer" in data and "num_layers" not in data:
            data["num_layers"] = data["num_layer"]
        if "lr" in data and "learning_rate" not in data:
            data["learning_rate"] = data["lr"]

        valid = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# ──────────────────────────────────────────────
# Sub-modules
# ──────────────────────────────────────────────

class Level2GATEncoder(nn.Module):
    """
    Multi-layer GATv2 encoder for the Level 2 relation graph.
    GATv2 is preferred over GAT for dynamic attention strength.
    """

    def __init__(
        self,
        in_dim:     int,
        hidden_dim: int,
        num_layers: int,
        num_heads:  int,
        dropout:    float,
        edge_dim:   Optional[int] = None,
    ):
        super().__init__()
        self.dropout = dropout
        self.layers  = nn.ModuleList()
        self.norms   = nn.ModuleList()

        _edge_dim = edge_dim if (edge_dim is not None and edge_dim > 0) else None

        # layer 0: in_dim → hidden_dim
        self.layers.append(
            GATv2Conv(
                in_channels=in_dim,
                out_channels=hidden_dim // num_heads,
                heads=num_heads,
                dropout=dropout,
                edge_dim=_edge_dim,
                concat=True,
            )
        )
        self.norms.append(nn.LayerNorm(hidden_dim))

        # layers 1 … (num_layers-1): hidden_dim → hidden_dim
        for _ in range(num_layers - 1):
            self.layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    edge_dim=_edge_dim,
                    concat=True,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

    def forward(
        self,
        x:          torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x
        for conv, norm in zip(self.layers, self.norms):
            kwargs = {}
            if edge_attr is not None:
                kwargs["edge_attr"] = edge_attr.float()
            h = conv(h, edge_index, **kwargs)
            h = norm(h)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class Level2GraphReadout(nn.Module):
    def __init__(self, mode: str = "meanmax"):
        super().__init__()
        if mode not in {"mean", "max", "add", "meanmax"}:
            raise ValueError(f"Unsupported readout mode: {mode}")
        self.mode = mode

    def forward(
        self,
        node_repr: torch.Tensor,
        batch_idx: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "mean":
            return global_mean_pool(node_repr, batch_idx)
        if self.mode == "max":
            return global_max_pool(node_repr, batch_idx)
        if self.mode == "add":
            return global_add_pool(node_repr, batch_idx)
        # meanmax
        mean_p = global_mean_pool(node_repr, batch_idx)
        max_p  = global_max_pool(node_repr, batch_idx)
        return torch.cat([mean_p, max_p], dim=-1)


class Level2FraudHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────
# Level 2 Model
# ──────────────────────────────────────────────

class Level2Model(nn.Module):
    """
    Phase-3 Level 2 모델.
    - Level 1 embeddings로 구성된 relation graph를 입력으로 받음
    - GATv2 기반 relation 모델링
    - graph-level embedding 및 fraud score 생성
    - 결과를 Level2Output으로 반환
    """

    def __init__(self, cfg: Level2ModelConfig):
        super().__init__()
        self.cfg = cfg

        # Added: Input normalization to stabilize Level 1 features especially on large Ethereum clusters
        self.input_norm = nn.LayerNorm(cfg.in_dim)

        _edge_dim = cfg.edge_dim if cfg.edge_dim > 0 else None

        self.encoder = Level2GATEncoder(
            in_dim=cfg.in_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            edge_dim=_edge_dim,
        )
        self.readout = Level2GraphReadout(mode=cfg.readout)

        readout_out_dim = cfg.hidden_dim
        if cfg.readout == "meanmax":
            readout_out_dim = cfg.hidden_dim * 2

        self.out_dim = readout_out_dim
        self.head = Level2FraudHead(
            in_dim=readout_out_dim,
            hidden_dim=cfg.hidden_dim,
            out_dim=cfg.out_dim,
        )

    def _resolve_batch_vector(self, batch) -> torch.Tensor:
        if hasattr(batch, "batch") and batch.batch is not None:
            return batch.batch
        return torch.zeros(batch.x.size(0), dtype=torch.long, device=batch.x.device)

    def _resolve_graph_id(self, batch, num_graphs: int) -> torch.Tensor:
        """
        Level 2에서 graph_id는 graph-level 식별자.
        batch.graph_id는 node-level (Level 1 graph_id의 집합)이므로
        graph-level index인 arange를 반환한다.
        """
        return torch.arange(num_graphs, device=batch.x.device)


    def forward(self, batch) -> Level2Output:
        if not hasattr(batch, "x") or not hasattr(batch, "edge_index"):
            raise ValueError("Level2Model expects batch.x and batch.edge_index")

        batch_idx = self._resolve_batch_vector(batch)
        try:
            num_graphs = i.item() if (i := batch_idx.max()).numel() > 0 else 0
            num_graphs = num_graphs + 1 if batch_idx.numel() > 0 else 1
        except Exception:
            num_graphs = 1

        edge_attr = getattr(batch, "edge_attr", None)
        if edge_attr is not None and self.cfg.edge_dim == 0:
            edge_attr = None

        # Stability: Normalize input features and clamp
        x = batch.x.float()
        x = self.input_norm(x)
        x = torch.clamp(x, min=-10.0, max=10.0)

        node_repr = self.encoder(
            x=x,
            edge_index=batch.edge_index,
            edge_attr=edge_attr,
        )
        graph_repr = self.readout(node_repr, batch_idx)
        logits = self.head(graph_repr)
        
        # Stability: Clamp logits before sigmoid and handle NaNs
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        score  = torch.sigmoid(logits)
        
        # Last resort: convert any residual NaNs to small finite values
        score = torch.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)

        label = getattr(batch, "y", None)
        if label is not None:
            label = label.view(-1, 1).float()

        return Level2Output(
            graph_id=self._resolve_graph_id(batch, num_graphs),
            embedding=graph_repr,
            logits=logits,
            score=score,
            label=label,
            aux={
                "num_graphs": num_graphs,
                "out_dim": self.out_dim,
            },
        )

    @classmethod
    def from_config(cls, cfg: Any) -> "Level2Model":
        cfg_obj = Level2ModelConfig.from_config(cfg)

        # 1순위: 생성자가 config 객체를 직접 받는 경우
        try:
            return cls(cfg_obj)
        except TypeError:
            pass

        # 2순위: 생성자가 kwargs를 받는 경우
        data = asdict(cfg_obj)
        kwargs = _filter_kwargs_for_cls_init(cls, data)

        try:
            return cls(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"{cls.__name__}.from_config() failed. "
                f"Config keys={list(data.keys())}, filtered={list(kwargs.keys())}, error={e}"
            ) from e

    @torch.no_grad()
    def predict(self, *args, **kwargs):
        self.eval()
        out = self.forward(*args, **kwargs)

        if hasattr(out, "score"):
            score = out.score
        elif isinstance(out, dict):
            score = (
                out.get("score", None)
                or out.get("anomaly_score", None)
                or out.get("logit", None)
                or out.get("logits", None)
            )
            if score is None:
                raise KeyError("predict/forward output has no score-like key")
        else:
            score = out

        if not torch.is_tensor(score):
            score = torch.tensor(score, dtype=torch.float32)

        score = score.reshape(-1)
        return SimpleNamespace(score=score)
