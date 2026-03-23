# src/gog_fraud/adapters/legacy_adapter.py

from __future__ import annotations

import importlib
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

log = logging.getLogger(__name__)


# ---------------------------------------------------------
# Public config expected by run_fraud_benchmark.py
# ---------------------------------------------------------

@dataclass
class LegacyAdapterConfig:
    agg_method: str = "max"          # "max" | "mean" | "topk"
    topk: int = 3
    normalize_score: bool = True

    gpu: int = -1
    hid_dim: int = 64
    num_layers: int = 2
    epoch: int = 100
    lr: float = 0.003

    weight_decay: float = 0.0
    dropout: float = 0.0
    batch_size: int = 0

    # optional extension
    extra_kwargs: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------
# Supported detector registry
# ---------------------------------------------------------

_SUPPORTED_MODELS: Dict[str, str] = {
    "DOMINANT": "pygod.detector.DOMINANT",
    "DONE": "pygod.detector.DONE",
    "GAE": "pygod.detector.GAE",
    "AnomalyDAE": "pygod.detector.AnomalyDAE",
    "CoLA": "pygod.detector.CoLA",
    # Optional extras
    "CONAD": "pygod.detector.CONAD",
    "GUIDE": "pygod.detector.GUIDE",
    "VGAE": "pygod.detector.VGAE",
    "GAAN": "pygod.detector.GAAN",
    "OCGNN": "pygod.detector.OCGNN",
}


# ---------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------

def _import_detector_class(model_name: str):
    if model_name not in _SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported legacy model: {model_name}. "
            f"Supported={list(_SUPPORTED_MODELS.keys())}"
        )

    path = _SUPPORTED_MODELS[model_name]
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _filter_kwargs_for_callable(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pass only kwargs accepted by the detector's __init__.
    This avoids version-specific constructor mismatch.
    """
    try:
        sig = inspect.signature(cls.__init__)
        params = sig.parameters
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

        if accepts_var_kw:
            return dict(kwargs)

        allowed = set(params.keys()) - {"self"}
        return {k: v for k, v in kwargs.items() if k in allowed}
    except Exception:
        return dict(kwargs)


def _sanitize_tensor(x: torch.Tensor) -> torch.Tensor:
    x = x.detach()
    if not torch.is_floating_point(x):
        return x
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _prepare_graph(graph: Data) -> Data:
    """
    Build a safe PyG Data object for legacy detectors.
    """
    if getattr(graph, "x", None) is None:
        raise ValueError("Graph has no node feature 'x'")
    if getattr(graph, "edge_index", None) is None:
        raise ValueError("Graph has no 'edge_index'")

    x = _sanitize_tensor(graph.x).float()
    edge_index = graph.edge_index.detach().long()

    g = Data(x=x, edge_index=edge_index)
    g.num_nodes = int(getattr(graph, "num_nodes", x.size(0)))

    if getattr(graph, "edge_attr", None) is not None:
        g.edge_attr = _sanitize_tensor(graph.edge_attr).float()

    # Some detectors internally expect y to exist.
    y = getattr(graph, "y", None)
    if y is None:
        g.y = torch.zeros(g.num_nodes, dtype=torch.long)
    else:
        g.y = y.detach().long()

    return g


def _extract_contract_and_graph(item: Any, idx: int) -> Tuple[str, Data]:
    """
    Supports:
      - object with .contract_id and .graph
      - tuple(contract_id, graph)
      - raw Data object
    """
    if hasattr(item, "contract_id") and hasattr(item, "graph"):
        return str(item.contract_id), item.graph

    if isinstance(item, tuple) and len(item) == 2:
        cid, graph = item
        return str(cid), graph

    if isinstance(item, Data):
        return f"graph_{idx}", item

    raise TypeError(
        f"Unsupported graph item type at index={idx}: {type(item)}"
    )


def _to_numpy_1d(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def _aggregate_node_scores(
    raw_scores: Any,
    agg_method: str = "max",
    topk: int = 3,
) -> float:
    """
    Aggregate node anomaly scores into a single contract score.
    Non-finite values are dropped before aggregation.
    """
    arr = _to_numpy_1d(raw_scores)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        return float("nan")

    if agg_method == "max":
        return float(arr.max())

    if agg_method == "mean":
        return float(arr.mean())

    if agg_method == "topk":
        k = max(1, min(int(topk), int(arr.size)))
        top = np.partition(arr, -k)[-k:]
        return float(top.mean())

    raise ValueError(
        f"Unknown agg_method={agg_method}. "
        f"Use one of ['max', 'mean', 'topk']."
    )


def _safe_minmax_normalize(score_dict: Dict[str, float]) -> Dict[str, float]:
    """
    NaN-safe min-max normalization.

    Rules:
      - drop non-finite scores first
      - if all values are constant, return 0.0 for all
      - else normal min-max scaling
    """
    if not score_dict:
        return {}

    clean_items: List[Tuple[str, float]] = []
    dropped: List[str] = []

    for cid, value in score_dict.items():
        try:
            fv = float(value)
        except Exception:
            fv = float("nan")

        if np.isfinite(fv):
            clean_items.append((cid, fv))
        else:
            dropped.append(cid)

    if dropped:
        log.warning(
            f"[LegacyAdapter] Dropping {len(dropped)} non-finite scores before "
            f"normalization: {dropped[:10]}"
            + ("..." if len(dropped) > 10 else "")
        )

    if not clean_items:
        return {}

    values = np.asarray([v for _, v in clean_items], dtype=np.float64)
    vmin = float(values.min())
    vmax = float(values.max())
    vrange = vmax - vmin

    # Constant-score protection
    if (not np.isfinite(vrange)) or vrange < 1e-12:
        log.warning(
            "[LegacyAdapter] All scores are constant during normalization. "
            "Returning 0.0 for all contracts."
        )
        return {cid: 0.0 for cid, _ in clean_items}

    return {cid: float((v - vmin) / vrange) for cid, v in clean_items}


def _sanitize_score_dict(score_dict: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    dropped = 0

    for cid, value in score_dict.items():
        try:
            fv = float(value)
        except Exception:
            fv = float("nan")

        if np.isfinite(fv):
            out[cid] = fv
        else:
            dropped += 1

    if dropped > 0:
        log.warning(
            f"[LegacyAdapter] Dropped {dropped} non-finite final scores."
        )
    return out


# ---------------------------------------------------------
# Internal single-model runner
# ---------------------------------------------------------

class _LegacySingleRunner:
    def __init__(self, model_name: str, cfg: LegacyAdapterConfig):
        self.model_name = str(model_name)
        self.cfg = cfg
        self.detector_cls = _import_detector_class(self.model_name)

    def _build_detector(self):
        raw_kwargs = {
            "hid_dim": self.cfg.hid_dim,
            "num_layers": self.cfg.num_layers,
            "epoch": self.cfg.epoch,
            "lr": self.cfg.lr,
            "gpu": self.cfg.gpu,
            "weight_decay": self.cfg.weight_decay,
            "dropout": self.cfg.dropout,
            "batch_size": self.cfg.batch_size,
        }
        raw_kwargs.update(self.cfg.extra_kwargs)

        kwargs = _filter_kwargs_for_callable(self.detector_cls, raw_kwargs)

        try:
            return self.detector_cls(**kwargs)
        except TypeError:
            # Last-resort fallback for older pygod versions
            fallback = {
                k: v for k, v in kwargs.items()
                if k in {"hid_dim", "num_layers", "epoch", "lr", "gpu"}
            }
            return self.detector_cls(**fallback)

    def _score_one(self, contract_id: str, graph: Data) -> Optional[float]:
        try:
            g = _prepare_graph(graph)

            if g.num_nodes < 2:
                log.warning(
                    f"[LegacyRunner:{self.model_name}] Skip {contract_id}: "
                    f"num_nodes={g.num_nodes}"
                )
                return None

            detector = self._build_detector()
            detector.fit(g)

            raw_scores = getattr(detector, "decision_scores_", None)
            if raw_scores is None:
                raise RuntimeError("detector.decision_scores_ is missing after fit()")

            score = _aggregate_node_scores(
                raw_scores,
                agg_method=self.cfg.agg_method,
                topk=self.cfg.topk,
            )

            if not np.isfinite(score):
                log.warning(
                    f"[LegacyRunner:{self.model_name}] Non-finite aggregated score "
                    f"for {contract_id}; skipping."
                )
                return None

            return float(score)

        except Exception as exc:
            log.warning(
                f"[LegacyRunner:{self.model_name}] Skip {contract_id}: {exc}"
            )
            return None

    def run(self, graph_items: Sequence[Any]) -> Dict[str, float]:
        total = len(graph_items)
        scores: Dict[str, float] = {}

        if total == 0:
            log.warning(f"[LegacyRunner:{self.model_name}] No graphs to score.")
            return scores

        log.info(f"[LegacyRunner:{self.model_name}] 0/{total} (0%)")

        log_every = max(1, total // 10)
        skipped = 0

        for i, item in enumerate(graph_items):
            cid, graph = _extract_contract_and_graph(item, i)
            score = self._score_one(cid, graph)

            if score is None:
                skipped += 1
            else:
                scores[cid] = score

            if ((i + 1) % log_every == 0) or ((i + 1) == total):
                pct = int(((i + 1) / total) * 100)
                log.info(
                    f"[LegacyRunner:{self.model_name}] {i + 1}/{total} ({pct}%)"
                )

        log.info(
            f"[LegacyRunner:{self.model_name}] Done. "
            f"Scored {len(scores)} contracts."
            + (f" (skipped={skipped})" if skipped > 0 else "")
        )
        return scores


# ---------------------------------------------------------
# Public batch runner expected by run_fraud_benchmark.py
# ---------------------------------------------------------

class LegacyBatchRunner:
    """
    Interface expected by run_fraud_benchmark.py:

        batch = LegacyBatchRunner(model_names=[...], base_cfg=LegacyAdapterConfig(...))
        all_scores = batch.run_all(test_graphs)

    Returns:
        {
            "DOMINANT": {contract_id: score, ...},
            "DONE": {contract_id: score, ...},
            ...
        }
    """

    def __init__(
        self,
        model_names: Sequence[str],
        base_cfg: LegacyAdapterConfig,
    ) -> None:
        self.model_names = [str(x) for x in model_names]
        self.base_cfg = base_cfg

    def run_all(self, graph_items: Sequence[Any]) -> Dict[str, Dict[str, float]]:
        all_scores: Dict[str, Dict[str, float]] = {}

        for model_name in self.model_names:
            log.info(f"\n[LegacyBatchRunner] === Running {model_name} ===")

            runner = _LegacySingleRunner(model_name=model_name, cfg=self.base_cfg)
            score_dict = runner.run(graph_items)

            # Final sanitize
            score_dict = _sanitize_score_dict(score_dict)

            # Safe normalization
            if self.base_cfg.normalize_score:
                score_dict = _safe_minmax_normalize(score_dict)

            all_scores[model_name] = score_dict

        return all_scores


__all__ = [
    "LegacyAdapterConfig",
    "LegacyBatchRunner",
]
