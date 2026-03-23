# legacy_adapter.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Data

logger = logging.getLogger(__name__)


@dataclass
class LegacyAdapterConfig:
    detector_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    score_reduce: str = "mean"
    progress_every: int = 6
    agg_method: Optional[str] = None
    
    # 아래 두 줄을 추가하여 yaml 설정값(kwargs)을 받을 수 있게 합니다.
    topk: int = 3
    normalize_score: bool = True

    # 아래의 하이퍼파라미터 필드들을 추가합니다.
    gpu: int = 0
    hid_dim: int = 16
    num_layers: int = 2
    epoch: int = 20
    lr: float = 0.003
    weight_decay: float = 0.0
    dropout: float = 0.0

    def __post_init__(self):
        if self.agg_method and not self.score_reduce:
            self.score_reduce = self.agg_method
        elif self.agg_method:
            self.score_reduce = self.agg_method

# -----------------------------------------------------------------------------
# Detector defaults
# -----------------------------------------------------------------------------
_DEFAULT_DETECTOR_KWARGS: Dict[str, Dict[str, Any]] = {
    "DOMINANT": {"epoch": 5, "verbose": 0},
    "CONAD": {"epoch": 5, "verbose": 0},
    "DONE": {"epoch": 5, "verbose": 0},
    "ANOMALYDAE": {"epoch": 5, "verbose": 0},
    "COLA": {"epoch": 5, "verbose": 0},
    "GAAN": {"epoch": 5, "verbose": 0},
    "GUIDE": {"epoch": 5, "verbose": 0},
}


# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------
def _safe_list(items: Iterable[Any]) -> List[Any]:
    if isinstance(items, list):
        return items
    return list(items)


def _unwrap_data(item: Any) -> Optional[Data]:
    """
    Accept:
      - torch_geometric.data.Data
      - wrapper objects with `.graph` field
    """
    if isinstance(item, Data):
        return item
    if hasattr(item, "graph") and isinstance(item.graph, Data):
        return item.graph
    return None


def _extract_contract_id(item: Any, data: Optional[Data], idx: int) -> str:
    for obj in (item, data):
        if obj is None:
            continue
        if hasattr(obj, "contract_id") and getattr(obj, "contract_id") is not None:
            return str(getattr(obj, "contract_id"))
        if hasattr(obj, "address") and getattr(obj, "address") is not None:
            return str(getattr(obj, "address"))
        if hasattr(obj, "id") and getattr(obj, "id") is not None:
            return str(getattr(obj, "id"))
    return f"graph_{idx:06d}"


def _extract_label(item: Any, data: Optional[Data]) -> Optional[float]:
    for obj in (item, data):
        if obj is None:
            continue
        if hasattr(obj, "label"):
            val = getattr(obj, "label")
            try:
                return float(val)
            except Exception:
                pass

    if data is not None and hasattr(data, "y") and getattr(data, "y") is not None:
        y = getattr(data, "y")
        try:
            yt = torch.as_tensor(y).view(-1)
            if yt.numel() == 1:
                return float(yt.item())
        except Exception:
            pass

    return None


def _prepare_graph_for_detector(data: Data) -> Data:
    """
    Minimal normalization for PyG / PyGOD detectors.
    """
    if getattr(data, "x", None) is None:
        raise ValueError("graph has no `x`")
    if getattr(data, "edge_index", None) is None:
        raise ValueError("graph has no `edge_index`")

    data.x = data.x.float()
    data.edge_index = data.edge_index.long()
    return data


# -----------------------------------------------------------------------------
# Score helpers
# -----------------------------------------------------------------------------
def _to_1d_float_tensor(x: Any) -> Optional[torch.Tensor]:
    if x is None:
        return None
    try:
        t = torch.as_tensor(x).detach().view(-1).float()
    except Exception:
        return None
    if t.numel() == 0:
        return None
    return t


def _sanitize_scores(scores: Any) -> Optional[torch.Tensor]:
    t = _to_1d_float_tensor(scores)
    if t is None:
        return None

    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)

    if t.numel() == 0:
        return None
    return t


def _extract_detector_scores(
    detector: Any,
    data: Data,
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """
    Try multiple detector score APIs in order.

    Returns:
        (scores, source_name)
    """
    # 1) plural attribute
    if hasattr(detector, "decision_scores_"):
        scores = _sanitize_scores(getattr(detector, "decision_scores_"))
        if scores is not None:
            return scores, "decision_scores_"

    # 2) singular attribute
    if hasattr(detector, "decision_score_"):
        scores = _sanitize_scores(getattr(detector, "decision_score_"))
        if scores is not None:
            return scores, "decision_score_"

    # 3) decision_function(data) or decision_function()
    if hasattr(detector, "decision_function"):
        fn = getattr(detector, "decision_function")

        try:
            scores = _sanitize_scores(fn(data))
            if scores is not None:
                return scores, "decision_function(data)"
        except TypeError:
            pass
        except Exception as exc:
            logger.debug(
                "[legacy_adapter] decision_function(data) failed: %r",
                exc,
            )

        try:
            scores = _sanitize_scores(fn())
            if scores is not None:
                return scores, "decision_function()"
        except Exception as exc:
            logger.debug(
                "[legacy_adapter] decision_function() failed: %r",
                exc,
            )

    # 4) predict(..., return_score=True)
    if hasattr(detector, "predict"):
        pred_fn = getattr(detector, "predict")

        try:
            out = pred_fn(data, return_score=True)
            if isinstance(out, tuple) and len(out) >= 2:
                scores = _sanitize_scores(out[1])
                if scores is not None:
                    return scores, "predict(data, return_score=True)"
        except TypeError:
            pass
        except Exception as exc:
            logger.debug(
                "[legacy_adapter] predict(data, return_score=True) failed: %r",
                exc,
            )

        try:
            out = pred_fn(return_score=True)
            if isinstance(out, tuple) and len(out) >= 2:
                scores = _sanitize_scores(out[1])
                if scores is not None:
                    return scores, "predict(return_score=True)"
        except Exception as exc:
            logger.debug(
                "[legacy_adapter] predict(return_score=True) failed: %r",
                exc,
            )

    return None, None


def _reduce_node_scores_to_graph_score(
    scores: torch.Tensor,
    reduce: str = "mean",
) -> float:
    scores = torch.as_tensor(scores).view(-1).float()
    scores = torch.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    if scores.numel() == 0:
        return 0.0

    reduce = (reduce or "mean").lower()

    if reduce == "max":
        return float(scores.max().item())

    if reduce == "topk_mean":
        k = min(10, int(scores.numel()))
        return float(torch.topk(scores, k=k).values.mean().item())

    return float(scores.mean().item())


# -----------------------------------------------------------------------------
# Detector builder
# -----------------------------------------------------------------------------
def _resolve_detector_class(model_name: str):
    """
    Resolve detector class lazily from pygod.detector.
    """
    from pygod import detector as pygod_detector

    alias = {
        "DOMINANT": "DOMINANT",
        "CONAD": "CONAD",
        "DONE": "DONE",
        "ANOMALYDAE": "AnomalyDAE",
        "COLA": "CoLA",
        "GAAN": "GAAN",
        "GUIDE": "GUIDE",
    }

    key = str(model_name).upper()
    cls_name = alias.get(key, model_name)

    if not hasattr(pygod_detector, cls_name):
        raise ValueError(
            f"Unsupported legacy detector: {model_name} "
            f"(resolved class='{cls_name}' not found in pygod.detector)"
        )

    return getattr(pygod_detector, cls_name)


def _build_detector(
    model_name: str,
    detector_kwargs: Optional[Dict[str, Any]] = None,
):
    cls = _resolve_detector_class(model_name)

    key = str(model_name).upper()
    kwargs = dict(_DEFAULT_DETECTOR_KWARGS.get(key, {}))
    if detector_kwargs:
        kwargs.update(detector_kwargs)

    return cls(**kwargs)


# -----------------------------------------------------------------------------
# Result structures
# -----------------------------------------------------------------------------
@dataclass
class LegacyRecord:
    model_name: str
    contract_id: str
    score: float
    label: Optional[float]
    score_source: str
    num_scores: int


@dataclass
class LegacyRunOutput:
    model_name: str
    records: List[LegacyRecord]
    skipped: int

    @property
    def contract_ids(self) -> List[str]:
        return [r.contract_id for r in self.records]

    @property
    def scores(self) -> List[float]:
        return [r.score for r in self.records]

    @property
    def labels(self) -> List[Optional[float]]:
        return [r.label for r in self.records]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "contract_ids": self.contract_ids,
            "scores": self.scores,
            "labels": self.labels,
            "score_sources": [r.score_source for r in self.records],
            "num_scores_each": [r.num_scores for r in self.records],
            "skipped": self.skipped,
        }


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------
class LegacyBatchRunner:
    """
    Run PyGOD-style legacy detectors graph-by-graph.

    This version includes:
      - automatic wrapper unwrapping
      - robust score extraction fallback
      - NaN/Inf sanitization
      - node-score -> graph-score reduction
    """
    def __init__(
        self,
        config: Optional[LegacyAdapterConfig] = None,
        *,
        detector_overrides=None,
        score_reduce="mean",
        progress_every=6,
    ):
        if config is not None:
            detector_overrides = getattr(config, "detector_overrides", detector_overrides)
            score_reduce = getattr(config, "score_reduce", score_reduce)
            progress_every = getattr(config, "progress_every", progress_every)
 
        self.detector_overrides = detector_overrides or {}
        self.score_reduce = score_reduce or "mean"
        self.progress_every = max(1, int(progress_every))
        

    def _get_detector_kwargs(self, model_name: str) -> Dict[str, Any]:
        key = str(model_name).upper()
        return dict(self.detector_overrides.get(key, {}))
    
    # Ensure the 'run_all' method exists
    def run_all(self, test_graphs):
        # Your processing logic here; for example:
        scores = []
        for graph in test_graphs:
            score = self.process_graph(graph)
            scores.append(score)
        return {graph.name: s for graph, s in zip(test_graphs, scores)}   # 반드시 dict 형태로 반환해야 합니다.

    # def run_all(self, test_graphs):
    #     scores = []
    #     for g in test_graphs:
    #         scores.append(self._predict(g))
    #     return {g.name: s for g, s in zip(test_graphs, scores)}   # 반드시 dict 형태로 반환해야 합니다.
    
    def process_graph(self, graph):
        # Process your graph and return a score
        pass

    def run_detector(
        self,
        model_name: str,
        graphs: Sequence[Any],
    ) -> LegacyRunOutput:
        items = _safe_list(graphs)

        logger.info("")
        logger.info("[LegacyBatchRunner] === Running %s ===", model_name)

        records: List[LegacyRecord] = []
        skipped = 0
        total = len(items)

        for idx, item in enumerate(items):
            if idx % self.progress_every == 0:
                pct = int((100.0 * idx / total)) if total > 0 else 100
                logger.info("[LegacyRunner:%s] %d/%d (%d%%)", model_name, idx, total, pct)

            data = _unwrap_data(item)
            contract_id = _extract_contract_id(item, data, idx)
            label = _extract_label(item, data)

            if data is None:
                logger.warning(
                    "[LegacyRunner:%s] Skip %s: cannot unwrap graph to PyG Data",
                    model_name,
                    contract_id,
                )
                skipped += 1
                continue

            try:
                data = _prepare_graph_for_detector(data)
            except Exception as exc:
                logger.warning(
                    "[LegacyRunner:%s] Skip %s: invalid graph (%s)",
                    model_name,
                    contract_id,
                    exc,
                )
                skipped += 1
                continue

            try:
                detector = _build_detector(
                    model_name=model_name,
                    detector_kwargs=self._get_detector_kwargs(model_name),
                )
            except Exception as exc:
                logger.exception(
                    "[LegacyRunner:%s] Failed to build detector: %s",
                    model_name,
                    exc,
                )
                raise

            try:
                detector.fit(data)
            except Exception as exc:
                logger.warning(
                    "[LegacyRunner:%s] Skip %s: fit() failed: %r",
                    model_name,
                    contract_id,
                    exc,
                )
                skipped += 1
                continue

            scores, score_src = _extract_detector_scores(detector, data)

            if scores is None or score_src is None:
                available = [
                    name
                    for name in [
                        "decision_scores_",
                        "decision_score_",
                        "decision_function",
                        "predict",
                    ]
                    if hasattr(detector, name)
                ]
                logger.warning(
                    "[LegacyRunner:%s] Skip %s: no usable detector scores after fit() "
                    "(available=%s)",
                    model_name,
                    contract_id,
                    available,
                )
                skipped += 1
                continue

            try:
                graph_score = _reduce_node_scores_to_graph_score(
                    scores,
                    reduce=self.score_reduce,
                )
            except Exception as exc:
                logger.warning(
                    "[LegacyRunner:%s] Skip %s: score reduction failed: %r",
                    model_name,
                    contract_id,
                    exc,
                )
                skipped += 1
                continue

            records.append(
                LegacyRecord(
                    model_name=str(model_name),
                    contract_id=contract_id,
                    score=float(graph_score),
                    label=label,
                    score_source=score_src,
                    num_scores=int(scores.numel()),
                )
            )

            logger.debug(
                "[LegacyRunner:%s] %s score extracted via %s | num_scores=%d | "
                "graph_score=%.6f",
                model_name,
                contract_id,
                score_src,
                int(scores.numel()),
                float(graph_score),
            )

        logger.info(
            "[LegacyRunner:%s] Done. Scored %d contracts. (skipped=%d)",
            model_name,
            len(records),
            skipped,
        )

        return LegacyRunOutput(
            model_name=str(model_name),
            records=records,
            skipped=skipped,
        )

    def run_many(
        self,
        model_names: Sequence[str],
        graphs: Sequence[Any],
    ) -> Dict[str, LegacyRunOutput]:
        out: Dict[str, LegacyRunOutput] = {}
        for model_name in model_names:
            out[str(model_name)] = self.run_detector(model_name, graphs)
        return out


# -----------------------------------------------------------------------------
# Convenience functions
# -----------------------------------------------------------------------------
def run_legacy_detector(
    model_name: str,
    graphs: Sequence[Any],
    *,
    detector_kwargs: Optional[Dict[str, Any]] = None,
    score_reduce: str = "mean",
    progress_every: int = 6,
) -> LegacyRunOutput:
    runner = LegacyBatchRunner(
        detector_overrides={str(model_name).upper(): detector_kwargs or {}},
        score_reduce=score_reduce,
        progress_every=progress_every,
    )
    return runner.run_detector(model_name, graphs)


def run_legacy_detectors(
    model_names: Sequence[str],
    graphs: Sequence[Any],
    *,
    detector_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    score_reduce: str = "mean",
    progress_every: int = 6,
) -> Dict[str, LegacyRunOutput]:
    runner = LegacyBatchRunner(
        detector_overrides=detector_overrides,
        score_reduce=score_reduce,
        progress_every=progress_every,
    )
    return runner.run_many(model_names, graphs)
