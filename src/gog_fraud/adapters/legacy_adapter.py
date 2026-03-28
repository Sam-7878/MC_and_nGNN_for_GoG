# legacy_adapter.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch_geometric.data import Data
from pygod import detector as pygod_detector

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
    

    # def run_all(self, model_name: str, test_graphs: List[Any]) -> Dict[str, float]:
    #         import logging
    #         try:
    #             import pygod.detector as pygod_detector
    #         except ImportError:
    #             logging.getLogger(__name__).error("pygod library is not installed.")
    #             return {}

    #         logger = logging.getLogger(__name__)
    #         scores = []
    #         logger.info(f"[LegacyBatchRunner] Running run_all for {model_name} on {len(test_graphs)} graphs...")
            
    #         # 1. pygod.detector에서 해당 모델(예: DOMINANT) 클래스를 동적으로 가져옵니다.
    #         model_cls = getattr(pygod_detector, model_name, None)
    #         if model_cls is None:
    #             logger.error(f"[LegacyBatchRunner] Unknown detector class: {model_name}")
    #             # 빈 딕셔너리 반환 시 벤치마크 파이프라인에서 에러가 날 수 있으므로 임시 점수 생성
    #             return {getattr(g, "name", f"graph_{i}"): 0.0 for i, g in enumerate(test_graphs)}

    #         # 2. 모델 파라미터(kwargs) 설정
    #         kwargs = {}
    #         # (선택) 모듈 레벨에 정의된 _DEFAULT_DETECTOR_KWARGS가 있다면 가져오기
    #         global_vars = globals()
    #         if "_DEFAULT_DETECTOR_KWARGS" in global_vars:
    #             kwargs.update(global_vars["_DEFAULT_DETECTOR_KWARGS"].get(model_name, {}))
                
    #         cfg = getattr(self, "config", getattr(self, "cfg", None))
    #         if cfg and hasattr(cfg, "detector_overrides") and cfg.detector_overrides:
    #             override = cfg.detector_overrides.get(model_name)
    #             if override:
    #                 kwargs.update(override)

    #         # 3. 각 그래프 순회하며 모델 생성 및 예측 수행
    #         for graph in test_graphs:
    #             try:
    #                 # PyGOD 모델 객체 생성 (비지도 모델이므로 매 그래프 평가 시마다 새 인스턴스가 안전함)
    #                 model = model_cls(**kwargs)
    #             except Exception as e:
    #                 logger.error(f"Failed to initialize model {model_name}: {e}")
    #                 model = None

    #             # process_graph로 모델과 데이터를 넘겨 스코어 반환
    #             score = self.process_graph(model, graph)
    #             scores.append(score)
                
    #         # 4. dict 반환 (TransactionGraph 객체의 속성 활용)
    #         result_dict = {}
    #         for i, (graph, s) in enumerate(zip(test_graphs, scores)):
    #             # graph.name 혹은 tx_hash 속성 등을 식별자로 사용 (없으면 기본값 할당)
    #             graph_id = getattr(graph, "name", getattr(graph, "tx_hash", getattr(graph, "id", f"graph_{i}")))
    #             result_dict[graph_id] = s
                
    #         return result_dict

    
    # def process_graph(self, model: Any, graph_item: Any) -> float:
    #         import torch
    #         import numpy as np
    #         import logging
    #         from torch_geometric.data import Data

    #         logger = logging.getLogger(__name__)

    #         # 1. 원본 데이터 추출
    #         raw_data = getattr(graph_item, "graph", graph_item)
    #         if raw_data is None or not hasattr(raw_data, "x") or raw_data.x is None:
    #             return 0.0

    #         try:
    #             # =========================================================
    #             # [핵심 수정] PyGOD 내부의 NeighborLoader가 딕셔너리 등을
    #             # 슬라이싱(slice)하려다 에러가 나는 것을 원천 차단하기 위해,
    #             # 필수 텐서만 포함된 순수 Data 객체로 재포장(Sanitize) 합니다.
    #             # =========================================================
    #             clean_data = Data(
    #                 x=raw_data.x.float(),
    #                 edge_index=raw_data.edge_index.long()
    #             )
    #             # 타겟 라벨 복사
    #             if hasattr(raw_data, 'y') and raw_data.y is not None:
    #                 clean_data.y = raw_data.y
                
    #             # PyGOD의 특정 모델들은 num_nodes 속성을 명시적으로 요구함
    #             if hasattr(raw_data, 'num_nodes'):
    #                 clean_data.num_nodes = raw_data.num_nodes
    #             else:
    #                 clean_data.num_nodes = raw_data.x.size(0)

    #             # =========================================================

    #             # 2. 모델 학습 (fit) 및 예측 (predict)
    #             if model is not None and hasattr(model, "fit"):
    #                 model.fit(clean_data)  # 정제된 clean_data 사용

    #             if model is not None and hasattr(model, "decision_function"):
    #                 node_scores = model.decision_function(clean_data)
    #             elif model is not None and hasattr(model, "predict_proba"):
    #                 probs = model.predict_proba(clean_data)
    #                 node_scores = probs[:, 1] if probs.ndim > 1 else probs
    #             elif model is not None and hasattr(model, "predict"):
    #                 node_scores = model.predict(clean_data)
    #             else:
    #                 logger.warning("[LegacyBatchRunner] Valid model object not found or has no predict function.")
    #                 return 0.0

    #             # 3. Tensor -> Numpy 변환
    #             if isinstance(node_scores, torch.Tensor):
    #                 node_scores = node_scores.detach().cpu().numpy()
                    
    #             node_scores = np.array(node_scores, dtype=np.float32)
                
    #             if node_scores.size == 0:
    #                 return 0.0

    #             # 4. 노드별 스코어를 그래프 1개의 스코어로 축소 (Reduce)
    #             cfg = getattr(self, "config", getattr(self, "cfg", None))
    #             reduce_method = getattr(cfg, "score_reduce", "mean") if cfg else "mean"
                
    #             if reduce_method == "mean":
    #                 graph_score = np.mean(node_scores)
    #             elif reduce_method == "max":
    #                 graph_score = np.max(node_scores)
    #             elif reduce_method == "sum":
    #                 graph_score = np.sum(node_scores)
    #             elif reduce_method == "topk":
    #                 k = getattr(cfg, "topk", 3) if cfg else 3
    #                 k = min(int(k), len(node_scores))
    #                 topk_scores = np.sort(node_scores)[-k:]
    #                 graph_score = np.mean(topk_scores)
    #             else:
    #                 graph_score = np.mean(node_scores)

    #             return float(graph_score)

    #         except Exception as e:
    #             # traceback을 포함하여 어떤 에러인지 더 명확하게 찍어줍니다.
    #             logger.error(f"[LegacyBatchRunner] process_graph Error: {e}", exc_info=True)
    #             return 0.0


    # --- End of dummy block ---


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
