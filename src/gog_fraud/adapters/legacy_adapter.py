# legacy_adapter.py
from __future__ import annotations

import logging
import psutil
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

    # Large Graph Partitioning Settings
    max_nodes: int = 4096   ## 지금은 Sample size가 작아서 1500으로 설정했지만 나중에는 3000-5000으로 변경해야함
    large_graph_mode: str = "partition"  # "skip" | "partition"
    partition_size: int = 4096   ## 지금은 Sample size가 작아서 1500으로 설정했지만 나중에는 3000-5000으로 변경해야함
    partition_overlap: float = 0.0
    aggregation_method: str = "max"     # "max" | "topk_mean"
    
    # New: Auto-load best params if available
    use_best_params: bool = False
    best_params_dir: str = "configs/legacy/best_params/"
    chain: str = "polygon"

    # Partition Caching
    partition_cache_dir: str = "../_data/dataset/.cache/partitioned_graphs/"
    cleanup_partition_cache: bool = False

    def __post_init__(self):
        if self.aggregation_method and not self.score_reduce:
            self.score_reduce = self.aggregation_method
        elif self.aggregation_method:
            self.score_reduce = self.aggregation_method

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
    data.num_nodes = data.x.size(0)  # Explicitly set to avoid inference errors
    return data


def _repackage_minimal(data: Data) -> Data:
    """
    Keep only the fields required for the legacy detector to save memory.
    Validates edge_index to prevent CUDA device-side assert (out-of-bounds).
    """
    num_nodes = data.x.size(0)
    edge_index = data.edge_index.long()

    # --- Safety: remove any edges referencing out-of-range node indices ---
    if edge_index.numel() > 0:
        valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes) \
                   & (edge_index[0] >= 0) & (edge_index[1] >= 0)
        if not valid_mask.all():
            n_invalid = (~valid_mask).sum().item()
            logger.debug(
                "[_repackage_minimal] Dropping %d out-of-range edges (num_nodes=%d)",
                n_invalid, num_nodes
            )
            edge_index = edge_index[:, valid_mask]

    clean_data = Data(
        x=data.x.float(),
        edge_index=edge_index,
        num_nodes=num_nodes
    )
    if hasattr(data, "y") and data.y is not None:
        clean_data.y = data.y
        clean_data.num_nodes = num_nodes
        
    return clean_data


def _partition_graph(
    data: Data, 
    partition_size: int, 
    overlap: float = 0.0
) -> List[Data]:
    """
    Split graph into subgraphs using node-chunking / clustering.
    For simplicity, we use induced subgraphs of node chunks.
    """
    from torch_geometric.utils import subgraph
    
    num_nodes = data.num_nodes or data.x.size(0)
    if num_nodes <= partition_size:
        return [data]
    
    # Simple chunking without shuffle for stability/locality
    indices = torch.arange(num_nodes)
    
    subgraphs = []
    # If overlap > 0, we can implementation sliding window, 
    # but for now we follow the "no overlap" user preference.
    step = int(partition_size * (1.0 - overlap))
    if step < 1: step = partition_size
    
    for i in range(0, num_nodes, step):
        chunk = indices[i : i + partition_size]
        if chunk.numel() == 0:
            continue
            
        edge_index, _ = subgraph(chunk, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
        
        # Induced subgraph
        sub_data = Data(
            x=data.x[chunk].clone(),
            edge_index=edge_index,
            num_nodes=len(chunk)
        )
        if hasattr(data, "y") and data.y is not None:
            # If label is per-graph, duplicate it. If per-node, slice it.
            if data.y.numel() == 1:
                sub_data.y = data.y
            elif data.y.numel() == num_nodes:
                sub_data.y = data.y[chunk].clone()
        
        subgraphs.append(sub_data)
        
    return subgraphs


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

    # Legacy detectors run on CPU by design.
    # PyGOD's GPU mode triggers CUDA device-side asserts on partitioned
    # sub-graphs, which permanently corrupts the CUDA context and
    # cascade-fails the subsequent Revision (nGNN) stages.
    # To explicitly enable GPU, set "gpu": 0 in best_params JSON.
    if "gpu" not in kwargs:
        kwargs["gpu"] = -1

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
class LegacyEvalItem:
    contract_id: str
    label: Optional[float]
    data: Optional[Data] = None  # Full graph
    subgraphs: Optional[List[Data]] = None  # Partitioned subgraphs
    is_large: bool = False


@dataclass
class LegacyRunOutput:
    model_name: str
    records: List[LegacyRecord]
    skipped: int
    
    # Extra diagnostics (Added for Large Graph Partition Logic)
    processed_total: int = 0
    full_graph_count: int = 0
    partitioned_graph_count: int = 0
    partition_size_dist: List[int] = None
    failure_reasons: Dict[str, int] = None
    skipped_labels: List[float] = None
    
    # Resource metrics
    max_nodes_processed: int = 0
    peak_ram_mb: float = 0.0
    peak_gpu_mb: float = 0.0

    def __post_init__(self):
        if self.partition_size_dist is None: self.partition_size_dist = []
        if self.failure_reasons is None: self.failure_reasons = {}
        if self.skipped_labels is None: self.skipped_labels = []

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
            "max_nodes_processed": self.max_nodes_processed,
            "peak_ram_mb": self.peak_ram_mb,
            "peak_gpu_mb": self.peak_gpu_mb,
        }


# -----------------------------------------------------------------------------
# Main runner
# -----------------------------------------------------------------------------
def _get_partition_cache_path(
    base_dir: str,
    chain: str,
    contract_id: str,
    size: int,
    overlap: float,
) -> Path:
    from pathlib import Path
    d = Path(base_dir) / chain
    d.mkdir(parents=True, exist_ok=True)
    
    # Sanitize contract_id for filename
    safe_id = str(contract_id).replace("/", "_").replace("\\", "_")
    return d / f"{safe_id}_s{size}_o{overlap}.pt"


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
            self.config = config
            if self.config.use_best_params:
                self._apply_best_params()
        else:
            self.config = LegacyAdapterConfig(
                detector_overrides=detector_overrides,
                score_reduce=score_reduce,
                progress_every=progress_every
            )

        self.detector_overrides = self.config.detector_overrides or {}
        self.score_reduce = self.config.score_reduce or "mean"
        self.progress_every = max(1, int(self.config.progress_every))

    def _apply_best_params(self):
        """
        Load best parameters from the configured directory based on the chain.
        """
        import json
        import os
        from pathlib import Path

        # We assume the user has set the chain in some context, 
        # or we try to detect it. For now, we expect it to be passed
        # OR we rely on a global setting. 
        # Let's add a `chain` field to LegacyAdapterConfig if not already there.
        chain = getattr(self.config, "chain", "polygon").lower()
        path = Path(self.config.best_params_dir) / f"best_params_{chain}.json"
        
        if path.exists():
            try:
                with open(path, 'r') as f:
                    best_dict = json.load(f)
                
                # Merge into detector_overrides
                if not self.config.detector_overrides:
                    self.config.detector_overrides = {}
                
                for model_name, params in best_dict.items():
                    key = str(model_name).upper()
                    if key not in self.config.detector_overrides:
                        self.config.detector_overrides[key] = {}
                    self.config.detector_overrides[key].update(params)
                
                logger.info("[LegacyBatchRunner] Loaded best params for chain '%s' from %s", chain, path)
            except Exception as e:
                logger.error("[LegacyBatchRunner] Failed to load best params from %s: %r", path, e)
        else:
            logger.debug("[LegacyBatchRunner] No best params file found at %s", path)
        

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
        total = len(items)

        logger.info("")
        logger.info("[LegacyBatchRunner] === Running %s (total=%d) ===", model_name, total)
        
        # Optional cache cleanup if requested
        if self.config.cleanup_partition_cache:
            import shutil
            cache_root = Path(self.config.partition_cache_dir) / self.config.chain
            if cache_root.exists():
                logger.info("[LegacyBatchRunner] Cleaning up partition cache for chain '%s'...", self.config.chain)
                shutil.rmtree(cache_root)
            cache_root.mkdir(parents=True, exist_ok=True)
        
        process = psutil.Process()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        output = LegacyRunOutput(
            model_name=str(model_name),
            records=[],
            skipped=0,
            processed_total=total,
            full_graph_count=0,
            partitioned_graph_count=0,
            partition_size_dist=[],
            failure_reasons={},
            skipped_labels=[]
        )

        for idx, item in enumerate(items):
            # Progress update every 10 graphs for better diagnostic visibility
            if idx % 10 == 0 or idx == total - 1:
                pct = int((100.0 * idx / total)) if total > 0 else 100
                logger.info("[LegacyRunner:%s] %d/%d (%d%%)", model_name, idx, total, pct)

            data = _unwrap_data(item)
            contract_id = _extract_contract_id(item, data, idx)
            label = _extract_label(item, data)

            if data is None:
                logger.warning("[LegacyRunner:%s] Skip %s: cannot unwrap", model_name, contract_id)
                output.skipped += 1
                output.failure_reasons["cannot_unwrap"] = output.failure_reasons.get("cannot_unwrap", 0) + 1
                if label is not None: output.skipped_labels.append(label)
                continue

            # ** SAFETY CHECK & PARTITION FALLBACK **
            num_nodes = getattr(data, 'num_nodes', None)
            if num_nodes is None and getattr(data, 'x', None) is not None:
                num_nodes = data.x.size(0)
            
            max_nodes = self.config.max_nodes
            is_large = (num_nodes is not None and num_nodes > max_nodes)

            if is_large:
                if self.config.large_graph_mode == "skip":
                    logger.warning("[LegacyRunner:%s] Skip %s: graph too large (%d)", model_name, contract_id, num_nodes)
                    output.skipped += 1
                    output.failure_reasons["graph_too_large_skipped"] = output.failure_reasons.get("graph_too_large_skipped", 0) + 1
                    if label is not None: output.skipped_labels.append(label)
                    continue
                else:
                    logger.info("[LegacyRunner:%s] %s is large (%d nodes) -> Partitioning", model_name, contract_id, num_nodes)
                    record = self._run_partitioned(model_name, data, contract_id, label)
                    if record:
                        output.records.append(record)
                        output.partitioned_graph_count += 1
                        if num_nodes is not None: output.partition_size_dist.append(num_nodes)
                    else:
                        output.skipped += 1
                        output.failure_reasons["partition_run_failed"] = output.failure_reasons.get("partition_run_failed", 0) + 1
                        if label is not None: output.skipped_labels.append(label)
                    continue

            # Normal path for safe graphs
            try:
                # Prepare/Sanitize
                data = _prepare_graph_for_detector(data)
                record = self._run_single_graph_full(model_name, data, contract_id, label)
                if record:
                    output.records.append(record)
                    output.full_graph_count += 1
                else:
                    logger.warning("[LegacyRunner:%s] %s: full run returned no scores", model_name, contract_id)
                    output.skipped += 1
                    output.failure_reasons["full_run_no_scores"] = output.failure_reasons.get("full_run_no_scores", 0) + 1
                    if label is not None: output.skipped_labels.append(label)
            except Exception as exc:
                logger.warning("[LegacyRunner:%s] Skip %s: %r", model_name, contract_id, exc)
                output.skipped += 1
                output.failure_reasons["exception"] = output.failure_reasons.get("exception", 0) + 1
                if label is not None: output.skipped_labels.append(label)


            # --- Resource Telemetry Tracking ---
            # Node count tracking
            curr_nodes = num_nodes if num_nodes is not None else 0
            if curr_nodes > output.max_nodes_processed:
                output.max_nodes_processed = curr_nodes

            # RAM tracking
            curr_ram = process.memory_info().rss / (1024 * 1024)
            if curr_ram > output.peak_ram_mb:
                output.peak_ram_mb = curr_ram
            
            # GPU tracking
            if torch.cuda.is_available():
                curr_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024)
                if curr_gpu > output.peak_gpu_mb:
                    output.peak_gpu_mb = curr_gpu

        logger.info(
            "[LegacyRunner:%s] Done. Scored %d. Full=%d Part=%d (skipped=%d)",
            model_name, len(output.records), output.full_graph_count, output.partitioned_graph_count, output.skipped
        )
        return output

    def run_on_items(
        self,
        model_name: str,
        eval_items: List[LegacyEvalItem],
    ) -> LegacyRunOutput:
        """
        Run evaluating on pre-processed items to skip all partitioning/unwrap logic.
        Used for fast hyperparameter search.
        """
        total = len(eval_items)
        output = LegacyRunOutput(
            model_name=str(model_name),
            records=[],
            skipped=0,
            processed_total=total,
            full_graph_count=0,
            partitioned_graph_count=0,
            partition_size_dist=[],
            failure_reasons={},
            skipped_labels=[]
        )

        for idx, item in enumerate(eval_items):
            if idx % 50 == 0 or idx == total - 1:
                pct = int((100.0 * idx / total)) if total > 0 else 100
                logger.info("[LegacyRunner:%s] %d/%d (%d%%)", model_name, idx, total, pct)

            try:
                import torch
                import psutil
                process = psutil.Process()
                
                # Pre-run tracking
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                start_ram = process.memory_info().rss / (1024 * 1024)

                if item.is_large and item.subgraphs:
                    # nodes in a subgraph might be limited to partition_size, but the logical size is the sum or original max
                    num_nodes = item.subgraphs[0].num_nodes * len(item.subgraphs) if item.subgraphs else 0
                    if num_nodes > output.max_nodes_processed:
                        output.max_nodes_processed = num_nodes
                        
                    record = self._run_partitioned_direct(model_name, item.subgraphs, item.contract_id, item.label)
                    if record:
                        output.records.append(record)
                        output.partitioned_graph_count += 1
                    else:
                        output.skipped += 1
                elif item.data:
                    num_nodes = item.data.num_nodes if hasattr(item.data, "num_nodes") else item.data.x.size(0)
                    if num_nodes > output.max_nodes_processed:
                        output.max_nodes_processed = num_nodes

                    record = self._run_single_graph_full(model_name, item.data, item.contract_id, item.label)
                    if record:
                        output.records.append(record)
                        output.full_graph_count += 1
                    else:
                        output.skipped += 1
                else:
                    output.skipped += 1
                    
                # Post-run tracking
                end_ram = process.memory_info().rss / (1024 * 1024)
                if end_ram > output.peak_ram_mb:
                    output.peak_ram_mb = end_ram
                if torch.cuda.is_available():
                    peak_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024)
                    if peak_gpu > output.peak_gpu_mb:
                        output.peak_gpu_mb = peak_gpu
                        
            except Exception as e:
                logger.warning("[LegacyRunner:%s] Eval item error: %r", model_name, e)
                print(f"idx={idx} contract_id={item.contract_id} label={item.label} data={item.data}, is_large={item.is_large}")
                output.skipped += 1

        return output

    def _run_partitioned_direct(self, model_name: str, subgraphs: List[Data], contract_id: str, label: Optional[float]) -> Optional[LegacyRecord]:
        import gc
        import torch
        all_node_scores = []
        
        for i, sub_data in enumerate(subgraphs):
            detector = None
            try:
                detector = _build_detector(
                    model_name=model_name,
                    detector_kwargs=self._get_detector_kwargs(model_name),
                )
                detector.fit(sub_data)
                scores, _ = _extract_detector_scores(detector, sub_data)
                if scores is not None:
                    all_node_scores.append(scores.detach().cpu())
            except Exception as exc:
                logger.debug("[LegacyRunner:%s] Partition %d failed for %s: %r", model_name, i, contract_id, exc)
            finally:
                if detector is not None: del detector
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        if not all_node_scores:
            return None
            
        combined_scores = torch.cat(all_node_scores, dim=0)
        agg_method = self.config.aggregation_method.lower()
        if agg_method == "max":
            final_score = float(combined_scores.max().item())
        elif agg_method == "topk_mean":
            k = min(self.config.topk, int(combined_scores.numel()))
            final_score = float(torch.topk(combined_scores, k=k).values.mean().item())
        else:
            final_score = float(combined_scores.mean().item())
            
        return LegacyRecord(
            model_name=str(model_name),
            contract_id=contract_id,
            score=final_score,
            label=label,
            score_source=f"pre_partitioned_{agg_method}",
            num_scores=int(combined_scores.numel()),
        )

    def _run_single_graph_full(self, model_name: str, data: Data, contract_id: str, label: Optional[float]) -> Optional[LegacyRecord]:
        import gc
        detector = None
        try:
            # Minimal repackaging
            clean_data = _repackage_minimal(data)
            
            detector = _build_detector(
                model_name=model_name,
                detector_kwargs=self._get_detector_kwargs(model_name),
            )
            detector.fit(clean_data)
            scores, score_src = _extract_detector_scores(detector, clean_data)

            if scores is None or score_src is None:
                return None

            graph_score = _reduce_node_scores_to_graph_score(
                scores,
                reduce=self.score_reduce,
            )
            return LegacyRecord(
                model_name=str(model_name),
                contract_id=contract_id,
                score=float(graph_score),
                label=label,
                score_source=score_src,
                num_scores=int(scores.numel()),
            )
        finally:
            if detector is not None: del detector
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    def _run_partitioned(self, model_name: str, data: Data, contract_id: str, label: Optional[float]) -> Optional[LegacyRecord]:
        import gc
        import torch
        from pathlib import Path
        
        # 1. Check/Load from cache
        cache_path = _get_partition_cache_path(
            base_dir=self.config.partition_cache_dir,
            chain=self.config.chain,
            contract_id=contract_id,
            size=self.config.partition_size,
            overlap=self.config.partition_overlap
        )
        
        subgraphs = None
        if cache_path.exists():
            try:
                subgraphs = torch.load(cache_path, weights_only=False)
                logger.debug("[LegacyRunner:%s] Cache Hit for %s: %d subgraphs loaded.", model_name, contract_id, len(subgraphs))
            except Exception as e:
                logger.warning("[LegacyRunner:%s] Failed to load partition cache for %s: %r", model_name, contract_id, e)
                
        # 2. Partition if cache miss
        if subgraphs is None:
            subgraphs = _partition_graph(data, self.config.partition_size, self.config.partition_overlap)
            logger.info("[LegacyRunner:%s] Cache Initialized for %s: %d subgraphs generated.", model_name, contract_id, len(subgraphs))
            try:
                torch.save(subgraphs, cache_path)
            except Exception as e:
                logger.warning("[LegacyRunner:%s] Failed to save partition cache for %s: %r", model_name, contract_id, e)
        
        all_node_scores = []
        
        logger.debug("[LegacyRunner:%s] %s split into %d subgraphs", model_name, contract_id, len(subgraphs))
        
        for i, sub_data in enumerate(subgraphs):
            detector = None
            try:
                detector = _build_detector(
                    model_name=model_name,
                    detector_kwargs=self._get_detector_kwargs(model_name),
                )
                detector.fit(sub_data)
                scores, _ = _extract_detector_scores(detector, sub_data)
                
                if scores is not None:
                    # Move to CPU immediately to save GPU memory during loop
                    all_node_scores.append(scores.detach().cpu())
            except Exception as exc:
                logger.debug("[LegacyRunner:%s] Partition %d failed for %s: %r", model_name, i, contract_id, exc)
            finally:
                if detector is not None: del detector
                # Conservative cleanup after EVERY subgraph as requested
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        if not all_node_scores:
            return None
            
        combined_scores = torch.cat(all_node_scores, dim=0)
        
        # User preference: Max or Top-K mean
        agg_method = self.config.aggregation_method.lower()
        if agg_method == "max":
            final_score = float(combined_scores.max().item())
        elif agg_method == "topk_mean":
            k = min(self.config.topk, int(combined_scores.numel()))
            final_score = float(torch.topk(combined_scores, k=k).values.mean().item())
        else:
            final_score = float(combined_scores.mean().item())
            
        return LegacyRecord(
            model_name=str(model_name),
            contract_id=contract_id,
            score=final_score,
            label=label,
            score_source=f"partitioned_{agg_method}",
            num_scores=int(combined_scores.numel()),
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
