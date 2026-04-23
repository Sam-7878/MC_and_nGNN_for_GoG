# src/gog_fraud/pipelines/run_fraud_benchmark.py
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import torch
import yaml
import psutil

from gog_fraud.adapters.legacy_adapter import LegacyAdapterConfig, LegacyBatchRunner
from gog_fraud.data.io.dataset import FraudDataset
from gog_fraud.evaluation.benchmark import BenchmarkTable, evaluate_benchmark
from gog_fraud.models.level1.model import Level1Model
from gog_fraud.models.level2.model import Level2Model
from gog_fraud.training.loops.level1 import Level1Trainer
from gog_fraud.training.loops.level2 import Level2Trainer
from gog_fraud.training.loops.level1 import Level1Trainer
from gog_fraud.models.level1.model import Level1Model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


log = logging.getLogger(__name__)


def _safe_len(obj: Any) -> Optional[int]:
    try:
        return len(obj)
    except Exception:
        return None


def _is_empty(obj: Any) -> bool:
    n = _safe_len(obj)
    return n == 0


def _is_none_or_empty(obj: Any) -> bool:
    return obj is None or _is_empty(obj)


def _get_dataset_attr(dataset, *names, default=None):
    for name in names:
        if hasattr(dataset, name):
            value = getattr(dataset, name)
            if value is not None:
                return value
    return default


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _nested_get(cfg, *keys, default=None):
    cur = cfg
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, None)
        else:
            cur = getattr(cur, key, None)
    return default if cur is None else cur


def _resolve_device(cfg):
    device = (
        _nested_get(cfg, "training", "device")
        or _nested_get(cfg, "trainer", "device")
        or _nested_get(cfg, "model", "device")
        or _cfg_get(cfg, "device", "cpu")
    )
    return str(device)


def _build_level1_model(cfg):
    """
    Level1 모델을 config 기반으로 생성한다.
    우선 Level1Model.from_config(cfg)를 사용하고,
    필요시 device로 이동한다.
    """
    model_cfg = (
        _nested_get(cfg, "model", "level1")
        or _nested_get(cfg, "level1")
        or _nested_get(cfg, "model")
        or cfg
    )

    return Level1Model.from_config(model_cfg).to(_resolve_device(cfg))


def _normalize_graph_splits(dataset):
    train_graphs = _get_dataset_attr(dataset, "train_graphs", "train", default=[])
    valid_graphs = _get_dataset_attr(dataset, "valid_graphs", "val_graphs", "valid", "val", default=None)
    test_graphs = _get_dataset_attr(dataset, "test_graphs", "test", default=[])
    labels = _get_dataset_attr(dataset, "labels", "label_dict", default=None)
    global_graph = _get_dataset_attr(dataset, "global_graph", default=None)

    if _is_empty(valid_graphs):
        valid_graphs = None

    return train_graphs, valid_graphs, test_graphs, labels, global_graph


def _record_skip(table, model_name: str, reason: str, setting: str):
    log.warning("[%s] Skipping: %s", model_name, reason)
    try:
        if isinstance(table, list):
            table.append({
                "setting": setting,
                "model": model_name,
                "status": "skipped",
                "reason": reason,
            })
    except Exception:
        pass


def _record_scores(table, model_name: str, scores: dict):
    try:
        if isinstance(table, list):
            row = {"model": model_name, "status": "success"}
            if isinstance(scores, dict):
                row.update(scores)
            table.append(row)
    except Exception as e:
        log.warning(f"Could not record scores for {model_name}: {e}")


# ============================================================
# [FIX 1] _call_level1_trainer_fit (line ~85~120)
# - label_dict 중복 전달 제거
# - split 키워드 제거
# ============================================================
def _call_level1_trainer_fit(
    trainer,
    train_graphs: list,
    valid_graphs: list,
    label_dict: dict,
    cfg: dict,
) -> dict:
    l1_cfg = cfg.get("level1", {})

    log.debug(
        f"[_call_level1_trainer_fit] "
        f"train={len(train_graphs)}, valid={len(valid_graphs)}, "
        f"label_dict type={type(label_dict).__name__}, "
        f"l1_cfg={l1_cfg}"
    )

    # label_dict 타입 검증
    if not isinstance(label_dict, dict):
        raise TypeError(
            f"[_call_level1_trainer_fit] label_dict must be dict, "
            f"got {type(label_dict).__name__}"
        )

    return trainer.fit(
        train_graphs=train_graphs,
        valid_graphs=valid_graphs,
        label_dict=label_dict,          # ✅ keyword 하나만 사용
        epochs=l1_cfg.get("epochs", 50),
        lr=l1_cfg.get("lr", 1e-3),
        batch_size=l1_cfg.get("batch_size", 32),
    )


# ------------------------------------------------------------------
# 1) Level-1 trainer builder
# ------------------------------------------------------------------
def _build_level1_trainer(cfg: dict) -> "Level1Trainer":
    """
    cfg['level1'] 하위 키를 읽어 Level1Trainer 인스턴스를 반환.
    """

    l1_cfg = cfg.get("level1", {})
    
    model_cfg = (
        _nested_get(cfg, "model", "level1")
        or _nested_get(cfg, "level1")
        or _nested_get(cfg, "model")
        or cfg
    )
    
    model = Level1Model.from_config(model_cfg)
    
    # Detect CUDA error state and fall back to CPU if needed
    requested_device = cfg.get("device", "cuda")
    if requested_device == "cuda" and torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()  # Probe: will raise if CUDA context is corrupted
        except RuntimeError as e:
            log.warning(
                f"[Benchmark] CUDA unavailable (possibly corrupted by Legacy stage): {e}. "
                "Falling back to CPU for Revision models."
            )
            requested_device = "cpu"

    return Level1Trainer(
        model=model,
        cfg=_nested_get(cfg, "training", "level1") or _nested_get(cfg, "trainer", "level1") or cfg,
        optimizer=_build_optimizer(model, l1_cfg),
        device=requested_device,
    )


# ------------------------------------------------------------------
# 2) Level-2 trainer builder
# ------------------------------------------------------------------
def _build_level2_trainer(cfg: dict, l1_model=None) -> "Level2Trainer":
    """
    cfg['level2'] 하위 키를 읽어 Level2Trainer 인스턴스를 반환.
    """
    from gog_fraud.training.loops.level2 import Level2Trainer
    from gog_fraud.models.level2.model import Level2Model

    l2_cfg = cfg.get("level2", {})
    
    model_cfg = (
        _nested_get(cfg, "model", "level2")
        or _nested_get(cfg, "level2")
        or _nested_get(cfg, "model")
        or cfg
    )
    
    import copy
    model_cfg = copy.deepcopy(model_cfg)
    if l1_model is not None and "in_dim" not in model_cfg:
        l1_out_dim = getattr(l1_model, "out_dim", 256)
        # Level 1 embedding + score
        model_cfg["in_dim"] = l1_out_dim + 1
    
    model = Level2Model.from_config(model_cfg)
    
    cfg_obj = _nested_get(cfg, "training", "level2") or _nested_get(cfg, "trainer", "level2") or cfg

    # Detect CUDA error state and fall back to CPU if needed
    requested_device = cfg.get("device", "cuda")
    if requested_device == "cuda" and torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()  # Probe: will raise if CUDA context is corrupted
        except RuntimeError as e:
            log.warning(
                f"[Benchmark] CUDA unavailable for L2 trainer: {e}. "
                "Falling back to CPU."
            )
            requested_device = "cpu"

    return Level2Trainer(
        model=model,
        optimizer=_build_optimizer(model, l2_cfg),
        cfg=cfg_obj,
        device=requested_device,
    )


# Replaced by consolidated versions later in the file





def _build_l2_dynamic_loader_builder(l1_model, cfg):
    def loader_builder(split, ids, label_dict=None, **kwargs):
        from gog_fraud.training.loops.level1 import _prepare_level1_loader
        from gog_fraud.data.level2.relation_builder import build_level2_graph, RelationBuilderConfig
        from torch_geometric.loader import DataLoader as PyGDataLoader
        import torch

        # Determine default chunk size (configured as 16 by user)
        default_chunk = int(_cfg_get(cfg, "eval_chunk_size", _nested_get(cfg, "level2", "eval_chunk_size") or 16))
        
        if split == "train":
            chunk_size = int(_cfg_get(cfg, "train_chunk_size", _nested_get(cfg, "level2", "train_chunk_size") or default_chunk))
            # Just split sequentially for training if needed, or shuffle
            import random
            shuffled_ids = list(ids)
            random.shuffle(shuffled_ids)
            id_chunks = [shuffled_ids[i : i + chunk_size] for i in range(0, len(shuffled_ids), chunk_size)]
        else:
            # INTERLEAVED CHUNKING for Evaluation (Better relational context than pure stratification)
            chunk_size = default_chunk
            
            # Robust label lookup for splitting
            def _get_label(item):
                if label_dict is None: return 0
                cid = str(getattr(item, "contract_id", item)).strip().lower()
                return label_dict.get(cid, label_dict.get(getattr(item, "contract_id", item), 0))

            pos_ids = [i for i in ids if _get_label(i) == 1]
            neg_ids = [i for i in ids if _get_label(i) == 0]
            
            id_chunks = []
            max_len = max(len(pos_ids), len(neg_ids))
            interleaved = []
            for i in range(max_len):
                if i < len(pos_ids): interleaved.append(pos_ids[i])
                if i < len(neg_ids): interleaved.append(neg_ids[i])
            
            # Now create chunks from interleaved list
            id_chunks = [interleaved[i : i + chunk_size] for i in range(0, len(interleaved), chunk_size)]

        l2_graphs = []
        for chunk_ids in id_chunks:
            if not chunk_ids: continue
            try:
                loader = _prepare_level1_loader(
                    chunk_ids, split_name=split, batch_size=128, shuffle=False, label_dict=label_dict, num_workers=0
                )
            except Exception as e:
                log.warning("[Dynamic L2 Builder] Failed to prepare L1 loader for chunk: %s", e)
                continue

            if l1_model is None:
                log.error("[Dynamic L2 Builder] l1_model is None. Cannot build Level 2 graph without Level 1 embeddings.")
                return None

            l1_model.eval()
            device = next(l1_model.parameters()).device
            
            all_emb, all_score, all_logits, all_id, all_label = [], [], [], [], []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    out = l1_model(batch)
                    all_emb.append(out.embedding.cpu())
                    all_score.append(out.score.cpu().view(-1, 1))
                    all_logits.append(out.logits.cpu().view(-1, 1))
                    all_id.append(out.graph_id.cpu() if hasattr(out, "graph_id") and out.graph_id is not None else torch.zeros(out.score.size(0), dtype=torch.long))
                    if getattr(out, "label", None) is not None:
                        all_label.append(out.label.cpu().view(-1, 1))

            if not all_emb:
                continue

            # Numerical stability: clamp logits and scores
            embs = torch.cat(all_emb, dim=0)
            logits_cat = torch.cat(all_logits, dim=0).clamp(min=-10.0, max=10.0)
            scores_cat = torch.cat(all_score, dim=0).clamp(min=1e-7, max=1.0 - 1e-7)

            # Record names/ids for tracking back to test set
            chunk_contract_ids = [getattr(cid, "contract_id", str(cid)) for cid in chunk_ids]
            
            bundle = {
                "embedding": embs,
                "score": scores_cat,
                "logits": logits_cat,
                "graph_id": torch.cat(all_id, dim=0),
                "contract_id": chunk_contract_ids,
            }
            if all_label:
                bundle["label"] = torch.cat(all_label, dim=0)
                
            rel_cfg = RelationBuilderConfig()
            if "relation_modes" in cfg.get("level2", {}):
                rel_cfg.relation_modes = cfg["level2"]["relation_modes"]
                
            try:
                l2_graph = build_level2_graph(bundle, rel_cfg)
                l2_graph.contract_id = chunk_contract_ids
                l2_graphs.append(l2_graph)
            except Exception as e:
                log.error("[Dynamic L2 Builder] Failed to build L2 graph chunk: %s", e)

        if not l2_graphs:
            return None
        
        return PyGDataLoader(l2_graphs, batch_size=1, shuffle=(split == "train"))

    return loader_builder

def _call_level2_trainer_fit(
    trainer,
    *,
    l1_model=None,
    global_graph=None,
    train_ids=None,
    valid_ids=None,
    labels=None,
    train_loader=None,
    valid_loader=None,
    loader_builder=None,
    cfg=None,
    **kwargs,
):
    if global_graph is None:
        log.warning("[Revision L2] global_graph is None. trainer.fit() skipped.")
        return {
            "history": [],
            "best_score": None,
            "best_metric": None,
            "best_mode": None,
            "epochs_ran": 0,
            "skipped": True,
            "reason": "global_graph_is_none",
        }

    if _is_none_or_empty(train_loader) and _is_none_or_empty(train_ids):
        log.warning("[Revision L2] Empty training split. trainer.fit() skipped.")
        return {
            "history": [],
            "best_score": None,
            "best_metric": None,
            "best_mode": None,
            "epochs_ran": 0,
            "skipped": True,
            "reason": "empty_train_ids",
        }

    if valid_loader is not None and _is_empty(valid_loader):
        valid_loader = None
    if valid_ids is not None and _is_empty(valid_ids):
        valid_ids = None

    return trainer.fit(
        l1_model=l1_model,
        global_graph=global_graph,
        train_ids=train_ids,
        valid_ids=valid_ids,
        label_dict=labels,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loader_builder=loader_builder,
        **kwargs,
    )



# ---------------------------------------------------------------------------
# config helpers
# ---------------------------------------------------------------------------
def _cfg_to_dict(cfg: Any) -> dict:
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
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    return {}


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[key] = value
        return value

    def __setattr__(self, key, value):
        self[key] = value

    def copy(self):
        return AttrDict(super().copy())


def _cfg_to_attrdict(cfg: Any) -> AttrDict:
    data = _cfg_to_dict(cfg)

    def _convert(v):
        if isinstance(v, dict):
            return AttrDict({kk: _convert(vv) for kk, vv in v.items()})
        if isinstance(v, list):
            return [_convert(x) for x in v]
        return v

    return AttrDict({k: _convert(v) for k, v in data.items()})


# Replaced by consolidated version at top


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# dataset split helpers
# ---------------------------------------------------------------------------
def _get_split_graphs(dataset, cfg, setting):
    train, valid, test = dataset.train_graphs, dataset.valid_graphs, dataset.test_graphs
    smoke_val = _cfg_get(cfg, "smoke_test", default=None)
    if smoke_val is None:
        smoke_val = _nested_get(cfg, "dataset", "smoke_test", default=False)
    smoke_test = bool(smoke_val)

    norm_val = _cfg_get(cfg, "normalize_features", default=None)
    if norm_val is None:
        norm_val = _nested_get(cfg, "dataset", "normalize_features", default=False)
    normalize_features = bool(norm_val)

    if smoke_test:
        train_limit = max(len(train) // 25, 1)
        valid_limit = max(len(valid) // 10, 1)
        test_limit = max(len(test) // 5, 1)
        train = train[:train_limit]
        valid = valid[:valid_limit]
        test = test[:test_limit]

    def _filter_mismatched_graphs(g_list):
        if not g_list: return g_list
        # Reference dimensions from first graph
        ref_x_dim = None
        ref_edge_dim = None
        ref_struct_dim = None
        
        filtered = []
        for g in g_list:
            data = g.graph if hasattr(g, "graph") else g
            if not hasattr(data, "x") or data.x is None:
                continue
            
            x_dim = data.x.size(-1)
            # Edge attr check
            e_dim = data.edge_attr.size(-1) if hasattr(data, "edge_attr") and data.edge_attr is not None else 0
            # Struct feat check
            struct_dim = data.struct_feat.size(-1) if hasattr(data, "struct_feat") and data.struct_feat is not None else 0
            
            # [Fix] Remove problematic single-letter attributes (like 's') if they exist, 
            # to prevent PyG collation errors.
            for k in list(data.keys()):
                if k == "s":
                    del data[k]

            if ref_x_dim is None:
                ref_x_dim = x_dim
                ref_edge_dim = e_dim
                ref_struct_dim = struct_dim
            
            if x_dim == ref_x_dim and e_dim == ref_edge_dim and struct_dim == ref_struct_dim:
                filtered.append(g)
            else:
                log.warning(f"Filter drop graph: x={x_dim}(ref={ref_x_dim}), e={e_dim}(ref={ref_edge_dim}), s={struct_dim}(ref={ref_struct_dim})")
        return filtered

    train = _filter_mismatched_graphs(train)
    valid = _filter_mismatched_graphs(valid)
    test = _filter_mismatched_graphs(test)

    # [Normalization]
    if normalize_features and train:
        log.info("[Normalization] Applying standard scaling to node features x …")
        all_x = []
        for g in train:
            data = g.graph if hasattr(g, "graph") else g
            all_x.append(data.x)
        all_x_cat = torch.cat(all_x, dim=0)
        mean = all_x_cat.mean(dim=0, keepdim=True)
        std = all_x_cat.std(dim=0, keepdim=True) + 1e-6
        
        def _apply_norm(g_list):
            for g in g_list:
                data = g.graph if hasattr(g, "graph") else g
                data.x = (data.x - mean) / std

        _apply_norm(train)
        _apply_norm(valid)
        _apply_norm(test)

    log.info(
        f"[get_split_graphs] smoke={smoke_test}, "
        f"train={len(train)}, valid={len(valid)}, test={len(test)}"
    )

    return train, valid, test



def _get_split_ids(dataset: FraudDataset, *names: str):
    last_exc = None
    for name in names:
        try:
            ids_ = dataset.split_ids(name)
            if ids_ is not None:
                return ids_
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    return []


# ---------------------------------------------------------------------------
# debug graph limiting
# ---------------------------------------------------------------------------
def _maybe_limit_graphs(graphs, cfg, split_name: str):
    debug_cfg = _cfg_get(cfg, "debug", {}) or {}
    if not _cfg_get(debug_cfg, "enabled", False):
        return graphs

    limit = _cfg_get(debug_cfg, f"max_{split_name}_graphs", None)
    if limit is None:
        return graphs

    limit = int(limit)
    if limit < len(graphs):
        log.info(
            f"[Benchmark] Smoke mode: limiting {split_name} graphs "
            f"from {len(graphs)} to {limit}"
        )
        return graphs[:limit]
    return graphs


# ---------------------------------------------------------------------------
# score helpers
# ---------------------------------------------------------------------------
def _extract_score_tensor(out) -> torch.Tensor:
    if hasattr(out, "score"):
        score = out.score
    elif isinstance(out, dict):
        score = (
            out.get("score", None)
            or out.get("anomaly_score", None)
            or out.get("logit", None)
            or out.get("logits", None)
            or out.get("prob", None)
            or out.get("probs", None)
        )
        if score is None:
            raise KeyError("No score-like field in model output")
    else:
        score = out

    if not torch.is_tensor(score):
        score = torch.tensor(score, dtype=torch.float32)

    return torch.nan_to_num(score.reshape(-1).detach().cpu(), nan=0.0, posinf=1.0, neginf=0.0)


def _extract_scalar_score(out) -> float:
    score = _extract_score_tensor(out)
    if score.numel() == 0:
        return 0.0
    if score.numel() == 1:
        return float(score.item())
    return float(score.mean().item())


# ---------------------------------------------------------------------------
# optimizer helper
# ---------------------------------------------------------------------------
def _build_optimizer(model, cfg: dict):
    lr = float(_cfg_get(cfg, "lr", _cfg_get(cfg, "learning_rate", 1e-3)))
    weight_decay = float(_cfg_get(cfg, "weight_decay", 0.0))
    opt_name = str(_cfg_get(cfg, "optimizer", "adam")).lower()

    if opt_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=float(_cfg_get(cfg, "momentum", 0.9)),
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )


# ---------------------------------------------------------------------------
# benchmark append helper
# ---------------------------------------------------------------------------
def _append_result(
    scores: Dict[str, float],
    dataset: FraudDataset,
    test_graphs,
    model_name: str,
    setting: str,
    cfg: dict,
    table: BenchmarkTable,
) -> None:
    contract_ids = [g.contract_id for g in test_graphs]

    filtered = []
    for cid in contract_ids:
        if cid not in dataset.labels:
            continue
        filtered.append((float(scores.get(cid, 0.0)), int(dataset.labels[cid])))

    if not filtered:
        log.warning(f"[{model_name}] No valid scores found.")
        return

    ys_arr = [x[0] for x in filtered]
    yt_arr = [x[1] for x in filtered]

    result = evaluate_benchmark(
        y_true=yt_arr,
        y_score=ys_arr,
        model_name=model_name,
        setting=setting,
        threshold=float(_cfg_get(cfg, "threshold", 0.5)),
        k_list=_cfg_get(cfg, "k_list", [10, 20, 50]),
        bootstrap=bool(_cfg_get(cfg, "bootstrap", True)),
    )
    table.add(result)
    log.info(str(result))


# ---------------------------------------------------------------------------
# (A) legacy
# ---------------------------------------------------------------------------
def run_legacy_baselines(
    dataset: FraudDataset,
    cfg: dict,
    table: BenchmarkTable,
    setting: str,
) -> None:
    legacy_cfg = _cfg_get(cfg, "legacy", {}) or {}
    model_names = _cfg_get(legacy_cfg, "models", ["DOMINANT", "DONE", "GAE", "AnomalyDAE", "CoLA"],)
    model_type = model_names[0] if model_names else "DOMINANT" # 기본값 설정, 기존 legacy에서는 model_names가 아니라 model_type이었음

    # Extract chain name from main cfg -> dataset -> chain
    dataset_cfg = _cfg_get(cfg, "dataset", {}) or {}
    chain_name = dataset_cfg.get("chain", "polygon").lower()

    base_adapter_cfg    = LegacyAdapterConfig(
        agg_method      = _cfg_get(legacy_cfg, "agg_method", "max"),
        topk            = int(_cfg_get(legacy_cfg, "topk", 3)),
        normalize_score = bool(_cfg_get(legacy_cfg, "normalize_score", True)),
        gpu             = int(_cfg_get(legacy_cfg, "gpu", 0)),
        hid_dim         = int(_cfg_get(legacy_cfg, "hid_dim", 16)),
        num_layers      = int(_cfg_get(legacy_cfg, "num_layers", 2)),
        epoch           = int(_cfg_get(legacy_cfg, "epoch", 100)),
        lr              = float(_cfg_get(legacy_cfg, "lr", 0.003)),
        use_best_params = True,
        chain           = chain_name
    )

    # ==================================================
    # 수정된 split 그래프 가져오기 부분
    # - _get_split_graphs를 한 번만 호출 (split-specific 제거)
    # - 각 split에 _maybe_limit_graphs 적용
    # - setting은 상위에서 전달된 값으로 사용 (예: "strict")
    # ==================================================

    # _get_split_graphs 한 번 호출로 모든 split 가져오기
    train_graphs, valid_graphs, test_graphs = _get_split_graphs(dataset, cfg, setting)

    # 각 split에 _maybe_limit_graphs 적용 (기존 로직 유지)
    train_graphs = _maybe_limit_graphs(train_graphs, cfg, "train")
    valid_graphs = _maybe_limit_graphs(valid_graphs, cfg, "val")  # "valid" 대신 "val" 사용 (기존 코드와 일치)
    test_graphs  = _maybe_limit_graphs(test_graphs, cfg, "test")
    log.info('finished loading and maybe limiting graphs for legacy baselines')

    if not test_graphs:
        log.warning("[Legacy] No test graphs found!")
        return
    
    try:
        import time
        _t0 = time.perf_counter()
        batch = LegacyBatchRunner(
            config=             base_adapter_cfg,
            detector_overrides= base_adapter_cfg.detector_overrides,
            score_reduce=       base_adapter_cfg.score_reduce,
            progress_every=     base_adapter_cfg.progress_every)
        
        log.info('***before batch.run_many for legacy baselines, test_graphs count: %d', len(test_graphs))
        all_scores = batch.run_many(model_names=model_names, graphs=test_graphs)

        if not all_scores:
            log.warning(f"[Legacy] ---- No scores returned from batch runner!")
            return
        
        log.info(f"[Legacy] Models run: {list(all_scores.keys())}")

        for model_name, run_output in all_scores.items():
            score_dict = {r.contract_id: r.score for r in run_output.records}
            contract_ids = [g.contract_id for g in test_graphs]

            filtered = [(float(score_dict.get(cid, 0.0)), int(dataset.labels[cid])) for cid in contract_ids if cid in dataset.labels] 
            log.info(f"Filtered scores for model {model_name}: {filtered[:3]}... (total {len(filtered)})")

            if not filtered:
                log.warning(f"[Legacy:{model_name}] No valid scores found.")
                continue

            ys_arr = [x[0] for x in filtered]
            yt_arr = [x[1] for x in filtered]

            result = evaluate_benchmark(
                y_true=yt_arr,
                y_score=ys_arr,
                model_name=f"Legacy_{model_name}",
                setting=setting,
                threshold=float(_cfg_get(cfg, "threshold", 0.5)),
                k_list=_cfg_get(cfg, "k_list", [10, 20, 50]),
                bootstrap=bool(_cfg_get(cfg, "bootstrap", True)),
                max_nodes_processed=run_output.max_nodes_processed,
                peak_ram_mb=run_output.peak_ram_mb,
                peak_gpu_mb=run_output.peak_gpu_mb,
                elapsed_sec=time.perf_counter() - _t0,
            )
            table.add(result)
            log.info(str(result))

    except Exception as e:
        print(f"Error on processing: {e}")
        
# ---------------------------------------------------------------------------
# (B) revision l1
# ---------------------------------------------------------------------------

from torch_geometric.loader import DataLoader as PyGDataLoader

def _to_level1_loader(graphs, batch_size, shuffle):
    data_list = [g.graph if hasattr(g, "graph") else g for g in graphs]
    return PyGDataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

 
# ============================================================
# [FIX 2] run_revision_l1 (line ~540~560)
# - smoke 모드 split size 로그 추가
# - _call_level1_trainer_fit 올바르게 호출
# ============================================================
def run_revision_l1(dataset, cfg, table, setting):
    import time
    _t0 = time.perf_counter()
    log.info("[Revision L1] Training Level1 model …")
 
    train_graphs, valid_graphs, test_graphs = _get_split_graphs(dataset, cfg, setting)
    log.info(
        f"[Revision L1] split sizes: "
        f"train={len(train_graphs)}, valid={len(valid_graphs)}, test={len(test_graphs)}"
    )
 
    trainer = _build_level1_trainer(cfg)
 
    try:
        fit_out = _call_level1_trainer_fit(
            trainer=trainer,
            train_graphs=train_graphs,
            valid_graphs=valid_graphs,
            label_dict=dataset.labels,  # ✅ dict 타입 확인
            cfg=cfg,
        )
    except Exception as e:
        log.error(f"[Revision L1] fit failed: {e}", exc_info=True)
        raise
 
    # Resource Tracking
    import psutil
    process = psutil.Process()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    # Calculate Max Nodes
    max_nodes = 0
    for g in test_graphs:
        n = g.graph.num_nodes if hasattr(g, "graph") else g.num_nodes
        if n > max_nodes: max_nodes = n

    # 평가
    metrics, yt, ys = trainer.evaluate(test_graphs, label_dict=dataset.labels, return_preds=True)
    
    # Capture Peak Metrics
    peak_ram = process.memory_info().rss / (1024 * 1024)
    peak_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    from gog_fraud.evaluation.benchmark import evaluate_benchmark
    res = evaluate_benchmark(
        y_true=yt, 
        y_score=ys, 
        model_name="Revision-L1", 
        setting=setting,
        max_nodes_processed=max_nodes,
        peak_ram_mb=peak_ram,
        peak_gpu_mb=peak_gpu,
        elapsed_sec=time.perf_counter() - _t0,
    )
    if hasattr(table, "add"):
        table.add(res)
    return metrics


# ============================================================
# [FIX 3] run_revision_l1_l2 (line ~610~630)
# - label_dict 중복 전달 제거
# - fallback 직접 호출 제거 (동일 에러 반복 방지)
# ============================================================
def run_revision_l1_l2(dataset, cfg, table, setting):
    import time
    _t0 = time.perf_counter()
    log.info("[Revision L1+L2] Training Level1 + Level2 …")
 
    train_graphs, valid_graphs, test_graphs = _get_split_graphs(dataset, cfg, setting)
 
    # Level1 Trainer
    l1_trainer = _build_level1_trainer(cfg)
    try:
        l1_fit_out = _call_level1_trainer_fit(
            trainer=l1_trainer,
            train_graphs=train_graphs,
            valid_graphs=valid_graphs,
            label_dict=dataset.labels,
            cfg=cfg,
        )
    except Exception as e:
        log.error(f"[Revision L1+L2] Level1 fit failed: {e}", exc_info=True)
        raise
 
    # Level2 Trainer
    l2_trainer = _build_level2_trainer(cfg, l1_trainer.model)
    l2_fit_out = _call_level2_trainer_fit(
        trainer=l2_trainer,
        l1_model=l1_trainer.model,
        cfg=cfg,
        train_ids=train_graphs,
        valid_ids=valid_graphs,
        labels=dataset.labels,
        global_graph=dataset.global_graph,
        loader_builder=_build_l2_dynamic_loader_builder(l1_trainer.model, cfg),
    )
 
    # Resource Tracking
    process = psutil.Process()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Calculate Max Nodes
    max_nodes = 0
    for g in test_graphs:
        n = g.num_nodes if hasattr(g, "num_nodes") else (g.graph.num_nodes if hasattr(g, "graph") else 0)
        if n > max_nodes: max_nodes = n

    metrics, yt, ys = l2_trainer.evaluate(
        test_graphs, 
        label_dict=dataset.labels,
        global_graph=dataset.global_graph,      # ✅ Added missing global_graph
        loader_builder=_build_l2_dynamic_loader_builder(l1_trainer.model, cfg),
        return_preds=True
    )

    # Capture Peak Metrics
    peak_ram = process.memory_info().rss / (1024 * 1024)
    peak_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    from gog_fraud.evaluation.benchmark import evaluate_benchmark
    res = evaluate_benchmark(
        y_true=yt, 
        y_score=ys, 
        model_name="Revision-L1+L2", 
        setting=setting,
        max_nodes_processed=max_nodes,
        peak_ram_mb=peak_ram,
        peak_gpu_mb=peak_gpu,
        elapsed_sec=time.perf_counter() - _t0,
    )
    if hasattr(table, "add"):
        table.add(res)
    return metrics


# ============================================================
# [FIX 4] run_revision_full (line ~720~730)
# - 동일하게 _call_level1_trainer_fit keyword 통일
# ============================================================
def run_revision_full(dataset, cfg, table, setting):
    import time
    _t0 = time.perf_counter()
    log.info("[Revision Full] Training Level1 + Level2 + Fusion …")
 
    train_graphs, valid_graphs, test_graphs = _get_split_graphs(dataset, cfg, setting)
    log.info(
        f"[Revision Full] split sizes: "
        f"train={len(train_graphs)}, valid={len(valid_graphs)}, "
        f"test={len(test_graphs)}, has_global_graph={dataset.global_graph is not None}"
    )
 
    # Level1
    l1_trainer = _build_level1_trainer(cfg)
    try:
        l1_fit_out = _call_level1_trainer_fit(
            trainer=l1_trainer,
            train_graphs=train_graphs,
            valid_graphs=valid_graphs,
            label_dict=dataset.labels,
            cfg=cfg,
        )
    except Exception as e:
        log.error(f"[Revision Full] Level1 fit failed: {e}", exc_info=True)
        raise
 
    # Level2 + Fusion
    l2_trainer = _build_level2_trainer(cfg, l1_trainer.model)
    l2_fit_out = _call_level2_trainer_fit(
        trainer=l2_trainer,
        l1_model=l1_trainer.model,
        cfg=cfg,
        train_ids=train_graphs,
        valid_ids=valid_graphs,
        labels=dataset.labels,
        global_graph=dataset.global_graph,
        loader_builder=_build_l2_dynamic_loader_builder(l1_trainer.model, cfg),
    )
 
    # Resource Tracking
    process = psutil.Process()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Calculate Max Nodes
    max_nodes = 0
    for g in test_graphs:
        n = g.num_nodes if hasattr(g, "num_nodes") else (g.graph.num_nodes if hasattr(g, "graph") else 0)
        if n > max_nodes: max_nodes = n

    metrics, yt, ys = l2_trainer.evaluate(
        test_graphs,
        label_dict=dataset.labels,
        global_graph=dataset.global_graph,      # ✅ Added missing global_graph
        loader_builder=_build_l2_dynamic_loader_builder(l1_trainer.model, cfg),
        return_preds=True
    )

    # Capture Peak Metrics
    peak_ram = process.memory_info().rss / (1024 * 1024)
    peak_gpu = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    from gog_fraud.evaluation.benchmark import evaluate_benchmark
    res = evaluate_benchmark(
        y_true=yt, 
        y_score=ys, 
        model_name="Revision-Full", 
        setting=setting,
        max_nodes_processed=max_nodes,
        peak_ram_mb=peak_ram,
        peak_gpu_mb=peak_gpu,
        elapsed_sec=time.perf_counter() - _t0,
    )
    if hasattr(table, "add"):
        table.add(res)
    return metrics


# ---------------------------------------------------------------------------
# save helper
# ---------------------------------------------------------------------------
def _best_effort_save_table(table, out_dir: Path, chain: str = "polygon") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for method_name in ["save", "dump", "write"]:
        if hasattr(table, method_name):
            fn = getattr(table, method_name)
            if callable(fn):
                try:
                    fn(out_dir)
                    log.info(f"[Benchmark] Saved results via table.{method_name}()")
                    return
                except Exception as exc:
                    log.warning(f"[Benchmark] table.{method_name}() failed: {exc}")

    rows = None
    for attr_name in ["results", "rows", "items"]:
        if hasattr(table, attr_name):
            obj = getattr(table, attr_name)

            if callable(obj):
                try:
                    obj = obj()
                except TypeError:
                    continue
                except Exception:
                    continue

            if isinstance(obj, (list, tuple)):
                rows = obj
                break

    if rows is None:
        log.warning("[Benchmark] Could not serialize BenchmarkTable; skipping save.")
        return

    serializable = []
    for row in rows:
        if hasattr(row, "__dict__"):
            serializable.append(dict(row.__dict__))
        else:
            serializable.append(str(row))

    out_path = out_dir / f"benchmark_results_{chain}.json"
    existing_data = []

    # [NEW] UPSERT LOGIC
    import json
    if out_path.exists():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception as e:
            log.warning(f"[Benchmark] Failed to load previous JSON for merging: {e}")

    for new_row in serializable:
        if not isinstance(new_row, dict):
            existing_data.append(new_row)
            continue
        
        m_name = new_row.get("model_name")
        m_set = new_row.get("setting")
        replaced = False
        
        for i, old_row in enumerate(existing_data):
            if isinstance(old_row, dict) and old_row.get("model_name") == m_name and old_row.get("setting") == m_set:
                existing_data[i] = new_row
                replaced = True
                break
        
        if not replaced:
            existing_data.append(new_row)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=2)

    log.info(f"[Benchmark] Saved merged JSON to {out_path}")



# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _build_dataset_from_cfg(cfg: dict) -> FraudDataset:
    candidates = []

    if isinstance(cfg, dict) and isinstance(cfg.get("dataset"), dict):
        candidates.append(("cfg['dataset']", cfg["dataset"]))

    candidates.append(("cfg", cfg))

    last_exc = None
    for name, cand in candidates:
        try:
            log.info(f"[Benchmark] Trying FraudDataset.from_config({name})")
            ds = FraudDataset.from_config(cand)

            try:
                train_n = len(ds.split_graphs("train"))
            except Exception:
                train_n = -1

            try:
                test_n = len(ds.split_graphs("test"))
            except Exception:
                test_n = -1

            log.info(
                f"[Benchmark] Dataset built from {name}: "
                f"train={train_n}, test={test_n}"
            )
            return ds
        except Exception as exc:
            last_exc = exc
            log.warning(f"[Benchmark] Failed building dataset from {name}: {exc}")

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Failed to build dataset from config")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--output", required=False, type=str, default=None)
    parser.add_argument("--smoke", required=False, type=bool, default=True)
    parser.add_argument("--stages", required=False, type=str, default="legacy,l1,l1_l2,full", 
                        help="Comma-separated stages to run: legacy, l1, l1_l2, full")
    args = parser.parse_args()

    active_stages = [s.strip().lower() for s in args.stages.split(",") if s.strip()]


    cfg = _load_config(args.config)
    setting = str(_cfg_get(cfg, "setting", "strict"))
    output_dir = Path(args.output or _cfg_get(cfg, "output_dir", "results/benchmark"))

    log.info(f"[Benchmark] Config: {args.config}")
    log.info(f"[Benchmark] Setting: {setting}")
    log.info(f"[Benchmark] Output: {output_dir}")





    dataset_cfg = _cfg_get(cfg, "dataset", {}) or {}
    dataset = _build_dataset_from_cfg(cfg)
    chain = dataset_cfg.get("chain", 'polygon')

    # =====================================================================
    # [추가된 코드] 데이터셋으로부터 실제 in_dim(피처 차원) 동적 추론 및 할당
    # =====================================================================
    in_dim = 32  # Fallback default
    if hasattr(dataset, "train_graphs") and len(dataset.train_graphs) > 0:
        first_item = dataset.train_graphs[0]
        # 데이터가 TransactionGraph 객체로 감싸져 있는지, 순수 PyG Data인지 확인
        data_obj = getattr(first_item, "graph", first_item) 
        if hasattr(data_obj, "x") and data_obj.x is not None:
            in_dim = data_obj.x.size(-1)  # 실제 차원 추출 (예: 3)
            log.info(f"[Benchmark] Inferred dynamic in_dim: {in_dim} from dataset")

    # 추론된 in_dim을 cfg(환경 설정)에 강제 덮어쓰기
    if "level1" not in cfg:
        cfg["level1"] = {}
    
    # Priority: Dataset Inferred > Config in_dim
    final_in_dim = in_dim if in_dim != 32 else cfg["level1"].get("in_dim", 32)
    cfg["level1"]["in_dim"] = final_in_dim
    
    log.info(f"[Benchmark] Final Level1 Input Dimension: {final_in_dim}")
    
    # Verify Level1 config
    from gog_fraud.models.level1.model import Level1ModelConfig
    l1_verify_cfg = Level1ModelConfig.from_config(cfg.get("level1", {}))
    log.info(f"[Benchmark] Resolved Level1 Config: {l1_verify_cfg}")
    # =====================================================================

    table = BenchmarkTable()

    # =====================================================================
    if "legacy" in active_stages and cfg.get("run_legacy", True):
        log.info("")
        log.info("=" * 50)
        log.info("(A) Running Legacy Baselines …")
        try:
            run_legacy_baselines(dataset, cfg, table, setting)
            _best_effort_save_table(table, output_dir, chain=chain)
        except Exception:
            log.exception("[Benchmark] Legacy baselines failed")

    # Clean up GPU state after Legacy stage to prevent CUDA context contamination
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            pass  # Ignore if CUDA is already in bad state

    # =====================================================================
    l1_cache_path = output_dir / "l1_model_weights.pt"
    l1_model = None

    if "l1" in active_stages and cfg.get("run_revision_l1", True):
        log.info("")
        log.info("=" * 50)
        log.info("(B) Running Revision Level1 …")
        try:
            # We explicitly need the trainer to get the trained model weight to cache
            from gog_fraud.pipelines.run_fraud_benchmark import _build_level1_trainer, _call_level1_trainer_fit, _get_split_graphs
            train_g, valid_g, test_g = _get_split_graphs(dataset, cfg, setting)
            trainer = _build_level1_trainer(cfg)
            _call_level1_trainer_fit(trainer, train_g, valid_g, dataset.labels, cfg)
            
            l1_model = trainer.model
            torch.save(l1_model.state_dict(), l1_cache_path)
            log.info(f"[Benchmark] Cached Level1 model state to {l1_cache_path}")

            # Inline evaluation metrics hook
            from gog_fraud.evaluation.benchmark import evaluate_benchmark
            backend = cfg.get("level1", {}).get("encoder_backend", "gnn").upper()
            m_name = f"Revision-L1-{backend}" if backend != "GNN" else "Revision-L1"
            
            # Resource Tracking
            process_l1 = psutil.Process()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            max_nodes_l1 = max(
                (g.graph.num_nodes if hasattr(g, "graph") else g.num_nodes)
                for g in test_g
            ) if test_g else 0

            metrics, yt, ys = trainer.evaluate(test_g, label_dict=dataset.labels, return_preds=True)

            peak_ram_l1 = process_l1.memory_info().rss / (1024 * 1024)
            peak_gpu_l1 = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

            res = evaluate_benchmark(
                y_true=yt, y_score=ys, model_name=m_name, setting=setting,
                max_nodes_processed=max_nodes_l1,
                peak_ram_mb=peak_ram_l1,
                peak_gpu_mb=peak_gpu_l1,
            )
            table.add(res)
            _best_effort_save_table(table, output_dir, chain=chain)

        except Exception:
            log.exception("[Benchmark] Revision L1 failed")
    
    # Load cached L1 model if we skipped L1 but need it for L2/Full
    if ("l1_l2" in active_stages or "full" in active_stages) and l1_model is None:
        log.info(f"[Benchmark] Loading cached L1 model from {l1_cache_path} for L2 dependencies")
        try:
            from gog_fraud.pipelines.run_fraud_benchmark import _build_level1_trainer
            dummy_trainer = _build_level1_trainer(cfg)
            dummy_trainer.model.load_state_dict(torch.load(l1_cache_path, map_location="cpu"))
            l1_model = dummy_trainer.model
        except Exception as e:
            log.error(f"[Benchmark] Failed to load L1 cache! Level 2 pipeline will fail. Run --stages l1 first. {e}")

    # =====================================================================
    if "l1_l2" in active_stages and cfg.get("run_revision_l1_l2", True):
        log.info("")
        log.info("=" * 50)
        log.info("(C) Running Revision Level1 + Level2 …")

        if l1_model is None:
            log.warning("[Benchmark] Skipping Revision L1+L2 because Level1 model is missing (L1 stage failed or cache load failed).")
        else:
            try:
                from gog_fraud.pipelines.run_fraud_benchmark import _get_split_graphs, _build_level2_trainer, _call_level2_trainer_fit, _build_l2_dynamic_loader_builder, evaluate_benchmark
                train_g, valid_g, test_g = _get_split_graphs(dataset, cfg, setting)
                l2_trainer = _build_level2_trainer(cfg, l1_model)

                _call_level2_trainer_fit(
                    trainer=l2_trainer,
                    l1_model=l1_model,
                    cfg=cfg,
                    train_ids=train_g,
                    valid_ids=valid_g,
                    labels=dataset.labels,
                    global_graph=dataset.global_graph,
                    loader_builder=_build_l2_dynamic_loader_builder(l1_model, cfg)
                )

                # Resource Tracking
                process_l1l2 = psutil.Process()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                max_nodes_l1l2 = max(
                    (g.num_nodes if hasattr(g, "num_nodes") else g.graph.num_nodes)
                    for g in test_g
                ) if test_g else 0

                metrics, yt, ys = l2_trainer.evaluate(
                    test_g,
                    label_dict=dataset.labels,
                    global_graph=dataset.global_graph,
                    loader_builder=_build_l2_dynamic_loader_builder(l1_model, cfg),
                    return_preds=True
                )
                backend = cfg.get("level1", {}).get("encoder_backend", "gnn").upper()
                m_name = f"Revision-L1+L2-{backend}" if backend != "GNN" else "Revision-L1+L2"

                peak_ram_l1l2 = process_l1l2.memory_info().rss / (1024 * 1024)
                peak_gpu_l1l2 = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

                res = evaluate_benchmark(
                    y_true=yt, y_score=ys, model_name=m_name, setting=setting,
                    max_nodes_processed=max_nodes_l1l2,
                    peak_ram_mb=peak_ram_l1l2,
                    peak_gpu_mb=peak_gpu_l1l2,
                )
                table.add(res)
                _best_effort_save_table(table, output_dir, chain=chain)

            except Exception:
                log.exception("[Benchmark] Revision L1+L2 failed")

    # =====================================================================
    if "full" in active_stages and cfg.get("run_revision_full", True):
        log.info("")
        log.info("=" * 50)
        log.info("(D) Running Revision Full …")

        if l1_model is None:
            log.warning("[Benchmark] Skipping Revision Full because Level1 model is missing.")
        else:
            try:
                from gog_fraud.pipelines.run_fraud_benchmark import _get_split_graphs, _build_level2_trainer, _call_level2_trainer_fit, _build_l2_dynamic_loader_builder, evaluate_benchmark
                train_g, valid_g, test_g = _get_split_graphs(dataset, cfg, setting)
                l2_trainer = _build_level2_trainer(cfg, l1_model)

                _call_level2_trainer_fit(
                    trainer=l2_trainer,
                    l1_model=l1_model,
                    cfg=cfg,
                    train_ids=train_g,
                    valid_ids=valid_g,
                    labels=dataset.labels,
                    global_graph=dataset.global_graph,
                    loader_builder=_build_l2_dynamic_loader_builder(l1_model, cfg)
                )

                # Resource Tracking
                process_full = psutil.Process()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                max_nodes_full = max(
                    (g.num_nodes if hasattr(g, "num_nodes") else g.graph.num_nodes)
                    for g in test_g
                ) if test_g else 0

                metrics, yt, ys = l2_trainer.evaluate(
                    test_g,
                    label_dict=dataset.labels,
                    global_graph=dataset.global_graph,
                    loader_builder=_build_l2_dynamic_loader_builder(l1_model, cfg),
                    return_preds=True
                )
                backend = cfg.get("level1", {}).get("encoder_backend", "gnn").upper()
                m_name = f"Revision-Full-{backend}" if backend != "GNN" else "Revision-Full"

                peak_ram_full = process_full.memory_info().rss / (1024 * 1024)
                peak_gpu_full = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

                res = evaluate_benchmark(
                    y_true=yt, y_score=ys, model_name=m_name, setting=setting,
                    max_nodes_processed=max_nodes_full,
                    peak_ram_mb=peak_ram_full,
                    peak_gpu_mb=peak_gpu_full,
                )
                table.add(res)
                _best_effort_save_table(table, output_dir, chain=chain)

            except Exception:
                log.exception("[Benchmark] Revision Full failed")

    # One final implicit serialization catch-all
    # _best_effort_save_table(table, output_dir, chain=chain)
    ## Note: We rely on intermediate saves after each stage, so we won't do a final save here to avoid overwriting with potentially incomplete results.


if __name__ == "__main__":
    main()
