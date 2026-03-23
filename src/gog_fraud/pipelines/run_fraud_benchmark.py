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

from gog_fraud.adapters.legacy_adapter import LegacyAdapterConfig, LegacyBatchRunner
from gog_fraud.data.io.dataset import FraudDataset
from gog_fraud.evaluation.benchmark import BenchmarkTable, evaluate_benchmark
from gog_fraud.models.level1.model import Level1Model
from gog_fraud.models.level2.model import Level2Model
from gog_fraud.training.loops.level1 import Level1Trainer
from gog_fraud.training.loops.level2 import Level2Trainer

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




def _call_level1_trainer_fit(
    trainer,
    *,
    train_graphs=None,
    valid_graphs=None,
    labels=None,
    train_loader=None,
    valid_loader=None,
    loader_builder=None,
    **kwargs,
):
    if _is_none_or_empty(train_loader) and _is_none_or_empty(train_graphs):
        log.warning("[Revision L1] Empty training split. trainer.fit() skipped.")
        return {
            "history": [],
            "best_score": None,
            "best_metric": None,
            "best_mode": None,
            "epochs_ran": 0,
            "skipped": True,
            "reason": "empty_train_graphs",
        }

    if valid_loader is not None and _is_empty(valid_loader):
        valid_loader = None
    if valid_graphs is not None and _is_empty(valid_graphs):
        valid_graphs = None

    return trainer.fit(
        train_graphs=train_graphs,
        valid_graphs=valid_graphs,
        label_dict=labels,
        train_loader=train_loader,
        valid_loader=valid_loader,
        loader_builder=loader_builder,
        **kwargs,
    )



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


def _cfg_get(cfg: Any, key: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# dataset split helpers
# ---------------------------------------------------------------------------
def _get_split_graphs(dataset: FraudDataset, *names: str):
    last_exc = None
    for name in names:
        try:
            graphs = dataset.split_graphs(name)
            if graphs is not None:
                return graphs
        except Exception as exc:
            last_exc = exc
    if last_exc:
        raise last_exc
    return []


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

    return score.reshape(-1).detach().cpu()


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
    model_names = _cfg_get(
        legacy_cfg,
        "models",
        ["DOMINANT", "DONE", "GAE", "AnomalyDAE", "CoLA"],
    )

    base_adapter_cfg = LegacyAdapterConfig(
        agg_method=_cfg_get(legacy_cfg, "agg_method", "max"),
        topk=int(_cfg_get(legacy_cfg, "topk", 3)),
        normalize_score=bool(_cfg_get(legacy_cfg, "normalize_score", True)),
        gpu=int(_cfg_get(legacy_cfg, "gpu", -1)),
        hid_dim=int(_cfg_get(legacy_cfg, "hid_dim", 64)),
        num_layers=int(_cfg_get(legacy_cfg, "num_layers", 2)),
        epoch=int(_cfg_get(legacy_cfg, "epoch", 100)),
        lr=float(_cfg_get(legacy_cfg, "lr", 0.003)),
    )

    train_graphs = _maybe_limit_graphs(_get_split_graphs(dataset, "train"), cfg, "train")
    _ = _maybe_limit_graphs(_get_split_graphs(dataset, "valid", "val"), cfg, "val")
    test_graphs = _maybe_limit_graphs(_get_split_graphs(dataset, "test"), cfg, "test")

    if not test_graphs:
        log.warning("[Legacy] No test graphs found!")
        return

    batch = LegacyBatchRunner(
        model_names=model_names,
        base_cfg=base_adapter_cfg,
    )
    all_scores = batch.run_all(test_graphs)

    for model_name, score_dict in all_scores.items():
        contract_ids = [g.contract_id for g in test_graphs]
        filtered = [
            (float(score_dict.get(cid, 0.0)), int(dataset.labels[cid]))
            for cid in contract_ids
            if cid in dataset.labels
        ]
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
        )
        table.add(result)
        log.info(str(result))


# ---------------------------------------------------------------------------
# (B) revision l1
# ---------------------------------------------------------------------------
def run_revision_l1(
    dataset: FraudDataset,
    cfg: dict,
    table: BenchmarkTable,
    setting: str,
) -> None:

    def _prepare_l1_cfg(raw: dict) -> dict:
        raw = dict(raw or {})
        raw.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
        raw.setdefault("use_amp", False)
        raw.setdefault("pos_weight", None)
        raw.setdefault("epochs", raw.get("max_epochs", 10))
        raw.setdefault("log_every", 10)
        raw.setdefault("grad_clip", 0.0)
        raw.setdefault("num_workers", 0)
        raw.setdefault("weight_decay", raw.get("weight_decay", 0.0))
        return raw

    l1_cfg_raw = _prepare_l1_cfg(_cfg_to_dict(_cfg_get(cfg, "level1", {}) or {}))
    l1_cfg = _cfg_to_attrdict(l1_cfg_raw)

    log.info("[Revision L1] Training Level1 model …")
 
    train_graphs, valid_graphs, test_graphs, labels, _ = _normalize_graph_splits(dataset)
 
    train_n = _safe_len(train_graphs) or 0
    valid_n = _safe_len(valid_graphs) or 0 if valid_graphs is not None else 0
    test_n = _safe_len(test_graphs) or 0
 
    log.info(
        "[Benchmark] Normalized split sizes: train=%d, valid=%d, test=%d",
        train_n, valid_n, test_n)
    
    if train_n == 0 and test_n == 0:
        log.warning(
            "[Benchmark] Dataset splits are empty. Training/evaluation stages will be skipped. "
            "Check configs/benchmark/strict_smoke.yaml dataset path/split settings."
        )

    log.info(
        "[Revision L1] split sizes: train=%d, valid=%d, test=%d",
        train_n, valid_n, test_n
    )
 
    if train_n == 0:
        _record_skip(table, "revision_level1", "empty_train_graphs", setting)
        return
 
    if test_n == 0:
        log.warning("[Revision L1] No test graphs found. Training may run, but benchmark evaluation will be skipped.")
 

    model = Level1Model.from_config(l1_cfg_raw)
    optimizer = _build_optimizer(model, l1_cfg_raw)
    trainer = Level1Trainer(model=model, optimizer=optimizer, cfg=l1_cfg)

    fit_out = _call_level1_trainer_fit(
        trainer,
        train_graphs=train_graphs,
        valid_graphs=valid_graphs,
        labels=labels,
    )
 
    if fit_out.get("skipped"):
        _record_skip(table, "revision_level1", fit_out.get("reason", "unknown"), setting)
        return
 
    if test_n == 0:
        _record_skip(table, "revision_level1", "empty_test_graphs_after_training", setting)
        return

    scores: Dict[str, float] = {}
    for tg in test_graphs:
        try:
            out = model.predict(tg.graph)
            scores[tg.contract_id] = _extract_scalar_score(out)
        except Exception as exc:
            log.warning(f"[L1] Skip {tg.contract_id}: {exc}")
            scores[tg.contract_id] = 0.0

    _append_result(scores, dataset, test_graphs, "Revision_L1", setting, cfg, table)


# ---------------------------------------------------------------------------
# (C) revision l1+l2
# ---------------------------------------------------------------------------
def run_revision_l1_l2(
    dataset: FraudDataset,
    cfg: dict,
    table: BenchmarkTable,
    setting: str,
) -> None:

    if not hasattr(dataset, "global_graph") or dataset.global_graph is None:
        log.warning("[Revision L1+L2] dataset.global_graph is None. Skipping.")
        return

    log.info("[Revision L1+L2] Training Level1 + Level2 …")

    l1_cfg_raw = _cfg_to_dict(_cfg_get(cfg, "level1", {}) or {})
    l2_cfg_raw = _cfg_to_dict(_cfg_get(cfg, "level2", {}) or {})
    l1_cfg = _cfg_to_attrdict(l1_cfg_raw)
    l2_cfg = _cfg_to_attrdict(l2_cfg_raw)

    train_graphs, valid_graphs, test_graphs, labels, global_graph = _normalize_graph_splits(dataset)
 
    train_n = _safe_len(train_graphs) or 0
    test_n = _safe_len(test_graphs) or 0
 
    if global_graph is None:
        _record_skip(table, "revision_l1_l2", "global_graph_is_none", setting)
        return
 
    if train_n == 0:
        _record_skip(table, "revision_l1_l2", "empty_train_graphs", setting)
        return
 
    if test_n == 0:
        _record_skip(table, "revision_l1_l2", "empty_test_graphs", setting)
        return

    l1_model = Level1Model.from_config(l1_cfg_raw)
    l1_optimizer = _build_optimizer(l1_model, l1_cfg_raw)
    l1_trainer = Level1Trainer(model=l1_model, optimizer=l1_optimizer, cfg=l1_cfg)
    try:
        # l1_trainer.fit(train_graphs=train_graphs, valid_graphs=valid_graphs, label_dict=dataset.labels)
        _call_level1_trainer_fit(
            l1_trainer,
            train_graphs=train_graphs,
            valid_graphs=valid_graphs,
            label_dict=dataset.labels,
        )
    except TypeError:
        l1_trainer.fit(train_graphs, valid_graphs, dataset.labels)

    l2_model = Level2Model.from_config(l2_cfg_raw)
    l2_optimizer = _build_optimizer(l2_model, l2_cfg_raw)
    l2_trainer = Level2Trainer(model=l2_model, optimizer=l2_optimizer, cfg=l2_cfg)

    train_ids = _get_split_ids(dataset, "train")
    valid_ids = _get_split_ids(dataset, "valid", "val")

    try:
        # l2_trainer.fit(
        #     l1_model=l1_model,
        #     global_graph=dataset.global_graph,
        #     train_ids=train_ids,
        #     valid_ids=valid_ids,
        #     label_dict=dataset.labels,
        # )
        _call_level2_trainer_fit(
            l2_trainer,
            l1_model=l1_model,
            global_graph=dataset.global_graph,
            train_ids=train_ids,
            valid_ids=valid_ids,
            label_dict=dataset.labels,
        )

    except TypeError:
        # l2_trainer.fit(l1_model, dataset.global_graph, train_ids, valid_ids, dataset.labels)
        _call_level2_trainer_fit(
            l2_trainer,
            l1_model,
            global_graph=dataset.global_graph,
            train_ids=train_ids,
            valid_ids=valid_ids,
            label_dict=dataset.labels,
        )
        
    scores: Dict[str, float] = {}
    test_ids = [tg.contract_id for tg in test_graphs]
    try:
        l2_out = l2_model.predict(
            l1_model=l1_model,
            global_graph=dataset.global_graph,
            contract_ids=test_ids,
        )
        score_vec = _extract_score_tensor(l2_out).tolist()
        for cid, score in zip(test_ids, score_vec):
            scores[cid] = float(score)
    except Exception as exc:
        log.warning(f"[L1+L2] Inference error: {exc}. Falling back to L1.")
        for tg in test_graphs:
            try:
                out = l1_model.predict(tg.graph)
                scores[tg.contract_id] = _extract_scalar_score(out)
            except Exception as sub_exc:
                log.warning(f"[L1+L2:FALLBACK] Skip {tg.contract_id}: {sub_exc}")
                scores[tg.contract_id] = 0.0

    _append_result(scores, dataset, test_graphs, "Revision_L1+L2", setting, cfg, table)


# ---------------------------------------------------------------------------
# (D) revision full
# ---------------------------------------------------------------------------
def run_revision_full(
    dataset: FraudDataset,
    cfg: dict,
    table: BenchmarkTable,
    setting: str,
) -> None:
    from gog_fraud.models.level1.model import Level1Model
    from gog_fraud.models.level2.model import Level2Model
    from gog_fraud.training.loops.level1 import Level1Trainer
    from gog_fraud.training.loops.level2 import Level2Trainer

    log.info("[Revision Full] Training Level1 + Level2 + Fusion …")

    l1_cfg_raw = _cfg_to_dict(_cfg_get(cfg, "level1", {}) or {})
    l2_cfg_raw = _cfg_to_dict(_cfg_get(cfg, "level2", {}) or {})
    fusion_cfg = _cfg_to_dict(_cfg_get(cfg, "fusion", {}) or {})

    l1_cfg = _cfg_to_attrdict(l1_cfg_raw)
    l2_cfg = _cfg_to_attrdict(l2_cfg_raw)

    train_graphs, valid_graphs, test_graphs, labels, global_graph = _normalize_graph_splits(dataset)
 
    train_n = _safe_len(train_graphs) or 0
    valid_n = _safe_len(valid_graphs) or 0 if valid_graphs is not None else 0
    test_n = _safe_len(test_graphs) or 0
 
    log.info(
        "[Revision Full] split sizes: train=%d, valid=%d, test=%d, has_global_graph=%s",
        train_n, valid_n, test_n, global_graph is not None
    )
 
    if train_n == 0:
        _record_skip(table, "revision_full", "empty_train_graphs", setting)
        return
 
    if global_graph is None:
        _record_skip(table, "revision_full", "global_graph_is_none", setting)
        return
 
    if test_n == 0:
        log.warning("[Revision Full] No test graphs found. Training may run, but final benchmark evaluation will be skipped.")
 
    l1_model = Level1Model.from_config(l1_cfg_raw)
    l1_optimizer = _build_optimizer(l1_model, l1_cfg_raw)
    l1_trainer = Level1Trainer(model=l1_model, optimizer=l1_optimizer, cfg=l1_cfg)

    l1_fit_out = _call_level1_trainer_fit(
        l1_trainer,
        train_graphs=train_graphs,
        valid_graphs=valid_graphs,
        labels=labels,
    )
 
    if l1_fit_out.get("skipped"):
        _record_skip(table, "revision_full", l1_fit_out.get("reason", "l1_skipped"), setting)
        return

    l1_scores: Dict[str, float] = {}
    for tg in test_graphs:
        try:
            l1_scores[tg.contract_id] = _extract_scalar_score(l1_model.predict(tg.graph))
        except Exception as exc:
            log.warning(f"[Full:L1] Skip {tg.contract_id}: {exc}")
            l1_scores[tg.contract_id] = 0.0

    ## Level 2 training and inference
    l2_scores: Dict[str, float] = {}
    if hasattr(dataset, "global_graph") and dataset.global_graph is not None:
        try:
            l2_model = Level2Model.from_config(l2_cfg_raw)
            l2_optimizer = _build_optimizer(l2_model, l2_cfg_raw)
            l2_trainer = Level2Trainer(model=l2_model, optimizer=l2_optimizer, cfg=l2_cfg)

            train_ids = _get_split_ids(dataset, "train")
            valid_ids = _get_split_ids(dataset, "valid", "val")
            test_ids = [tg.contract_id for tg in test_graphs]

            if _is_none_or_empty(train_ids):
                _record_skip(table, "revision_full", "empty_train_ids_for_level2", setting)
                return
    

            l2_fit_out = _call_level2_trainer_fit(
                l2_trainer,
                l1_model=l1_model,
                global_graph=global_graph,
                train_ids=train_ids,
                valid_ids=valid_ids,
                labels=labels,
            )
        
            if l2_fit_out.get("skipped"):
                _record_skip(table, "revision_full", l2_fit_out.get("reason", "l2_skipped"), setting)
                return
        
            if test_n == 0 or _is_none_or_empty(test_ids):
                _record_skip(table, "revision_full", "empty_test_split_for_final_eval", setting)
                return

            l2_out = l2_model.predict(
                l1_model=l1_model,
                global_graph=dataset.global_graph,
                contract_ids=test_ids,
            )
            score_vec = _extract_score_tensor(l2_out).tolist()
            for cid, score in zip(test_ids, score_vec):
                l2_scores[cid] = float(score)
        except Exception as exc:
            log.warning(f"[Full:L2] Failed. Falling back to L1-only fusion. Error: {exc}")

    alpha = float(fusion_cfg.get("alpha", 0.5))
    alpha = max(0.0, min(1.0, alpha))

    fused_scores: Dict[str, float] = {}
    for tg in test_graphs:
        cid = tg.contract_id
        s1 = float(l1_scores.get(cid, 0.0))
        s2 = float(l2_scores.get(cid, s1))
        fused_scores[cid] = alpha * s1 + (1.0 - alpha) * s2

    _append_result(fused_scores, dataset, test_graphs, "Revision_Full", setting, cfg, table)


# ---------------------------------------------------------------------------
# save helper
# ---------------------------------------------------------------------------
def _best_effort_save_table(table, out_dir: Path) -> None:
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

    out_path = out_dir / "benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        import json
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    log.info(f"[Benchmark] Saved fallback JSON to {out_path}")



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
    args = parser.parse_args()

    cfg = _load_config(args.config)
    setting = str(_cfg_get(cfg, "setting", "strict"))
    output_dir = Path(args.output or _cfg_get(cfg, "output", "results/benchmark"))

    log.info(f"[Benchmark] Config: {args.config}")
    log.info(f"[Benchmark] Setting: {setting}")
    log.info(f"[Benchmark] Output: {output_dir}")

    dataset_cfg = _cfg_get(cfg, "dataset", {}) or {}
    # dataset = FraudDataset.from_config(dataset_cfg)
    dataset = _build_dataset_from_cfg(cfg)

    table = BenchmarkTable()

    log.info("")
    log.info("=" * 50)
    log.info("(A) Running Legacy Baselines …")
    try:
        run_legacy_baselines(dataset, cfg, table, setting)
    except Exception:
        log.exception("[Benchmark] Legacy baselines failed")

    log.info("")
    log.info("=" * 50)
    log.info("(B) Running Revision Level1 …")
    try:
        run_revision_l1(dataset, cfg, table, setting)
    except Exception:
        log.exception("[Benchmark] Revision Level1 failed")

    log.info("")
    log.info("=" * 50)
    log.info("(C) Running Revision Level1 + Level2 …")
    try:
        run_revision_l1_l2(dataset, cfg, table, setting)
    except Exception:
        log.exception("[Benchmark] Revision L1+L2 failed")

    log.info("")
    log.info("=" * 50)
    log.info("(D) Running Revision Full …")
    try:
        run_revision_full(dataset, cfg, table, setting)
    except Exception:
        log.exception("[Benchmark] Revision Full failed")

    _best_effort_save_table(table, output_dir)


if __name__ == "__main__":
    main()
