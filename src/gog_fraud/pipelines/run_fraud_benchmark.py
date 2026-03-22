# src/gog_fraud/pipelines/run_fraud_benchmark.py
"""
Legacy vs Revision 4-way benchmark pipeline.

실험군
  (A) Legacy baseline : DOMINANT / DONE / GAE / AnomalyDAE / CoLA
  (B) Revision L1     : Level1 only
  (C) Revision L1+L2  : Level1 + Level2
  (D) Revision Full   : Level1 + Level2 + Fusion

실행 예시
---------
# strict 비교
python -m gog_fraud.pipelines.run_fraud_benchmark \
    --config configs/benchmark/strict.yaml

# full-system 비교
python -m gog_fraud.pipelines.run_fraud_benchmark \
    --config configs/benchmark/full_system.yaml \
    --output results/full_system/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml

# 내부 모듈
from gog_fraud.adapters.legacy_adapter import LegacyAdapterConfig, LegacyBatchRunner
from gog_fraud.data.io.dataset import DatasetConfig, FraudDataset
from gog_fraud.evaluation.benchmark import BenchmarkResult, BenchmarkTable, evaluate_benchmark

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


## ---------------------------------------------------------------------------
## Graph 수 제한 (디버그용)
## ---------------------------------------------------------------------------
def _maybe_limit_graphs(graphs, cfg, split_name: str):
    debug_cfg = cfg.get("debug", {}) or {}
    if not debug_cfg.get("enabled", False):
        return graphs
 
    limit = debug_cfg.get(f"max_{split_name}_graphs")
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
# 설정 구조
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 각 실험군 실행 함수
# ---------------------------------------------------------------------------

def run_legacy_baselines(
    dataset: FraudDataset,
    cfg: dict,
    table: BenchmarkTable,
    setting: str,
) -> None:
    """(A) Legacy DOMINANT/DONE/GAE/AnomalyDAE/CoLA."""
    legacy_cfg = cfg.get("legacy", {})
    model_names = legacy_cfg.get("models", ["DOMINANT", "DONE", "GAE", "AnomalyDAE", "CoLA"])

    base_adapter_cfg = LegacyAdapterConfig(
        agg_method      = legacy_cfg.get("agg_method", "max"),
        topk            = legacy_cfg.get("topk", 3),
        normalize_score = legacy_cfg.get("normalize_score", True),
        gpu             = legacy_cfg.get("gpu", -1),
        hid_dim         = legacy_cfg.get("hid_dim", 64),
        num_layers      = legacy_cfg.get("num_layers", 2),
        epoch           = legacy_cfg.get("epoch", 100),
        lr              = legacy_cfg.get("lr", 0.003),
    )

    train_graphs = dataset.split_graphs("train")
    val_graphs = dataset.split_graphs("val")
    test_graphs = dataset.split_graphs("test")

    train_graphs = _maybe_limit_graphs(train_graphs, cfg, "train")
    val_graphs = _maybe_limit_graphs(val_graphs, cfg, "val")
    test_graphs = _maybe_limit_graphs(test_graphs, cfg, "test")

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
        y_score = [score_dict.get(cid, 0.0) for cid in contract_ids]
        y_true, valid_ids = dataset.get_labels_tensor(contract_ids)

        # valid_ids 기준으로 y_score 재정렬
        valid_set = set(valid_ids)
        filtered = [
            (score_dict.get(cid, 0.0), int(dataset.labels[cid]))
            for cid in contract_ids
            if cid in valid_set
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
            threshold=cfg.get("threshold", 0.5),
            k_list=cfg.get("k_list", [10, 20, 50]),
            bootstrap=cfg.get("bootstrap", True),
        )
        table.add(result)
        log.info(str(result))


def run_revision_l1(
    dataset: FraudDataset,
    cfg: dict,
    table: BenchmarkTable,
    setting: str,
) -> None:
    """(B) Revision Level1 only."""
    from gog_fraud.models.level1.model import Level1Model
    from gog_fraud.training.loops.level1 import Level1Trainer

    l1_cfg = cfg.get("level1", {})
    log.info("[Revision L1] Training Level1 model …")

    # 학습
    train_graphs = dataset.split_graphs("train")
    valid_graphs = dataset.split_graphs("valid")
    test_graphs  = dataset.split_graphs("test")

    model = Level1Model.from_config(l1_cfg)
    trainer = Level1Trainer(model=model, cfg=l1_cfg)
    trainer.fit(
        train_graphs=train_graphs,
        valid_graphs=valid_graphs,
        label_dict=dataset.labels,
    )

    # 추론
    scores: Dict[str, float] = {}
    for tg in test_graphs:
        try:
            out = model.predict(tg.graph)
            scores[tg.contract_id] = float(out.score.item())
        except Exception as exc:
            log.warning(f"[L1] Skip {tg.contract_id}: {exc}")
            scores[tg.contract_id] = 0.0

    _append_result(scores, dataset, test_graphs, "Revision_L1", setting, cfg, table)


def run_revision_l1_l2(
    dataset: FraudDataset,
    cfg: dict,
    table: BenchmarkTable,
    setting: str,
) -> None:
    """(C) Revision Level1 + Level2."""
    from gog_fraud.models.level1.model import Level1Model
    from gog_fraud.models.level2.model import Level2Model
    from gog_fraud.training.loops.level1 import Level1Trainer
    from gog_fraud.training.loops.level2 import Level2Trainer

    log.info("[Revision L1+L2] Training Level1 + Level2 …")

    l1_cfg  = cfg.get("level1", {})
    l2_cfg  = cfg.get("level2", {})

    train_graphs = dataset.split_graphs("train")
    valid_graphs = dataset.split_graphs("valid")
    test_graphs  = dataset.split_graphs("test")

    # Level1 학습 + embedding 추출
    l1_model = Level1Model.from_config(l1_cfg)
    l1_trainer = Level1Trainer(model=l1_model, cfg=l1_cfg)
    l1_trainer.fit(train_graphs, valid_graphs, dataset.labels)

    # Level2 학습
    l2_model = Level2Model.from_config(l2_cfg)
    l2_trainer = Level2Trainer(model=l2_model, cfg=l2_cfg)
    l2_trainer.fit(
        l1_model=l1_model,
        global_graph=dataset.global_graph,
        train_ids=dataset.split_ids("train"),
        valid_ids=dataset.split_ids("valid"),
        label_dict=dataset.labels,
    )

    # 추론: Level2 score 사용
    scores: Dict[str, float] = {}
    test_ids = [tg.contract_id for tg in test_graphs]
    try:
        l2_out = l2_model.predict(
            l1_model=l1_model,
            global_graph=dataset.global_graph,
            contract_ids=test_ids,
        )
        for cid, score in zip(test_ids, l2_out.score.tolist()):
            scores[cid] = float(score)
    except Exception as exc:
        log.warning(f"[L1+L2] Inference error: {exc}. Falling back to L1.")
        for tg in test_graphs:
            out = l1_model.predict(tg.graph)
            scores[tg.contract_id] = float(out.score.item())

    _append_result(scores, dataset, test_graphs, "Revision_L1+L2", setting, cfg, table)


def run_revision_full(
    dataset: FraudDataset,
    cfg: dict,
    table: BenchmarkTable,
    setting: str,
) -> None:
    """(D) Revision Full: Level1 + Level2 + Fusion."""
    from gog_fraud.models.level1.model import Level1Model
    from gog_fraud.models.level2.model import Level2Model
    from gog_fraud.pipelines.fusion import (
        FusionInput,
        LearnedFusion,
        LearnedFusionConfig,
        FusionTrainer,
        FusionTrainerConfig,
        build_fusion,
    )
    from gog_fraud.training.loops.level1 import Level1Trainer
    from gog_fraud.training.loops.level2 import Level2Trainer

    log.info("[Revision Full] Level1 + Level2 + Fusion …")

    l1_cfg   = cfg.get("level1", {})
    l2_cfg   = cfg.get("level2", {})
    fus_cfg  = cfg.get("fusion", {})

    train_graphs = dataset.split_graphs("train")
    valid_graphs = dataset.split_graphs("valid")
    test_graphs  = dataset.split_graphs("test")

    # Level1
    l1_model = Level1Model.from_config(l1_cfg)
    l1_trainer = Level1Trainer(model=l1_model, cfg=l1_cfg)
    l1_trainer.fit(train_graphs, valid_graphs, dataset.labels)

    # Level2
    l2_model = Level2Model.from_config(l2_cfg)
    l2_trainer = Level2Trainer(model=l2_model, cfg=l2_cfg)
    l2_trainer.fit(
        l1_model=l1_model,
        global_graph=dataset.global_graph,
        train_ids=dataset.split_ids("train"),
        valid_ids=dataset.split_ids("valid"),
        label_dict=dataset.labels,
    )

    # Fusion
    fusion_strategy = fus_cfg.get("strategy", "learned")
    if fusion_strategy == "learned":
        fusion = LearnedFusion(
            LearnedFusionConfig(
                hidden_dim=fus_cfg.get("hidden_dim", 32),
                num_layers=fus_cfg.get("num_layers", 2),
            )
        )
        fus_trainer = FusionTrainer(
            fusion=fusion,
            cfg=FusionTrainerConfig(
                epochs=fus_cfg.get("epochs", 50),
                batch_size=fus_cfg.get("batch_size", 64),
                val_metric=fus_cfg.get("val_metric", "pr_auc"),
            ),
        )
        train_fi = _build_fusion_input(l1_model, l2_model, dataset, "train")
        valid_fi = _build_fusion_input(l1_model, l2_model, dataset, "valid")
        fus_trainer.fit(train_fi, valid_fi, verbose=False)
    else:
        fusion = build_fusion(
            fusion_strategy,
            level1_weight=fus_cfg.get("level1_weight", 0.4),
            level2_weight=fus_cfg.get("level2_weight", 0.6),
        )

    # 추론
    test_fi = _build_fusion_input(l1_model, l2_model, dataset, "test")
    fus_out = fusion.fuse(test_fi)

    test_ids = dataset.split_ids("test")
    # test_fi에 포함된 순서와 test_ids 정렬이 같다고 가정
    scores: Dict[str, float] = {}
    for cid, score in zip(test_ids, fus_out.score.tolist()):
        scores[cid] = float(score)

    _append_result(scores, dataset, test_graphs, "Revision_Full", setting, cfg, table)


# ---------------------------------------------------------------------------
# 공통 헬퍼
# ---------------------------------------------------------------------------

def _build_fusion_input(l1_model, l2_model, dataset, split: str):
    """Level1 + Level2 score를 묶어 FusionInput 생성."""
    from gog_fraud.pipelines.fusion import FusionInput

    ids    = dataset.split_ids(split)
    graphs = dataset.split_graphs(split)

    l1_scores, l1_logits = [], []
    for tg in graphs:
        try:
            out = l1_model.predict(tg.graph)
            l1_scores.append(float(out.score.item()))
            l1_logits.append(float(out.logits.item()) if hasattr(out, "logits") else 0.0)
        except Exception:
            l1_scores.append(0.5)
            l1_logits.append(0.0)

    try:
        l2_out    = l2_model.predict(l2_model, dataset.global_graph, ids)
        l2_scores = l2_out.score.tolist()
        l2_logits = l2_out.logits.tolist() if hasattr(l2_out, "logits") else [0.0] * len(ids)
    except Exception:
        l2_scores = [0.5] * len(ids)
        l2_logits = [0.0] * len(ids)

    labels_t, _ = dataset.get_labels_tensor(ids)

    import torch
    return FusionInput(
        level1_score  = torch.tensor(l1_scores, dtype=torch.float),
        level2_score  = torch.tensor(l2_scores, dtype=torch.float),
        level1_logits = torch.tensor(l1_logits, dtype=torch.float),
        level2_logits = torch.tensor(l2_logits, dtype=torch.float),
        label         = labels_t,
    )


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
    valid_set    = set(dataset.labels.keys())

    ys_list = [scores.get(cid, 0.0) for cid in contract_ids if cid in valid_set]
    yt_list = [dataset.labels[cid]  for cid in contract_ids if cid in valid_set]

    if not ys_list:
        log.warning(f"[{model_name}] No valid predictions. Skipping.")
        return

    result = evaluate_benchmark(
        y_true     = yt_list,
        y_score    = ys_list,
        model_name = model_name,
        setting    = setting,
        threshold  = cfg.get("threshold", 0.5),
        k_list     = cfg.get("k_list", [10, 20, 50]),
        bootstrap  = cfg.get("bootstrap", True),
    )
    table.add(result)
    log.info(str(result))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud Detection Benchmark")
    parser.add_argument("--config",  type=str, required=True,
                        help="Path to benchmark YAML config")
    parser.add_argument("--output",  type=str, default="results/benchmark",
                        help="Output directory for results")
    parser.add_argument("--setting", type=str, default=None,
                        help="Override setting name in output")
    args = parser.parse_args()

    # 설정 로드
    cfg = _load_config(args.config)
    setting = args.setting or cfg.get("setting", "benchmark")
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"[Benchmark] Config: {args.config}")
    log.info(f"[Benchmark] Setting: {setting}")
    log.info(f"[Benchmark] Output: {out_dir}")

    # 데이터 로드
    ds_cfg = DatasetConfig(**cfg.get("dataset", {}))
    dataset = FraudDataset(ds_cfg).load()

    table = BenchmarkTable()

    # (A) Legacy
    if cfg.get("run_legacy", True):
        log.info("\n" + "=" * 50)
        log.info("(A) Running Legacy Baselines …")
        run_legacy_baselines(dataset, cfg, table, setting)

    # (B) Revision L1
    if cfg.get("run_revision_l1", True):
        log.info("\n" + "=" * 50)
        log.info("(B) Running Revision Level1 …")
        run_revision_l1(dataset, cfg, table, setting)

    # (C) Revision L1+L2
    if cfg.get("run_revision_l1_l2", True):
        log.info("\n" + "=" * 50)
        log.info("(C) Running Revision Level1+Level2 …")
        run_revision_l1_l2(dataset, cfg, table, setting)

    # (D) Revision Full
    if cfg.get("run_revision_full", True):
        log.info("\n" + "=" * 50)
        log.info("(D) Running Revision Full …")
        run_revision_full(dataset, cfg, table, setting)

    # 결과 출력 및 저장
    table.print_all()
    table.to_csv(str(out_dir / f"{setting}_results.csv"))
    table.to_json(str(out_dir / f"{setting}_results.json"))

    # Markdown 저장
    md_path = out_dir / f"{setting}_results.md"
    md_path.write_text(table.to_markdown(), encoding="utf-8")
    log.info(f"[Benchmark] Markdown table → {md_path}")


if __name__ == "__main__":
    main()
