# train_ablation.py

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
import traceback
from typing import Any, Dict, List

import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
ROOT = Path(__file__).resolve().parent.parent
# ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from engine.evaluator import Evaluator
from engine.trainer import Trainer
from train import (
    build_class_weights,
    build_dataloaders,
    build_datasets,
    build_model,
    build_optimizer,
    build_scheduler,
    count_parameters,
    ensure_dir,
    load_config,
    resolve_device,
    save_csv,
    save_json,
    set_seed,
)


# ======================================================================
# Utils
# ======================================================================

def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update nested dict.
    """
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_ablation_plan(path: str | None) -> List[Dict[str, Any]]:
    """
    If external ablation yaml/json is provided, load it.
    Otherwise use built-in defaults.
    """
    if path is None:
        return default_ablation_plan()

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Ablation config not found: {p}")

    if p.suffix.lower() in [".json"]:
        with open(p, "r", encoding="utf-8") as f:
            obj = json.load(f)
    else:
        import yaml
        with open(p, "r", encoding="utf-8") as f:
            obj = yaml.safe_load(f)

    if "ablations" not in obj or not isinstance(obj["ablations"], list):
        raise ValueError("Ablation config must contain a list key: 'ablations'")

    return obj["ablations"]


def default_ablation_plan() -> List[Dict[str, Any]]:
    """
    Safe default ablation variants.
    Adjust freely.
    """
    return [
        {
            "name": "base",
            "overrides": {}
        },
        {
            "name": "local_only",
            "overrides": {
                "model": {
                    "use_global_gnn": False
                }
            }
        },
        {
            "name": "no_signed_log",
            "overrides": {
                "data": {
                    "apply_signed_log": False
                }
            }
        },
        {
            "name": "no_class_weights",
            "overrides": {
                "training": {
                    "use_class_weights": False
                }
            }
        },
        {
            "name": "hidden_128",
            "overrides": {
                "model": {
                    "hidden_dim": 128
                }
            }
        },
        {
            "name": "dropout_00",
            "overrides": {
                "model": {
                    "dropout": 0.0
                }
            }
        },
    ]


def save_summary(rows: List[Dict[str, Any]], output_dir: Path):
    save_csv(rows, output_dir / "ablation_summary.csv")
    save_json({"results": rows}, output_dir / "ablation_summary.json")


# ======================================================================
# Single run
# ======================================================================

def run_one_experiment(base_cfg: Dict[str, Any], ablation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run one ablation experiment and return summary row.
    """
    cfg = copy.deepcopy(base_cfg)
    name = ablation["name"]
    overrides = ablation.get("overrides", {})
    deep_update(cfg, overrides)

    base_run_name = cfg["experiment"]["run_name"]
    cfg["experiment"]["run_name"] = f"{base_run_name}__{name}"

    set_seed(
        cfg["system"].get("seed", 42),
        deterministic=cfg["system"].get("deterministic", True),
    )
    device = resolve_device(cfg["system"].get("device", "auto"))

    run_name = cfg["experiment"]["run_name"]
    output_root = ensure_dir(cfg["experiment"]["output_dir"])
    run_dir = ensure_dir(output_root / run_name)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")

    # save merged config
    try:
        import yaml
        with open(run_dir / "used_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    except Exception:
        pass

    print("=" * 80)
    print(f"[ABLATION] {name}")
    print(f"Run name  : {run_name}")
    print(f"Run dir   : {run_dir}")
    print(f"Device    : {device}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_ds, val_ds, test_ds = build_datasets(cfg)
    train_loader, val_loader, test_loader = build_dataloaders(train_ds, val_ds, test_ds, cfg)

    print(f"Train/Val/Test = {len(train_ds)} / {len(val_ds)} / {len(test_ds)}")
    print(
        f"Dims node={train_ds.node_dim}, edge={train_ds.edge_dim}, "
        f"global_feat={train_ds.global_feat_dim}"
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = build_model(cfg, train_ds).to(device)
    n_params = count_parameters(model)
    print(f"Trainable params: {n_params:,}")

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    class_weights = build_class_weights(
        train_ds=train_ds,
        num_classes=cfg["model"].get("num_classes", 2),
        device=device,
        enabled=cfg["training"].get("use_class_weights", True),
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        run_name=run_name,
        monitor_metric=cfg["training"].get("monitor_metric", "f1"),
        monitor_mode=cfg["training"].get("monitor_mode", "max"),
        patience=cfg["training"].get("patience", 10),
        grad_clip=cfg["training"].get("grad_clip", 1.0),
        label_smoothing=cfg["training"].get("label_smoothing", 0.0),
        use_class_weights=cfg["training"].get("use_class_weights", True),
        num_classes=cfg["model"].get("num_classes", 2),
        log_interval=cfg["training"].get("log_interval", 10),
        verbose=cfg["training"].get("verbose", True),
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=cfg["training"].get("epochs", 30),
        class_weights=class_weights,
    )
    save_json({"history": history}, run_dir / "history.json")

    trainer.load_best_checkpoint()

    evaluator = Evaluator(
        model=model,
        device=device,
        num_classes=cfg["model"].get("num_classes", 2),
        verbose=True,
    )

    threshold = cfg.get("evaluation", {}).get("threshold", 0.5)
    test_metrics, test_results = evaluator.evaluate(
        loader=test_loader,
        class_weights=class_weights,
        threshold=threshold,
    )

    save_json(test_metrics, run_dir / "test_metrics.json")
    save_csv(test_results, run_dir / "test_predictions.csv")

    # optional MC-dropout
    mc_cfg = cfg.get("evaluation", {}).get("mc_dropout", {})
    if mc_cfg.get("enabled", False):
        mc_metrics, mc_results = evaluator.evaluate_mc_dropout(
            loader=test_loader,
            n_samples=mc_cfg.get("n_samples", 20),
            class_weights=class_weights,
        )
        save_json(mc_metrics, run_dir / "test_mc_metrics.json")
        save_csv(mc_results, run_dir / "test_mc_predictions.csv")

    summary_row = {
        "ablation": name,
        "run_name": run_name,
        "status": "ok",
        "num_params": n_params,
        "best_epoch": trainer.best_epoch,
        "best_metric_value": trainer.best_metric_value,
        "test_loss": test_metrics.get("loss"),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_precision": test_metrics.get("precision"),
        "test_recall": test_metrics.get("recall"),
        "test_f1": test_metrics.get("f1"),
        "test_auroc": test_metrics.get("auroc"),
        "test_auprc": test_metrics.get("auprc"),
        "test_specificity": test_metrics.get("specificity"),
        "overrides": json.dumps(overrides, ensure_ascii=False),
    }
    return summary_row


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--ablation-config", type=str, default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    ablations = load_ablation_plan(args.ablation_config)

    output_root = ensure_dir(base_cfg["experiment"]["output_dir"])
    base_run_name = base_cfg["experiment"]["run_name"]
    summary_dir = ensure_dir(output_root / f"{base_run_name}__ablation_suite")

    try:
        shutil.copyfile(args.config, summary_dir / "base_config.yaml")
    except Exception:
        pass

    summary_rows: List[Dict[str, Any]] = []

    for ablation in ablations:
        name = ablation.get("name", "unnamed")

        try:
            row = run_one_experiment(base_cfg, ablation)
            summary_rows.append(row)
            save_summary(summary_rows, summary_dir)

        except Exception as e:
            err_row = {
                "ablation": name,
                "run_name": f"{base_run_name}__{name}",
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
            summary_rows.append(err_row)
            save_summary(summary_rows, summary_dir)

            print(f"\n[FAILED] {name}")
            print(traceback.format_exc())

            if not args.continue_on_error:
                raise

    print("\nAblation finished.")
    print(f"Summary saved to: {summary_dir}")


if __name__ == "__main__":
    main()
```__
