# infer.py

from __future__ import annotations

import argparse
import csv
import json
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
ROOT = Path(__file__).resolve().parent.parent
# ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from data.dataset import HierarchicalDataset
from engine.evaluator import Evaluator
from train import (
    build_hierarchical_collate_fn,
    build_model,
    load_config,
    resolve_device,
    set_seed,
)


# ======================================================================
# Utils
# ======================================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Dict[str, Any], path: str | Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_csv(rows: List[Dict[str, Any]], path: str | Path):
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return

    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    fieldnames = sorted(all_keys)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_single_dataset(cfg: Dict[str, Any], split: str) -> HierarchicalDataset:
    data_cfg = cfg["data"]
    return HierarchicalDataset(
        data_dir=data_cfg["data_dir"],
        contract_graph_path=data_cfg["contract_graph_path"],
        split=split,
        split_ratio=tuple(data_cfg.get("split_ratio", [0.7, 0.15, 0.15])),
        seed=cfg["system"].get("seed", 42),
        clip_value=data_cfg.get("clip_value", 1e12),
        apply_signed_log=data_cfg.get("apply_signed_log", True),
    )


def build_single_loader(dataset: HierarchicalDataset, cfg: Dict[str, Any]) -> DataLoader:
    data_cfg = cfg["data"]
    batch_size = data_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 0)
    pin_memory = data_cfg.get("pin_memory", False)
    persistent_workers = data_cfg.get("persistent_workers", False) and num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=build_hierarchical_collate_fn(dataset),
    )


def resolve_checkpoint_path(cfg: Dict[str, Any], checkpoint: Optional[str]) -> Path:
    if checkpoint is not None:
        return Path(checkpoint)

    run_name = cfg["experiment"]["run_name"]
    output_dir = Path(cfg["experiment"]["output_dir"])
    default_ckpt = output_dir / run_name / "checkpoints" / f"{run_name}_best.pt"
    return default_ckpt


def load_model_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"'model_state_dict' missing in checkpoint: {checkpoint_path}")

    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", -1)
    best_metric = ckpt.get("best_metric_value", None)

    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch : {epoch}")
    if best_metric is not None:
        print(f"Best metric      : {best_metric}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--mc-dropout", action="store_true")
    parser.add_argument("--mc-samples", type=int, default=20)

    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(
        cfg["system"].get("seed", 42),
        deterministic=cfg["system"].get("deterministic", True),
    )
    device = resolve_device(cfg["system"].get("device", "auto"))

    dataset = build_single_dataset(cfg, split=args.split)
    loader = build_single_loader(dataset, cfg)

    model = build_model(cfg, dataset).to(device)

    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint)
    load_model_checkpoint(model, checkpoint_path, device=device)

    evaluator = Evaluator(
        model=model,
        device=device,
        num_classes=cfg["model"].get("num_classes", 2),
        verbose=True,
    )

    threshold = (
        args.threshold
        if args.threshold is not None
        else cfg.get("evaluation", {}).get("threshold", 0.5)
    )

    run_name = cfg["experiment"]["run_name"]
    out_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path(cfg["experiment"]["output_dir"]) / run_name / f"infer_{args.split}"
    )
    ensure_dir(out_dir)

    if args.mc_dropout:
        metrics, results = evaluator.evaluate_mc_dropout(
            loader=loader,
            n_samples=args.mc_samples,
            class_weights=None,
        )
        save_json(metrics, out_dir / "mc_metrics.json")
        save_csv(results, out_dir / "mc_predictions.csv")
        print(f"Saved MC-Dropout outputs to: {out_dir}")
    else:
        metrics, results = evaluator.evaluate(
            loader=loader,
            class_weights=None,
            threshold=threshold,
        )
        save_json(metrics, out_dir / "metrics.json")
        save_csv(results, out_dir / "predictions.csv")
        print("\n" + evaluator.report(metrics))
        print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
