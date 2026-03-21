# train.py

from __future__ import annotations

import argparse
import csv
import inspect
import json
import random
import shutil
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
ROOT = Path(__file__).resolve().parent.parent
# ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from data.dataset import HierarchicalDataset
from engine.evaluator import Evaluator
from engine.trainer import Trainer
from model.hierarchical_gnn import HierarchicalGNN
from utils.metrics import compute_class_weights


# ======================================================================
# Utils
# ======================================================================

def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


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


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ======================================================================
# Collate
# ======================================================================

def build_hierarchical_collate_fn(dataset: HierarchicalDataset):
    """
    Batch dict expected by HierarchicalGNN / Trainer / Evaluator.

    Returns:
        {
            "local_batch": PyG Batch,
            "contract_ids": LongTensor [B],
            "contract_names": List[str],
            "labels": LongTensor [B],
            "global_features": FloatTensor [N, Fg],
            "global_edge_index": LongTensor [2, E],
        }
    """
    global_features = dataset.global_features
    global_edge_index = dataset.global_edge_index

    def collate_fn(items: List[Dict[str, Any]]) -> Dict[str, Any]:
        local_graphs = [item["local_graph"] for item in items]
        local_batch = Batch.from_data_list(local_graphs)

        contract_ids = torch.tensor(
            [item["contract_id"] for item in items],
            dtype=torch.long,
        )
        labels = torch.tensor(
            [item["label"] for item in items],
            dtype=torch.long,
        )
        contract_names = [item["contract_name"] for item in items]

        return {
            "local_batch": local_batch,
            "contract_ids": contract_ids,
            "contract_names": contract_names,
            "labels": labels,
            "global_features": global_features,
            "global_edge_index": global_edge_index,
        }

    return collate_fn


# ======================================================================
# Builders
# ======================================================================

def build_datasets(cfg: Dict[str, Any]):
    data_cfg = cfg["data"]

    common_kwargs = dict(
        data_dir=data_cfg["data_dir"],
        contract_graph_path=data_cfg["contract_graph_path"],
        split_ratio=tuple(data_cfg.get("split_ratio", [0.7, 0.15, 0.15])),
        seed=cfg["system"]["seed"],
        clip_value=data_cfg.get("clip_value", 1e12),
        apply_signed_log=data_cfg.get("apply_signed_log", True),
    )

    train_ds = HierarchicalDataset(split="train", **common_kwargs)
    val_ds = HierarchicalDataset(split="val", **common_kwargs)
    test_ds = HierarchicalDataset(split="test", **common_kwargs)

    return train_ds, val_ds, test_ds


def build_dataloaders(
    train_ds: HierarchicalDataset,
    val_ds: HierarchicalDataset,
    test_ds: HierarchicalDataset,
    cfg: Dict[str, Any],
):
    data_cfg = cfg["data"]

    batch_size = data_cfg.get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 0)
    pin_memory = data_cfg.get("pin_memory", False)
    persistent_workers = data_cfg.get("persistent_workers", False) and num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=build_hierarchical_collate_fn(train_ds),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=build_hierarchical_collate_fn(val_ds),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=build_hierarchical_collate_fn(test_ds),
    )

    return train_loader, val_loader, test_loader


def build_model(cfg: Dict[str, Any], train_ds: HierarchicalDataset) -> torch.nn.Module:
    """
    Robust model builder:
    - merges dataset dims + config["model"]
    - filters kwargs by actual HierarchicalGNN signature
    """
    model_cfg = dict(cfg["model"])

    candidate_kwargs = {
        "node_dim": train_ds.node_dim,
        "edge_dim": train_ds.edge_dim,
        "global_feat_dim": train_ds.global_feat_dim,
        **model_cfg,
    }

    sig = inspect.signature(HierarchicalGNN.__init__)
    supported = {
        k: v for k, v in candidate_kwargs.items()
        if k in sig.parameters
    }

    model = HierarchicalGNN(**supported)
    return model


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]):
    opt_cfg = cfg["optimizer"]
    name = opt_cfg.get("name", "adamw").lower()
    lr = opt_cfg.get("lr", 1e-3)
    weight_decay = opt_cfg.get("weight_decay", 0.0)

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    elif name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer, cfg: Dict[str, Any]):
    sch_cfg = cfg.get("scheduler", {})
    name = sch_cfg.get("name", "none").lower()

    if name == "none":
        return None

    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sch_cfg.get("mode", "max"),
            factor=sch_cfg.get("factor", 0.5),
            patience=sch_cfg.get("patience", 3),
            min_lr=sch_cfg.get("min_lr", 1e-6),
        )

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sch_cfg.get("t_max", 20),
            eta_min=sch_cfg.get("eta_min", 1e-6),
        )

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sch_cfg.get("step_size", 10),
            gamma=sch_cfg.get("gamma", 0.5),
        )

    raise ValueError(f"Unsupported scheduler: {name}")


def build_class_weights(
    train_ds: HierarchicalDataset,
    num_classes: int,
    device: torch.device,
    enabled: bool,
):
    if not enabled:
        return None

    labels = torch.tensor(
        [sample["label"] for sample in train_ds.samples],
        dtype=torch.long,
    )
    return compute_class_weights(
        labels=labels,
        num_classes=num_classes,
        device=device,
    )


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config yaml",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    system_cfg = cfg["system"]
    train_cfg = cfg["training"]

    set_seed(
        seed=system_cfg.get("seed", 42),
        deterministic=system_cfg.get("deterministic", True),
    )
    device = resolve_device(system_cfg.get("device", "auto"))

    output_root = ensure_dir(cfg["experiment"]["output_dir"])
    run_name = cfg["experiment"]["run_name"]
    run_dir = ensure_dir(output_root / run_name)
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")

    # save config snapshot
    shutil.copyfile(args.config, run_dir / "used_config.yaml")

    print("=" * 70)
    print(f"Run name       : {run_name}")
    print(f"Device         : {device}")
    print(f"Output dir     : {run_dir}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\n[1/6] Building datasets...")
    train_ds, val_ds, test_ds = build_datasets(cfg)

    print(f"  Train size    : {len(train_ds)}")
    print(f"  Val size      : {len(val_ds)}")
    print(f"  Test size     : {len(test_ds)}")
    print(f"  node_dim      : {train_ds.node_dim}")
    print(f"  edge_dim      : {train_ds.edge_dim}")
    print(f"  global_feat   : {train_ds.global_feat_dim}")
    print(f"  global_edges  : {tuple(train_ds.global_edge_index.shape)}")

    print("\n[2/6] Building dataloaders...")
    train_loader, val_loader, test_loader = build_dataloaders(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        cfg=cfg,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\n[3/6] Building model...")
    model = build_model(cfg, train_ds).to(device)
    print(model)
    print(f"\nTrainable params: {count_parameters(model):,}")

    # ------------------------------------------------------------------
    # Optimizer / Scheduler / Class weights
    # ------------------------------------------------------------------
    print("\n[4/6] Building optimizer and scheduler...")
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    class_weights = build_class_weights(
        train_ds=train_ds,
        num_classes=cfg["model"].get("num_classes", 2),
        device=device,
        enabled=train_cfg.get("use_class_weights", True),
    )
    if class_weights is not None:
        print(f"  class_weights : {class_weights.detach().cpu().tolist()}")
    else:
        print("  class_weights : None")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    print("\n[5/6] Training...")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        run_name=run_name,
        monitor_metric=train_cfg.get("monitor_metric", "f1"),
        monitor_mode=train_cfg.get("monitor_mode", "max"),
        patience=train_cfg.get("patience", 10),
        grad_clip=train_cfg.get("grad_clip", 1.0),
        label_smoothing=train_cfg.get("label_smoothing", 0.0),
        use_class_weights=train_cfg.get("use_class_weights", True),
        num_classes=cfg["model"].get("num_classes", 2),
        log_interval=train_cfg.get("log_interval", 10),
        verbose=train_cfg.get("verbose", True),
    )

    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=train_cfg.get("epochs", 30),
        class_weights=class_weights,
    )

    save_json({"history": history}, run_dir / "history.json")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    print("\n[6/6] Loading best checkpoint and evaluating...")
    trainer.load_best_checkpoint()

    evaluator = Evaluator(
        model=model,
        device=device,
        num_classes=cfg["model"].get("num_classes", 2),
        verbose=True,
    )

    eval_cfg = cfg.get("evaluation", {})
    threshold = eval_cfg.get("threshold", 0.5)

    test_metrics, test_results = evaluator.evaluate(
        loader=test_loader,
        class_weights=class_weights,
        threshold=threshold,
    )

    print("\n" + evaluator.report(test_metrics))

    save_json(test_metrics, run_dir / "test_metrics.json")
    save_csv(test_results, run_dir / "test_predictions.csv")

    # Optional MC-Dropout
    mc_cfg = eval_cfg.get("mc_dropout", {})
    if mc_cfg.get("enabled", False):
        print("\nRunning MC-Dropout evaluation...")
        mc_metrics, mc_results = evaluator.evaluate_mc_dropout(
            loader=test_loader,
            n_samples=mc_cfg.get("n_samples", 20),
            class_weights=class_weights,
        )
        save_json(mc_metrics, run_dir / "test_mc_metrics.json")
        save_csv(mc_results, run_dir / "test_mc_predictions.csv")

    print("\nDone.")
    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
