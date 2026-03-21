import argparse
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import torch
from torch_geometric.loader import DataLoader

from gog_fraud.data.level2.dataset import (
    Level2GraphDataset,
    infer_level2_edge_dim,
    infer_level2_node_dim,
    load_level2_graph_list_from_pt,
)
from gog_fraud.models.level2.model import Level2Model, Level2ModelConfig
from gog_fraud.training.loops.level2 import Level2Trainer, Level2TrainerConfig


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def build_dataloader(
    graphs: List,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def save_checkpoint(
    checkpoint_path: Path,
    model,
    model_cfg: Level2ModelConfig,
    trainer_cfg: Level2TrainerConfig,
    epoch: int,
    train_metrics: dict,
    valid_metrics: dict,
):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config":     asdict(model_cfg),
            "trainer_config":   asdict(trainer_cfg),
            "epoch":            epoch,
            "train_metrics":    train_metrics,
            "valid_metrics":    valid_metrics,
        },
        checkpoint_path,
    )


def select_monitor_score(valid_metrics: dict) -> float:
    if "pr_auc" in valid_metrics:
        return float(valid_metrics["pr_auc"])
    return -float(valid_metrics["loss"])


# ──────────────────────────────────────────────
# Core run_training (importable by tests)
# ──────────────────────────────────────────────

def run_training(
    train_graphs: List,
    valid_graphs: List,
    output_dir: str,
    model_cfg:   Optional[Level2ModelConfig]  = None,
    trainer_cfg: Optional[Level2TrainerConfig] = None,
    device:      Optional[str] = None,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer_cfg = trainer_cfg or Level2TrainerConfig()

    if model_cfg is None:
        in_dim   = infer_level2_node_dim(train_graphs)
        edge_dim = infer_level2_edge_dim(train_graphs)
        model_cfg = Level2ModelConfig(
            in_dim=in_dim,
            hidden_dim=64,
            num_layers=2,
            num_heads=4,
            dropout=0.2,
            edge_dim=edge_dim,
            readout="meanmax",
            out_dim=1,
        )

    train_loader = build_dataloader(
        train_graphs, batch_size=trainer_cfg.batch_size, shuffle=True
    )
    valid_loader = build_dataloader(
        valid_graphs, batch_size=trainer_cfg.batch_size, shuffle=False
    )

    model     = Level2Model(model_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainer_cfg.lr,
        weight_decay=trainer_cfg.weight_decay,
    )
    trainer = Level2Trainer(
        model=model,
        optimizer=optimizer,
        cfg=trainer_cfg,
        device=device,
    )

    output_dir     = Path(output_dir)
    best_ckpt_path = output_dir / "best.pt"
    last_ckpt_path = output_dir / "last.pt"

    history    = []
    best_score = float("-inf")

    for epoch in range(1, trainer_cfg.epochs + 1):
        train_metrics = trainer.train_one_epoch(train_loader)
        valid_metrics = trainer.evaluate(valid_loader)

        history.append({
            "epoch":  epoch,
            "train":  train_metrics,
            "valid":  valid_metrics,
        })

        monitor_score = select_monitor_score(valid_metrics)
        if monitor_score > best_score:
            best_score = monitor_score
            save_checkpoint(
                checkpoint_path=best_ckpt_path,
                model=trainer.model,
                model_cfg=model_cfg,
                trainer_cfg=trainer_cfg,
                epoch=epoch,
                train_metrics=train_metrics,
                valid_metrics=valid_metrics,
            )

        save_checkpoint(
            checkpoint_path=last_ckpt_path,
            model=trainer.model,
            model_cfg=model_cfg,
            trainer_cfg=trainer_cfg,
            epoch=epoch,
            train_metrics=train_metrics,
            valid_metrics=valid_metrics,
        )

        print(
            f"[L2 Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"valid_loss={valid_metrics['loss']:.4f}  "
            f"valid_pr_auc={valid_metrics.get('pr_auc', 0.0):.4f}"
        )

    return {
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
        "history":         history,
        "model_config":    asdict(model_cfg),
        "trainer_config":  asdict(trainer_cfg),
    }


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train Level 2 model")
    parser.add_argument("--train-data", type=str, required=True,
                        help="Path to saved List[PyG Data] for Level 2 train graphs")
    parser.add_argument("--valid-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--epochs",      type=int,   default=10)
    parser.add_argument("--batch-size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--weight-decay",type=float, default=1e-4)

    parser.add_argument("--hidden-dim",  type=int,   default=128)
    parser.add_argument("--num-layers",  type=int,   default=2)
    parser.add_argument("--num-heads",   type=int,   default=4)
    parser.add_argument("--dropout",     type=float, default=0.2)
    parser.add_argument("--readout",     type=str,   default="meanmax",
                        choices=["mean", "max", "add", "meanmax"])

    parser.add_argument("--grad-accum-steps", type=int,   default=1)
    parser.add_argument("--max-grad-norm",    type=float, default=1.0)
    parser.add_argument("--pos-weight",       type=float, default=None)
    parser.add_argument("--device",           type=str,   default=None)
    parser.add_argument("--no-amp",           action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    train_graphs = load_level2_graph_list_from_pt(args.train_data)
    valid_graphs = load_level2_graph_list_from_pt(args.valid_data)

    in_dim   = infer_level2_node_dim(train_graphs)
    edge_dim = infer_level2_edge_dim(train_graphs)

    model_cfg = Level2ModelConfig(
        in_dim=in_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        edge_dim=edge_dim,
        readout=args.readout,
        out_dim=1,
    )
    trainer_cfg = Level2TrainerConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        use_amp=not args.no_amp,
        pos_weight=args.pos_weight,
    )

    run_training(
        train_graphs=train_graphs,
        valid_graphs=valid_graphs,
        output_dir=args.output_dir,
        model_cfg=model_cfg,
        trainer_cfg=trainer_cfg,
        device=args.device,
    )


if __name__ == "__main__":
    main()
