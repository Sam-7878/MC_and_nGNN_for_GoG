import argparse
from dataclasses import asdict
from pathlib import Path
from sys import path
from typing import List, Optional

import torch
from torch_geometric.loader import DataLoader

from gog_fraud.models.level1.model import Level1Model, Level1ModelConfig
from gog_fraud.training.loops.level1 import Level1Trainer, Level1TrainerConfig


def load_graph_list(path: str):
    # graphs = torch.load(path, map_location="cpu")
    graphs = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(graphs, list):
        raise TypeError(f"Expected a list of PyG Data objects, but got: {type(graphs)}")
    return graphs


def infer_in_dim(graphs: List) -> int:
    if len(graphs) == 0:
        raise ValueError("Cannot infer input dimension from empty graph list")
    if not hasattr(graphs[0], "x"):
        raise ValueError("Each graph must contain node features in .x")
    return int(graphs[0].x.size(-1))


def infer_struct_dim(graphs: List) -> int:
    if len(graphs) == 0:
        return 0

    g = graphs[0]
    for attr_name in ("struct_feat", "graph_attr", "struct_x"):
        if hasattr(g, attr_name):
            feat = getattr(g, attr_name)
            if feat is None:
                continue
            if feat.dim() == 1:
                return int(feat.size(0))
            if feat.dim() == 2:
                return int(feat.size(-1))
    return 0


def build_dataloader(graphs: List, batch_size: int, shuffle: bool, num_workers: int = 0):
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def save_checkpoint(
    checkpoint_path: Path,
    model,
    model_cfg: Level1ModelConfig,
    trainer_cfg: Level1TrainerConfig,
    epoch: int,
    train_metrics,
    valid_metrics,
):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": asdict(model_cfg),
            "trainer_config": asdict(trainer_cfg),
            "epoch": epoch,
            "train_metrics": train_metrics,
            "valid_metrics": valid_metrics,
        },
        checkpoint_path,
    )


def select_monitor_score(valid_metrics: dict) -> float:
    if "pr_auc" in valid_metrics:
        return float(valid_metrics["pr_auc"])
    return -float(valid_metrics["loss"])


def run_training(
    train_graphs: List,
    valid_graphs: List,
    output_dir: str,
    model_cfg: Optional[Level1ModelConfig] = None,
    trainer_cfg: Optional[Level1TrainerConfig] = None,
    device: Optional[str] = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    trainer_cfg = trainer_cfg or Level1TrainerConfig()

    if model_cfg is None:
        model_cfg = Level1ModelConfig(
            in_dim=infer_in_dim(train_graphs),
            hidden_dim=128,
            num_layers=3,
            dropout=0.2,
            readout="meanmax",
            struct_dim=infer_struct_dim(train_graphs),
            struct_hidden_dim=64,
            out_dim=1,
        )

    train_loader = build_dataloader(
        train_graphs,
        batch_size=trainer_cfg.batch_size,
        shuffle=True,
    )
    valid_loader = build_dataloader(
        valid_graphs,
        batch_size=trainer_cfg.batch_size,
        shuffle=False,
    )

    model = Level1Model(model_cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=trainer_cfg.lr,
        weight_decay=trainer_cfg.weight_decay,
    )
    trainer = Level1Trainer(
        model=model,
        optimizer=optimizer,
        cfg=trainer_cfg,
        device=device,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_score = float("-inf")
    best_ckpt_path = output_dir / "best.pt"
    last_ckpt_path = output_dir / "last.pt"

    for epoch in range(1, trainer_cfg.epochs + 1):
        train_metrics = trainer.train_one_epoch(train_loader)
        valid_metrics = trainer.evaluate(valid_loader)

        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "valid": valid_metrics,
            }
        )

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
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"valid_loss={valid_metrics['loss']:.4f} "
            f"valid_pr_auc={valid_metrics.get('pr_auc', 0.0):.4f}"
        )

    return {
        "best_checkpoint": str(best_ckpt_path),
        "last_checkpoint": str(last_ckpt_path),
        "history": history,
        "model_config": asdict(model_cfg),
        "trainer_config": asdict(trainer_cfg),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, required=True, help="Path to torch-saved List[PyG Data]")
    parser.add_argument("--valid-data", type=str, required=True, help="Path to torch-saved List[PyG Data]")
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--readout", type=str, default="meanmax", choices=["mean", "max", "meanmax"])
    parser.add_argument("--struct-hidden-dim", type=int, default=64)
    parser.add_argument("--pos-weight", type=float, default=None)

    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    train_graphs = load_graph_list(args.train_data)
    valid_graphs = load_graph_list(args.valid_data)

    model_cfg = Level1ModelConfig(
        in_dim=infer_in_dim(train_graphs),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        readout=args.readout,
        struct_dim=infer_struct_dim(train_graphs),
        struct_hidden_dim=args.struct_hidden_dim,
        out_dim=1,
    )

    trainer_cfg = Level1TrainerConfig(
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
