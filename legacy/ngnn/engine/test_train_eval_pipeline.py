# test_train_eval_pipeline.py

import torch
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# 프로젝트 루트를 PYTHONPATH에 추가 (common 모듈 로드용)
ROOT = Path(__file__).resolve().parent.parent
# ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from data.dataset import HierarchicalDataset
from data.collate import hierarchical_collate_fn
from model.hierarchical_gnn import HierarchicalGNN
from engine.trainer import Trainer
from engine.evaluator import Evaluator
from utils.metrics import compute_class_weights


def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HIDDEN_DIM = 64
    BATCH_SIZE = 8
    MAX_EPOCHS = 3       # 빠른 검증용

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    train_ds = HierarchicalDataset(
        data_dir="../../../_data/GoG/polygon/graphs",
        contract_graph_path="../../../_data/GoG/polygon/polygon_hybrid_graph.pt",
        split="train",
    )
    val_ds = HierarchicalDataset(
        data_dir="../../../_data/GoG/polygon/graphs",
        contract_graph_path="../../../_data/GoG/polygon/polygon_hybrid_graph.pt",
        split="val",
    )
    test_ds = HierarchicalDataset(
        data_dir="../../../_data/GoG/polygon/graphs",
        contract_graph_path="../../../_data/GoG/polygon/polygon_hybrid_graph.pt",
        split="test",
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=hierarchical_collate_fn
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=hierarchical_collate_fn
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=hierarchical_collate_fn
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = HierarchicalGNN(
        node_dim=train_ds.node_dim,
        edge_dim=train_ds.edge_dim,
        hidden_dim=HIDDEN_DIM,
        global_feat_dim=train_ds.global_feat_dim,
        num_classes=2,
        use_global_gnn=True,
    ).to(DEVICE)

    # ------------------------------------------------------------------
    # Optimizer + Scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # ------------------------------------------------------------------
    # Class weights (from full train label distribution)
    # ------------------------------------------------------------------
    all_train_labels = torch.tensor(
        [s["label"] for s in train_ds.samples], dtype=torch.long
    )
    class_weights = compute_class_weights(
        all_train_labels, num_classes=2, device=DEVICE
    )
    print(f"Class weights: {class_weights}")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        checkpoint_dir="checkpoints",
        run_name="phase1_base",
        monitor_metric="f1",
        monitor_mode="max",
        patience=5,
        grad_clip=1.0,
        label_smoothing=0.05,
        use_class_weights=True,
        num_classes=2,
        log_interval=5,
        verbose=True,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        max_epochs=MAX_EPOCHS,
        class_weights=class_weights,
    )

    print("\nTraining history:")
    for row in history:
        print(
            f"  epoch={row['epoch']}  "
            f"train_loss={row['train_loss']:.4f}  "
            f"val_loss={row['val_loss']:.4f}  "
            f"val_f1={row['val_f1']:.4f}"
        )

    # ------------------------------------------------------------------
    # Load best checkpoint & Test
    # ------------------------------------------------------------------
    trainer.load_best_checkpoint()

    evaluator = Evaluator(model=model, device=DEVICE, num_classes=2, verbose=True)
    metrics, results = evaluator.evaluate(test_loader, class_weights=class_weights)
    print(evaluator.report(metrics))

    # ------------------------------------------------------------------
    # MC-Dropout evaluation (Phase 3 미리보기)
    # ------------------------------------------------------------------
    mc_metrics, mc_results = evaluator.evaluate_mc_dropout(
        test_loader, n_samples=10
    )
    print(f"\nMC-Dropout entropy sample: {mc_results[0]['entropy']:.4f}")
    print(f"MC-Dropout epistemic sample: {mc_results[0]['epistemic']:.4f}")

    # ------------------------------------------------------------------
    # Final assertions
    # ------------------------------------------------------------------
    assert "f1" in metrics
    assert "auroc" in metrics
    assert len(results) == len(test_ds)
    assert "entropy" in mc_results[0]

    print("\n✅ Full pipeline test PASSED.")


if __name__ == "__main__":
    main()
