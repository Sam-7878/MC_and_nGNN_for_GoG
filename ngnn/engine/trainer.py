# engine/trainer.py

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from utils.metrics import (
    MetricTracker,
    compute_class_weights,
    compute_loss,
    format_metrics,
    is_better,
)


class Trainer:
    """
    Phase 1~4 통합 Trainer.

    기능:
    - 단일 epoch train / eval loop
    - class-weighted loss
    - label smoothing
    - gradient clipping
    - early stopping
    - best checkpoint 자동 저장 / 불러오기
    - epoch log dict 반환 (wandb 등과 연동 용이)

    Usage:
        trainer = Trainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            checkpoint_dir="checkpoints/",
            monitor_metric="f1",
            monitor_mode="max",
            patience=10,
            grad_clip=1.0,
            label_smoothing=0.05,
            use_class_weights=True,
        )

        for epoch in range(max_epochs):
            train_log = trainer.train_epoch(train_loader, epoch)
            val_log   = trainer.eval_epoch(val_loader, epoch)

            should_stop = trainer.step_scheduler_and_early_stop(
                val_log[monitor_metric], epoch
            )
            if should_stop:
                break
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler] = None,
        device: torch.device = torch.device("cpu"),
        checkpoint_dir: str = "checkpoints",
        run_name: str = "run",
        monitor_metric: str = "f1",
        monitor_mode: str = "max",
        patience: int = 10,
        grad_clip: float = 1.0,
        label_smoothing: float = 0.0,
        use_class_weights: bool = True,
        num_classes: int = 2,
        log_interval: int = 10,
        verbose: bool = True,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name

        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode
        self.patience = patience
        self.grad_clip = grad_clip
        self.label_smoothing = label_smoothing
        self.use_class_weights = use_class_weights
        self.num_classes = num_classes
        self.log_interval = log_interval
        self.verbose = verbose

        # Early stopping state
        self.best_metric_value: float = float("-inf") if monitor_mode == "max" else float("inf")
        self.epochs_without_improvement: int = 0
        self.best_epoch: int = 0

        # Metric tracker
        self.train_tracker = MetricTracker(num_classes=num_classes)
        self.val_tracker   = MetricTracker(num_classes=num_classes)

        # History
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Train one epoch
    # ------------------------------------------------------------------
    def train_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Run one full training epoch.

        Args:
            loader       : DataLoader yielding collated batch dicts
            epoch        : current epoch index (for logging)
            class_weights: override auto-computed class weights

        Returns:
            log dict with loss + all metrics
        """
        self.model.train()
        self.train_tracker.reset()

        total_loss = 0.0
        num_batches = 0
        t0 = time.time()

        # Compute class weights from first batch if requested
        _class_weights = class_weights

        for batch_idx, batch in enumerate(loader):
            batch = self._to_device(batch)

            labels = batch["labels"]                      # [B]

            # Auto class weights from first batch
            if self.use_class_weights and _class_weights is None:
                _class_weights = compute_class_weights(
                    labels=labels,
                    num_classes=self.num_classes,
                    device=self.device,
                )

            # Forward
            logits = self.model(batch)                    # [B, C]

            # Loss
            loss = compute_loss(
                logits=logits,
                labels=labels,
                class_weights=_class_weights if self.use_class_weights else None,
                label_smoothing=self.label_smoothing,
            )

            # Backward
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.grad_clip,
                )

            self.optimizer.step()

            # Accumulate
            total_loss += loss.item()
            num_batches += 1
            self.train_tracker.update(logits, labels)

            # Batch-level log
            if self.verbose and (batch_idx + 1) % self.log_interval == 0:
                avg_loss_so_far = total_loss / num_batches
                print(
                    f"  Epoch {epoch:>3d} | batch {batch_idx+1:>4d}/{len(loader)} "
                    f"| loss={avg_loss_so_far:.4f}"
                )

        # Epoch metrics
        metrics = self.train_tracker.compute()
        metrics["loss"] = total_loss / max(num_batches, 1)
        metrics["epoch_time_s"] = time.time() - t0
        metrics["lr"] = self._get_lr()

        if self.verbose:
            print(f"Epoch {epoch:>3d} TRAIN | {format_metrics(metrics, prefix='Train')}")

        return metrics

    # ------------------------------------------------------------------
    # Eval one epoch
    # ------------------------------------------------------------------
    @torch.no_grad()
    def eval_epoch(
        self,
        loader: DataLoader,
        epoch: int,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """
        Run one full evaluation epoch (val or test).

        Args:
            loader       : DataLoader
            epoch        : current epoch index
            class_weights: for consistent loss computation

        Returns:
            log dict with loss + all metrics
        """
        self.model.eval()
        self.val_tracker.reset()

        total_loss = 0.0
        num_batches = 0
        t0 = time.time()

        for batch in loader:
            batch = self._to_device(batch)
            labels = batch["labels"]

            logits = self.model(batch)

            loss = compute_loss(
                logits=logits,
                labels=labels,
                class_weights=class_weights,
                label_smoothing=0.0,    # no smoothing at eval
            )

            total_loss += loss.item()
            num_batches += 1
            self.val_tracker.update(logits, labels)

        metrics = self.val_tracker.compute()
        metrics["loss"] = total_loss / max(num_batches, 1)
        metrics["epoch_time_s"] = time.time() - t0

        if self.verbose:
            print(f"Epoch {epoch:>3d} VAL   | {format_metrics(metrics, prefix='Val')}")

        return metrics

    # ------------------------------------------------------------------
    # Scheduler step + early stopping
    # ------------------------------------------------------------------
    def step_scheduler_and_early_stop(
        self,
        metric_value: float,
        epoch: int,
    ) -> bool:
        """
        Step the LR scheduler and check early stopping.

        Args:
            metric_value: current epoch val metric (e.g. f1)
            epoch       : current epoch index

        Returns:
            True if training should stop
        """
        # Scheduler step
        if self.scheduler is not None:
            if isinstance(
                self.scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                self.scheduler.step(metric_value)
            else:
                self.scheduler.step()

        # Best check
        if is_better(metric_value, self.best_metric_value, mode=self.monitor_mode):
            self.best_metric_value = metric_value
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            self.save_checkpoint(epoch, tag="best")

            if self.verbose:
                print(
                    f"  ✅ New best {self.monitor_metric}={metric_value:.4f} "
                    f"at epoch {epoch}. Checkpoint saved."
                )
        else:
            self.epochs_without_improvement += 1
            if self.verbose:
                print(
                    f"  ⏳ No improvement for "
                    f"{self.epochs_without_improvement}/{self.patience} epochs."
                )

        # Early stop check
        if self.epochs_without_improvement >= self.patience:
            if self.verbose:
                print(
                    f"  🛑 Early stopping triggered at epoch {epoch}. "
                    f"Best epoch was {self.best_epoch} "
                    f"({self.monitor_metric}={self.best_metric_value:.4f})."
                )
            return True

        return False

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int,
        class_weights: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """
        Convenience method: runs full train loop.

        Returns:
            history: list of per-epoch log dicts
        """
        history = []

        for epoch in range(1, max_epochs + 1):
            train_log = self.train_epoch(train_loader, epoch, class_weights)
            val_log   = self.eval_epoch(val_loader, epoch, class_weights)

            epoch_log = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_log.items()},
                **{f"val_{k}": v   for k, v in val_log.items()},
            }
            history.append(epoch_log)

            monitor_value = val_log.get(self.monitor_metric, 0.0)

            should_stop = self.step_scheduler_and_early_stop(monitor_value, epoch)

            if should_stop:
                break

        self.history = history
        return history

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, tag: str = "best"):
        ckpt_path = self.checkpoint_dir / f"{self.run_name}_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": (
                    self.scheduler.state_dict()
                    if self.scheduler is not None
                    else None
                ),
                "best_metric_value": self.best_metric_value,
                "monitor_metric": self.monitor_metric,
            },
            ckpt_path,
        )

    def load_best_checkpoint(self) -> int:
        """
        Load best checkpoint into model.

        Returns:
            epoch of the best checkpoint
        """
        ckpt_path = self.checkpoint_dir / f"{self.run_name}_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if self.scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        epoch = ckpt["epoch"]

        if self.verbose:
            print(
                f"  ✅ Loaded best checkpoint from epoch {epoch} "
                f"({ckpt.get('monitor_metric','?')}="
                f"{ckpt.get('best_metric_value', '?'):.4f})"
            )

        return epoch

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _to_device(self, batch: Dict) -> Dict:
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            elif hasattr(v, "to"):          # PyG Batch
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result

    def _get_lr(self) -> float:
        for pg in self.optimizer.param_groups:
            return float(pg["lr"])
        return 0.0
