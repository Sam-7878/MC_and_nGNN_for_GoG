# utils/metrics.py

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ======================================================================
# Core metric computation
# ======================================================================

class MetricTracker:
    """
    Accumulates predictions and labels over batches,
    then computes all metrics at epoch end.

    Usage:
        tracker = MetricTracker()

        for batch in loader:
            logits = model(batch)
            labels = batch["labels"]
            tracker.update(logits, labels)

        metrics = tracker.compute()
        tracker.reset()
    """

    def __init__(self, num_classes: int = 2, device: str = "cpu"):
        self.num_classes = num_classes
        self.device = device
        self.reset()

    def reset(self):
        self._all_logits: List[torch.Tensor] = []
        self._all_labels: List[torch.Tensor] = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            logits : FloatTensor [B, num_classes]
            labels : LongTensor  [B]
        """
        self._all_logits.append(logits.detach().cpu())
        self._all_labels.append(labels.detach().cpu())

    def compute(self) -> Dict[str, float]:
        if not self._all_logits:
            return {}

        logits = torch.cat(self._all_logits, dim=0)   # [N, C]
        labels = torch.cat(self._all_labels, dim=0)   # [N]

        return compute_metrics(logits, labels, num_classes=self.num_classes)


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int = 2,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute full evaluation metrics from logits and labels.

    Args:
        logits     : FloatTensor [N, num_classes]
        labels     : LongTensor  [N]
        num_classes: number of classes
        threshold  : decision threshold for binary classification

    Returns:
        dict of metric_name → float
    """
    probs = F.softmax(logits, dim=-1)           # [N, C]
    preds = torch.argmax(probs, dim=-1)         # [N]

    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    probs_np = probs.cpu().numpy()

    metrics: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Basic
    # ------------------------------------------------------------------
    metrics["accuracy"] = float((preds == labels).float().mean().item())

    # ------------------------------------------------------------------
    # Binary classification metrics (fraud class = 1)
    # ------------------------------------------------------------------
    if num_classes == 2:
        fraud_probs = probs_np[:, 1]

        # precision / recall / f1
        metrics["precision"] = float(
            precision_score(labels_np, preds_np, zero_division=0)
        )
        metrics["recall"] = float(
            recall_score(labels_np, preds_np, zero_division=0)
        )
        metrics["f1"] = float(
            f1_score(labels_np, preds_np, zero_division=0)
        )

        # AUC-ROC
        try:
            metrics["auroc"] = float(roc_auc_score(labels_np, fraud_probs))
        except ValueError:
            metrics["auroc"] = float("nan")

        # AUC-PR (AUPRC)
        try:
            metrics["auprc"] = float(
                average_precision_score(labels_np, fraud_probs)
            )
        except ValueError:
            metrics["auprc"] = float("nan")

        # Confusion matrix components
        try:
            tn, fp, fn, tp = confusion_matrix(
                labels_np, preds_np, labels=[0, 1]
            ).ravel()
            metrics["tp"] = float(tp)
            metrics["fp"] = float(fp)
            metrics["fn"] = float(fn)
            metrics["tn"] = float(tn)
        except ValueError:
            metrics["tp"] = metrics["fp"] = metrics["fn"] = metrics["tn"] = 0.0

        # Specificity (True Negative Rate)
        tn_val = metrics.get("tn", 0.0)
        fp_val = metrics.get("fp", 0.0)
        denom = tn_val + fp_val
        metrics["specificity"] = float(tn_val / denom) if denom > 0 else 0.0

        # Best threshold based on F1
        metrics["best_threshold"], metrics["best_f1_at_threshold"] = (
            _find_best_threshold(labels_np, fraud_probs)
        )

    # ------------------------------------------------------------------
    # Multiclass metrics
    # ------------------------------------------------------------------
    else:
        metrics["macro_f1"] = float(
            f1_score(labels_np, preds_np, average="macro", zero_division=0)
        )
        metrics["weighted_f1"] = float(
            f1_score(labels_np, preds_np, average="weighted", zero_division=0)
        )
        try:
            metrics["auroc"] = float(
                roc_auc_score(
                    labels_np,
                    probs_np,
                    multi_class="ovr",
                    average="macro",
                )
            )
        except ValueError:
            metrics["auroc"] = float("nan")

    return metrics


def compute_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """
    Cross-entropy loss with optional class weighting and label smoothing.

    Args:
        logits        : FloatTensor [B, C]
        labels        : LongTensor  [B]
        class_weights : FloatTensor [C], optional
        label_smoothing: float in [0, 1)

    Returns:
        scalar loss tensor
    """
    return F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        label_smoothing=label_smoothing,
    )


# ======================================================================
# Threshold tuning
# ======================================================================

def _find_best_threshold(
    labels: np.ndarray,
    probs: np.ndarray,
) -> Tuple[float, float]:
    """
    Sweep thresholds and find one that maximizes F1.
    """
    precision, recall, thresholds = precision_recall_curve(labels, probs)

    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )

    best_idx = int(np.argmax(f1_scores[:-1]))  # last point has no threshold
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    return best_threshold, best_f1


# ======================================================================
# Class weight computation for imbalanced datasets
# ======================================================================

def compute_class_weights(
    labels: torch.Tensor,
    num_classes: int = 2,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for imbalanced datasets.

    w_c = N / (num_classes * count_c)

    Args:
        labels     : LongTensor [N]
        num_classes: int
        device     : target device

    Returns:
        weights: FloatTensor [num_classes]
    """
    counts = torch.zeros(num_classes, dtype=torch.float32)

    for c in range(num_classes):
        counts[c] = (labels == c).sum().float()

    counts = counts.clamp(min=1.0)
    weights = labels.size(0) / (num_classes * counts)
    return weights.to(device)


# ======================================================================
# Metric logging helpers
# ======================================================================

def format_metrics(
    metrics: Dict[str, float],
    prefix: str = "",
    keys: Optional[List[str]] = None,
) -> str:
    """
    Format metric dict to a human-readable string.

    Args:
        metrics: dict of metric_name → float
        prefix : optional prefix (e.g., "Train" / "Val")
        keys   : specific keys to show (None = all)

    Returns:
        formatted string
    """
    show_keys = keys or [
        "loss", "accuracy", "precision", "recall",
        "f1", "auroc", "auprc", "specificity",
    ]
    parts = []
    for k in show_keys:
        v = metrics.get(k, None)
        if v is not None:
            if isinstance(v, float) and not (
                v == float("inf") or v == float("-inf") or v != v
            ):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")

    body = "  ".join(parts)
    if prefix:
        return f"[{prefix}] {body}"
    return body


def is_better(
    new_value: float,
    best_value: float,
    mode: str = "max",
    delta: float = 1e-6,
) -> bool:
    """
    Returns True if new_value is better than best_value.
    """
    if mode == "max":
        return new_value > best_value + delta
    elif mode == "min":
        return new_value < best_value - delta
    else:
        raise ValueError(f"mode must be 'max' or 'min', got {mode}")
