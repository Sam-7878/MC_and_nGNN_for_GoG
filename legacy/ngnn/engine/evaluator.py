# engine/evaluator.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.metrics import (
    MetricTracker,
    compute_loss,
    compute_metrics,
    format_metrics,
)


class Evaluator:
    """
    Phase 1~4 통합 Evaluator.

    기능:
    - test set 전체에 대한 정확한 metric 계산
    - per-sample 예측 결과 반환 (contract_name, label, pred, prob)
    - MC-Dropout 불확실성 추정 (Phase 3+)
    - best threshold 적용
    - confusion matrix / classification report 출력

    Usage:
        evaluator = Evaluator(model, device=device)
        metrics, results_df = evaluator.evaluate(test_loader)
        print(evaluator.report(metrics))
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
        num_classes: int = 2,
        verbose: bool = True,
    ):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Standard deterministic evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        class_weights: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
    ) -> Tuple[Dict[str, float], List[Dict]]:
        """
        Run full evaluation on a dataloader.

        Args:
            loader       : DataLoader
            class_weights: optional loss class weights
            threshold    : decision threshold for binary classification

        Returns:
            metrics : dict of metric_name → float
            results : list of per-sample result dicts
        """
        self.model.eval()

        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        all_names:  List[str] = []
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = self._to_device(batch)
            labels = batch["labels"]
            contract_names = batch.get("contract_names", [""] * labels.size(0))

            logits = self.model(batch)

            loss = compute_loss(
                logits=logits,
                labels=labels,
                class_weights=class_weights,
                label_smoothing=0.0,
            )

            total_loss += loss.item()
            num_batches += 1

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_names.extend(contract_names)

        # Concat
        logits_all = torch.cat(all_logits, dim=0)      # [N, C]
        labels_all = torch.cat(all_labels, dim=0)      # [N]

        # Metrics
        metrics = compute_metrics(
            logits=logits_all,
            labels=labels_all,
            num_classes=self.num_classes,
            threshold=threshold,
        )
        metrics["loss"] = total_loss / max(num_batches, 1)

        # Per-sample results
        probs_all = F.softmax(logits_all, dim=-1)       # [N, C]
        preds_all = torch.argmax(probs_all, dim=-1)     # [N]

        best_threshold = metrics.get("best_threshold", threshold)
        if self.num_classes == 2:
            preds_at_best = (probs_all[:, 1] >= best_threshold).long()
        else:
            preds_at_best = preds_all

        results: List[Dict] = []
        for i in range(len(labels_all)):
            row = {
                "contract_name"  : all_names[i],
                "label"          : int(labels_all[i].item()),
                "pred"           : int(preds_all[i].item()),
                "pred_best_thr"  : int(preds_at_best[i].item()),
                "prob_fraud"     : float(probs_all[i, 1].item())
                                   if self.num_classes == 2
                                   else None,
                "correct"        : int(
                    (preds_all[i] == labels_all[i]).item()
                ),
            }
            if self.num_classes > 2:
                for c in range(self.num_classes):
                    row[f"prob_class_{c}"] = float(probs_all[i, c].item())
            results.append(row)

        if self.verbose:
            print(f"TEST | {format_metrics(metrics, prefix='Test')}")

        return metrics, results

    # ------------------------------------------------------------------
    # MC-Dropout uncertainty estimation (Phase 3+)
    # ------------------------------------------------------------------
    def evaluate_mc_dropout(
        self,
        loader: DataLoader,
        n_samples: int = 20,
        class_weights: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, float], List[Dict]]:
        """
        MC-Dropout forward pass: runs n_samples stochastic forward passes
        to estimate predictive uncertainty.

        Model must have Dropout layers that are ACTIVE during inference.
        Call model.train() before this method OR pass a MC-dropout wrapper.

        Args:
            loader   : DataLoader
            n_samples: number of MC samples
            class_weights: optional

        Returns:
            metrics : dict (using mean predictions)
            results : per-sample dict including
                      "pred_mean", "pred_std", "prob_fraud_mean",
                      "prob_fraud_std", "entropy", "mutual_info"
        """
        # Enable dropout
        self._enable_dropout()

        all_logits_mc: List[torch.Tensor] = []  # [n_samples, N, C]
        all_labels: List[torch.Tensor] = []
        all_names: List[str] = []

        # First collect all batch logits across MC samples
        with torch.no_grad():
            for sample_idx in range(n_samples):
                sample_logits: List[torch.Tensor] = []
                sample_labels: List[torch.Tensor] = []
                sample_names: List[str] = []

                for batch in loader:
                    batch = self._to_device(batch)
                    labels = batch["labels"]
                    contract_names = batch.get(
                        "contract_names", [""] * labels.size(0)
                    )

                    logits = self.model(batch)
                    sample_logits.append(logits.cpu())

                    if sample_idx == 0:
                        sample_labels.append(labels.cpu())
                        sample_names.extend(contract_names)

                all_logits_mc.append(
                    torch.cat(sample_logits, dim=0)       # [N, C]
                )

                if sample_idx == 0:
                    all_labels = torch.cat(sample_labels, dim=0)
                    all_names = sample_names

        # Stack: [n_samples, N, C]
        logits_mc = torch.stack(all_logits_mc, dim=0)
        probs_mc  = F.softmax(logits_mc, dim=-1)          # [n_samples, N, C]

        # Mean probability
        probs_mean = probs_mc.mean(dim=0)                 # [N, C]
        probs_std  = probs_mc.std(dim=0)                  # [N, C]
        logits_mean = torch.log(probs_mean + 1e-8)        # [N, C] approx logits

        # Metrics using mean prediction
        metrics = compute_metrics(
            logits=logits_mean,
            labels=all_labels,
            num_classes=self.num_classes,
        )

        # Uncertainty decomposition
        # Predictive entropy: H[E_p[y|x, w]]
        pred_entropy = _entropy(probs_mean)                # [N]

        # Aleatoric: E_p[H[y|x, w]]  (mean of per-sample entropies)
        per_sample_entropy = _entropy(probs_mc)            # [n_samples, N]
        aleatoric = per_sample_entropy.mean(dim=0)         # [N]

        # Epistemic (BALD): predictive entropy - aleatoric
        epistemic = (pred_entropy - aleatoric).clamp(min=0.0)  # [N]

        preds_mean = torch.argmax(probs_mean, dim=-1)      # [N]

        results: List[Dict] = []
        for i in range(len(all_labels)):
            row: Dict = {
                "contract_name"   : all_names[i],
                "label"           : int(all_labels[i].item()),
                "pred_mean"       : int(preds_mean[i].item()),
                "correct"         : int((preds_mean[i] == all_labels[i]).item()),
                "entropy"         : float(pred_entropy[i].item()),
                "aleatoric"       : float(aleatoric[i].item()),
                "epistemic"       : float(epistemic[i].item()),
            }
            if self.num_classes == 2:
                row["prob_fraud_mean"] = float(probs_mean[i, 1].item())
                row["prob_fraud_std"]  = float(probs_std[i, 1].item())

            results.append(row)

        if self.verbose:
            print(
                f"MC-DROPOUT (n={n_samples}) | "
                f"{format_metrics(metrics, prefix='MC-Test')}"
            )

        # Restore eval mode
        self.model.eval()

        return metrics, results

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    @staticmethod
    def report(metrics: Dict[str, float]) -> str:
        """
        Human-readable full evaluation report.
        """
        sep = "=" * 52
        lines = [
            sep,
            "  Evaluation Report",
            sep,
            f"  Loss             : {metrics.get('loss', float('nan')):.4f}",
            f"  Accuracy         : {metrics.get('accuracy', float('nan')):.4f}",
            "",
            "  ── Binary (fraud=1) ──────────────────",
            f"  Precision        : {metrics.get('precision', float('nan')):.4f}",
            f"  Recall           : {metrics.get('recall', float('nan')):.4f}",
            f"  F1               : {metrics.get('f1', float('nan')):.4f}",
            f"  Specificity      : {metrics.get('specificity', float('nan')):.4f}",
            f"  AUROC            : {metrics.get('auroc', float('nan')):.4f}",
            f"  AUPRC            : {metrics.get('auprc', float('nan')):.4f}",
            "",
            "  ── Confusion Matrix ──────────────────",
            f"  TP={int(metrics.get('tp',0))}  FP={int(metrics.get('fp',0))}  "
            f"FN={int(metrics.get('fn',0))}  TN={int(metrics.get('tn',0))}",
            "",
            "  ── Threshold ─────────────────────────",
            f"  Best Threshold   : {metrics.get('best_threshold', 0.5):.4f}",
            f"  F1 at Best Thr   : {metrics.get('best_f1_at_threshold', float('nan')):.4f}",
            sep,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _to_device(self, batch: Dict) -> Dict:
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            elif hasattr(v, "to"):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result

    def _enable_dropout(self):
        """
        Set model to train mode only for Dropout layers.
        Leaves BatchNorm in eval mode.
        """
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()


# ======================================================================
# Uncertainty helpers
# ======================================================================

def _entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Shannon entropy along the class dimension.

    Args:
        probs: [..., C]

    Returns:
        entropy: [...] (all dims except last)
    """
    return -(probs * torch.log(probs + eps)).sum(dim=-1)
