from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F
# from torch.cuda.amp import GradScaler, autocast
from torch.amp import GradScaler, autocast

@dataclass
class Level1TrainerConfig:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_grad_norm: Optional[float] = None
    use_amp: bool = True
    pos_weight: Optional[float] = None


def _safe_roc_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true.cpu().numpy(), y_score.cpu().numpy()))
    except Exception:
        return 0.0


def _safe_pr_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(y_true.cpu().numpy(), y_score.cpu().numpy()))
    except Exception:
        return 0.0


def compute_binary_metrics(y_true: torch.Tensor, y_score: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    y_true = y_true.view(-1).detach().cpu()
    y_score = y_score.view(-1).detach().cpu()
    y_pred = (y_score >= threshold).long()

    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "pr_auc": _safe_pr_auc(y_true, y_score),
    }


class Level1Trainer:
    def __init__(self, model, optimizer, cfg: Level1TrainerConfig, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.cfg = cfg

        # self.use_amp = bool(cfg.use_amp and self.device.startswith("cuda"))
        # self.scaler = GradScaler(enabled=self.use_amp)
        self.use_amp = bool(cfg.use_amp and self.device.startswith("cuda"))
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        pos_weight = None
        if cfg.pos_weight is not None:
            pos_weight = torch.tensor([cfg.pos_weight], dtype=torch.float32, device=self.device)

        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _move_batch(self, batch):
        return batch.to(self.device)

    def train_one_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        all_y = []
        all_score = []

        for step_idx, batch in enumerate(loader, start=1):
            batch = self._move_batch(batch)

            # with autocast(enabled=self.use_amp):
            with autocast("cuda", enabled=self.use_amp):
                out = self.model(batch)
                if out.label is None:
                    raise ValueError("Training batch must contain graph-level labels in batch.y")

                loss = self.loss_fn(out.logits, out.label)
                scaled_loss = loss / self.cfg.grad_accum_steps

            self.scaler.scale(scaled_loss).backward()

            if self.cfg.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)

            should_step = (
                step_idx % self.cfg.grad_accum_steps == 0
                or step_idx == len(loader)
            )

            if should_step:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            total_loss += float(loss.item())
            all_y.append(out.label.detach().cpu())
            all_score.append(out.score.detach().cpu())

        all_y = torch.cat(all_y, dim=0)
        all_score = torch.cat(all_score, dim=0)

        metrics = compute_binary_metrics(all_y, all_score)
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        all_y = []
        all_score = []

        for batch in loader:
            batch = self._move_batch(batch)
            out = self.model(batch)

            if out.label is None:
                raise ValueError("Evaluation batch must contain graph-level labels in batch.y")

            loss = self.loss_fn(out.logits, out.label)
            total_loss += float(loss.item())

            all_y.append(out.label.detach().cpu())
            all_score.append(out.score.detach().cpu())

        all_y = torch.cat(all_y, dim=0)
        all_score = torch.cat(all_score, dim=0)

        metrics = compute_binary_metrics(all_y, all_score)
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics
