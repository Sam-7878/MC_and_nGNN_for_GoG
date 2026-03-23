from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch.amp import GradScaler, autocast



import copy
import inspect
import logging

log = logging.getLogger(__name__)


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _call_compatible(fn, **kwargs):
    sig = inspect.signature(fn)
    params = sig.parameters

    has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if has_var_kw:
        return fn(**kwargs)

    filtered = {k: v for k, v in kwargs.items() if k in params}
    return fn(**filtered)


def _extract_monitor_value(eval_out):
    if eval_out is None:
        return None, None, None

    if isinstance(eval_out, (int, float)):
        return float(eval_out), "min", "loss"

    if isinstance(eval_out, dict):
        for key in ("loss", "val_loss", "avg_loss", "mean_loss"):
            if key in eval_out and eval_out[key] is not None:
                return float(eval_out[key]), "min", key

        for key in ("f1", "macro_f1", "auc", "roc_auc", "pr_auc", "ap", "accuracy", "acc"):
            if key in eval_out and eval_out[key] is not None:
                return float(eval_out[key]), "max", key

    return None, None, None


@dataclass
class Level2TrainerConfig:
    lr:               float = 3e-4
    weight_decay:     float = 1e-4
    epochs:           int   = 10
    batch_size:       int   = 8
    grad_accum_steps: int   = 1
    max_grad_norm:    Optional[float] = 1.0
    use_amp:          bool  = True
    pos_weight:       Optional[float] = None


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


def compute_level2_metrics(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true  = y_true.view(-1).detach().cpu().float()
    y_score = y_score.view(-1).detach().cpu().float()
    y_pred  = (y_score >= threshold).long()

    tp = int(((y_pred == 1) & (y_true.long() == 1)).sum().item())
    fp = int(((y_pred == 1) & (y_true.long() == 0)).sum().item())
    fn = int(((y_pred == 0) & (y_true.long() == 1)).sum().item())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "roc_auc":   _safe_roc_auc(y_true, y_score),
        "pr_auc":    _safe_pr_auc(y_true, y_score),
    }


class Level2Trainer:
    def __init__(
        self,
        model,
        optimizer,
        cfg: Level2TrainerConfig,
        device: Optional[str] = None,
    ):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model     = model.to(self.device)
        self.optimizer = optimizer
        self.cfg       = cfg

        self.use_amp = bool(cfg.use_amp and self.device.startswith("cuda"))
        self.scaler  = GradScaler("cuda", enabled=self.use_amp)

        pos_weight = None
        if cfg.pos_weight is not None:
            pos_weight = torch.tensor(
                [cfg.pos_weight], dtype=torch.float32, device=self.device
            )
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _move_batch(self, batch):
        return batch.to(self.device)

    # ── Train ──────────────────────────────────────────────────────────────

    def train_one_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        all_y:     list = []
        all_score: list = []

        for step_idx, batch in enumerate(loader, start=1):
            batch = self._move_batch(batch)

            with autocast("cuda", enabled=self.use_amp):
                out = self.model(batch)
                if out.label is None:
                    raise ValueError(
                        "Training batch must contain graph-level labels in batch.y"
                    )
                loss = self.loss_fn(out.logits, out.label)
                scaled = loss / self.cfg.grad_accum_steps

            self.scaler.scale(scaled).backward()

            if self.cfg.max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.max_grad_norm
                )

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

        all_y     = torch.cat(all_y,     dim=0)
        all_score = torch.cat(all_score, dim=0)

        metrics = compute_level2_metrics(all_y, all_score)
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics

    # ── Eval ───────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        all_y:     list = []
        all_score: list = []

        for batch in loader:
            batch = self._move_batch(batch)
            out   = self.model(batch)

            if out.label is None:
                raise ValueError(
                    "Evaluation batch must contain graph-level labels in batch.y"
                )

            loss = self.loss_fn(out.logits, out.label)
            total_loss += float(loss.item())

            all_y.append(out.label.detach().cpu())
            all_score.append(out.score.detach().cpu())

        all_y     = torch.cat(all_y,     dim=0)
        all_score = torch.cat(all_score, dim=0)

        metrics = compute_level2_metrics(all_y, all_score)
        metrics["loss"] = total_loss / max(len(loader), 1)
        return metrics

    def fit(self, l1_model, global_graph, train_ids, valid_ids=None, label_dict=None, **kwargs):
        train_ids = train_ids or []
        valid_ids = valid_ids or []

        epochs = int(_cfg_get(self.cfg, "epochs", _cfg_get(self.cfg, "max_epochs", 10)))
        eval_every = int(_cfg_get(self.cfg, "eval_every", 1))
        patience = _cfg_get(self.cfg, "patience", None)
        load_best_at_end = bool(_cfg_get(self.cfg, "load_best_at_end", True))

        history = []
        best_score = None
        best_mode = None
        best_metric = None
        best_state = None
        no_improve = 0

        for epoch in range(1, epochs + 1):
            train_out = None
            last_exc = None

            train_variants = [
                dict(
                    l1_model=l1_model,
                    global_graph=global_graph,
                    train_ids=train_ids,
                    label_dict=label_dict,
                    epoch=epoch,
                    **kwargs,
                ),
                dict(
                    level1_model=l1_model,
                    global_graph=global_graph,
                    train_ids=train_ids,
                    label_dict=label_dict,
                    epoch=epoch,
                    **kwargs,
                ),
                dict(
                    l1_model=l1_model,
                    graph=global_graph,
                    train_ids=train_ids,
                    label_dict=label_dict,
                    epoch=epoch,
                    **kwargs,
                ),
                dict(
                    l1_model=l1_model,
                    global_graph=global_graph,
                    ids=train_ids,
                    label_dict=label_dict,
                    epoch=epoch,
                    **kwargs,
                ),
                dict(
                    l1_model=l1_model,
                    global_graph=global_graph,
                    contract_ids=train_ids,
                    label_dict=label_dict,
                    epoch=epoch,
                    **kwargs,
                ),
            ]

            for var in train_variants:
                try:
                    train_out = _call_compatible(self.train_one_epoch, **var)
                    break
                except TypeError as exc:
                    last_exc = exc

            if train_out is None and last_exc is not None:
                raise last_exc

            row = {"epoch": epoch, "train": train_out}

            if valid_ids and (epoch % eval_every == 0):
                valid_out = None
                last_exc = None

                eval_variants = [
                    dict(
                        l1_model=l1_model,
                        global_graph=global_graph,
                        valid_ids=valid_ids,
                        label_dict=label_dict,
                        epoch=epoch,
                        **kwargs,
                    ),
                    dict(
                        level1_model=l1_model,
                        global_graph=global_graph,
                        valid_ids=valid_ids,
                        label_dict=label_dict,
                        epoch=epoch,
                        **kwargs,
                    ),
                    dict(
                        l1_model=l1_model,
                        graph=global_graph,
                        valid_ids=valid_ids,
                        label_dict=label_dict,
                        epoch=epoch,
                        **kwargs,
                    ),
                    dict(
                        l1_model=l1_model,
                        global_graph=global_graph,
                        ids=valid_ids,
                        label_dict=label_dict,
                        epoch=epoch,
                        **kwargs,
                    ),
                    dict(
                        l1_model=l1_model,
                        global_graph=global_graph,
                        contract_ids=valid_ids,
                        label_dict=label_dict,
                        epoch=epoch,
                        **kwargs,
                    ),
                ]

                for var in eval_variants:
                    try:
                        valid_out = _call_compatible(self.evaluate, **var)
                        break
                    except TypeError as exc:
                        last_exc = exc

                if valid_out is None and last_exc is not None:
                    raise last_exc

                row["valid"] = valid_out

                score, mode, metric = _extract_monitor_value(valid_out)
                if score is not None:
                    improved = (
                        best_score is None
                        or (mode == "min" and score < best_score)
                        or (mode == "max" and score > best_score)
                    )

                    if improved:
                        best_score = score
                        best_mode = mode
                        best_metric = metric
                        best_state = copy.deepcopy(self.model.state_dict())
                        no_improve = 0
                    else:
                        no_improve += 1

            history.append(row)

            if patience is not None and valid_ids:
                if no_improve >= int(patience):
                    log.info(
                        "[Level2Trainer] Early stopping at epoch %d (metric=%s, best=%s)",
                        epoch,
                        best_metric,
                        best_score,
                    )
                    break

        if best_state is not None and load_best_at_end:
            self.model.load_state_dict(best_state)

        self.history = history
        self.best_score = best_score
        self.best_metric = best_metric
        self.best_mode = best_mode

        return {
            "history": history,
            "best_score": best_score,
            "best_metric": best_metric,
            "best_mode": best_mode,
            "epochs_ran": len(history),
        }

