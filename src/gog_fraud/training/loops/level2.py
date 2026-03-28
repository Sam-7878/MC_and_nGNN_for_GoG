from dataclasses import dataclass
from typing import Dict, Optional

import copy
import inspect
import logging

import torch
from torch.amp import GradScaler, autocast

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
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    batch_size: int = 8
    grad_accum_steps: int = 1
    max_grad_norm: Optional[float] = 1.0
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


def compute_level2_metrics(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_true = y_true.view(-1).detach().cpu().float()
    y_score = y_score.view(-1).detach().cpu().float()
    y_pred = (y_score >= threshold).long()

    tp = int(((y_pred == 1) & (y_true.long() == 1)).sum().item())
    fp = int(((y_pred == 1) & (y_true.long() == 0)).sum().item())
    fn = int(((y_pred == 0) & (y_true.long() == 1)).sum().item())

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


def _empty_level2_metrics() -> Dict[str, float]:
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "roc_auc": 0.0,
        "pr_auc": 0.0,
        "loss": 0.0,
    }


def _safe_len(obj):
    try:
        return len(obj)
    except Exception:
        return None


def _is_loader_like(obj) -> bool:
    return (
        obj is not None
        and not isinstance(obj, (list, tuple, dict, str, bytes))
        and hasattr(obj, "__iter__")
        and hasattr(obj, "__len__")
    )


def _looks_like_graph_item(item) -> bool:
    return hasattr(item, "to") or hasattr(item, "x") or hasattr(item, "edge_index")


def _can_auto_build_graph_loader(source) -> bool:
    if source is None:
        return False

    if isinstance(source, (list, tuple)):
        if len(source) == 0:
            return True
        return _looks_like_graph_item(source[0])

    if hasattr(source, "__getitem__") and hasattr(source, "__len__"):
        try:
            n = len(source)
            if n == 0:
                return True
            first = source[0]
            return _looks_like_graph_item(first)
        except Exception:
            return False

    return False


def _build_pyg_loader(data, batch_size: int, shuffle: bool):
    try:
        from torch_geometric.loader import DataLoader as PyGDataLoader
    except Exception as e:
        raise ImportError(
            "Auto-building a graph loader requires torch_geometric.loader.DataLoader."
        ) from e

    return PyGDataLoader(data, batch_size=batch_size, shuffle=shuffle)


def _prepare_level2_loader(
    source,
    *,
    split: str,
    batch_size: int,
    shuffle: bool,
    l1_model=None,
    global_graph=None,
    label_dict=None,
    loader_builder=None,
    **kwargs,
):
    if source is None:
        return None

    if _is_loader_like(source):
        return source

    if loader_builder is not None:
        built = _call_compatible(
            loader_builder,
            split=split,
            ids=source,
            train_ids=source if split == "train" else None,
            valid_ids=source if split != "train" else None,
            contract_ids=source,
            l1_model=l1_model,
            level1_model=l1_model,
            global_graph=global_graph,
            graph=global_graph,
            label_dict=label_dict,
            batch_size=batch_size,
            shuffle=shuffle,
            **kwargs,
        )
        if built is not None:
            return built

    if _can_auto_build_graph_loader(source):
        return _build_pyg_loader(source, batch_size=batch_size, shuffle=shuffle)

    raise TypeError(
        f"Unable to resolve {split} loader for Level2Trainer. "
        f"If `{split}_ids` are plain ids, pass `{split}_loader` directly or provide `loader_builder`."
    )


class Level2Trainer:
    def __init__(
        self,
        model,
        optimizer,
        cfg: "Any",
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.cfg = cfg

        raw_amp = _cfg_get(cfg, "use_amp", True)
        self.use_amp = bool(raw_amp and self.device.startswith("cuda"))
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        raw_pos_weight = _cfg_get(cfg, "pos_weight", None)
        pos_weight = None
        if raw_pos_weight is not None:
            pos_weight = torch.tensor(
                [raw_pos_weight],
                dtype=torch.float32,
                device=self.device,
            )
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def _move_batch(self, batch):
        return batch.to(self.device)

    def train_one_epoch(self, loader) -> Dict[str, float]:
        if loader is None or (_safe_len(loader) == 0):
            log.warning("[Level2Trainer] Empty train loader. Returning zero metrics.")
            return _empty_level2_metrics()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        num_batches = 0
        all_y = []
        all_score = []

        grad_accum_steps = max(int(_cfg_get(self.cfg, "grad_accum_steps", 1)), 1)
        micro_steps = 0

        def _optimizer_step():
            if _cfg_get(self.cfg, "max_grad_norm", None) is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(self.cfg.max_grad_norm),
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

        for batch in loader:
            batch = self._move_batch(batch)

            with autocast("cuda", enabled=self.use_amp):
                out = self.model(batch)
                if out.label is None:
                    raise ValueError(
                        "Training batch must contain graph-level labels in batch.y"
                    )

                label = out.label.float().view_as(out.logits)
                loss = self.loss_fn(out.logits, label)
                scaled = loss / grad_accum_steps

            self.scaler.scale(scaled).backward()
            micro_steps += 1

            if micro_steps >= grad_accum_steps:
                _optimizer_step()
                micro_steps = 0

            total_loss += float(loss.item())
            num_batches += 1
            all_y.append(label.detach().cpu())
            all_score.append(out.score.detach().cpu())

        if micro_steps > 0:
            _optimizer_step()

        if not all_y:
            return _empty_level2_metrics()

        all_y = torch.cat(all_y, dim=0)
        all_score = torch.cat(all_score, dim=0)

        metrics = compute_level2_metrics(all_y, all_score)
        metrics["loss"] = total_loss / max(num_batches, 1)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader=None, l1_model=None, global_graph=None, label_dict=None, loader_builder=None, return_preds=False, **kwargs) -> Dict[str, float]:
        eval_source = loader
        for key in ("eval_loader", "val_loader", "eval_ids", "valid_ids", "test_ids", "test_graphs"):
            if eval_source is None and key in kwargs and kwargs[key] is not None:
                eval_source = kwargs[key]
                break

        _batch_size = int(_cfg_get(self.cfg, "batch_size", 8))

        try:
            loader = _prepare_level2_loader(
                eval_source,
                split="eval",
                batch_size=_batch_size,
                shuffle=False,
                l1_model=l1_model,
                global_graph=global_graph,
                label_dict=label_dict,
                loader_builder=loader_builder,
                **kwargs,
            )
        except Exception as exc:
            log.warning("[Level2Trainer] Could not build valid loader: %s", exc)
            loader = None

        if loader is None or (_safe_len(loader) == 0):
            log.warning("[Level2Trainer] Empty valid loader. Returning zero metrics.")
            if return_preds:
                import numpy as np
                return _empty_level2_metrics(), np.array([]), np.array([])
            return _empty_level2_metrics()

        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        all_y = []
        all_score = []

        for batch in loader:
            batch = self._move_batch(batch)
            out = self.model(batch)

            if out.label is None:
                raise ValueError(
                    "Evaluation batch must contain graph-level labels in batch.y"
                )

            label = out.label.float().view_as(out.logits)
            loss = self.loss_fn(out.logits, label)
            total_loss += float(loss.item())
            num_batches += 1

            all_y.append(label.detach().cpu())
            all_score.append(out.score.detach().cpu())

        if not all_y:
            if return_preds:
                import numpy as np
                return _empty_level2_metrics(), np.array([]), np.array([])
            return _empty_level2_metrics()

        all_y = torch.cat(all_y, dim=0)
        all_score = torch.cat(all_score, dim=0)

        metrics = compute_level2_metrics(all_y, all_score)
        metrics["loss"] = total_loss / max(num_batches, 1)
        if return_preds:
            return metrics, all_y.cpu().numpy(), all_score.cpu().numpy()
        return metrics

    def fit(
        self,
        l1_model=None,
        global_graph=None,
        train_ids=None,
        valid_ids=None,
        label_dict=None,
        train_loader=None,
        valid_loader=None,
        loader_builder=None,
        **kwargs,
    ):
        train_source = train_loader
        if train_source is None:
            for key in ("loader", "dataloader", "train_loader", "train_dataloader"):
                if key in kwargs and kwargs[key] is not None:
                    train_source = kwargs[key]
                    break
        if train_source is None:
            train_source = train_ids

        valid_source = valid_loader
        if valid_source is None:
            for key in ("valid_loader", "eval_loader", "val_loader", "valid_dataloader"):
                if key in kwargs and kwargs[key] is not None:
                    valid_source = kwargs[key]
                    break
        if valid_source is None:
            valid_source = valid_ids

        train_loader = _prepare_level2_loader(
            train_source,
            split="train",
            batch_size=int(_cfg_get(self.cfg, "batch_size", 8)),
            shuffle=True,
            l1_model=l1_model,
            global_graph=global_graph,
            label_dict=label_dict,
            loader_builder=loader_builder,
            **kwargs,
        )

        valid_loader = _prepare_level2_loader(
            valid_source,
            split="valid",
            batch_size=int(_cfg_get(self.cfg, "batch_size", 8)),
            shuffle=False,
            l1_model=l1_model,
            global_graph=global_graph,
            label_dict=label_dict,
            loader_builder=loader_builder,
            **kwargs,
        ) if valid_source is not None else None

        if train_loader is None or (_safe_len(train_loader) == 0):
            log.warning("[Level2Trainer] No training data. Skipping fit().")
            self.history = []
            self.best_score = None
            self.best_metric = None
            self.best_mode = None
            return {
                "history": [],
                "best_score": None,
                "best_metric": None,
                "best_mode": None,
                "epochs_ran": 0,
            }

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
            train_out = self.train_one_epoch(train_loader)
            row = {"epoch": epoch, "train": train_out}

            if valid_loader is not None and (epoch % eval_every == 0):
                valid_out = self.evaluate(valid_loader)
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

            if patience is not None and valid_loader is not None:
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
