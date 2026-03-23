from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Dict, Optional

import copy
import inspect
import logging

import torch
from torch.amp import GradScaler, autocast

from typing import Any, Iterable, List, Optional

from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

try:
    from gog_fraud.data.io.transaction_loader import TransactionGraph
except Exception:
    TransactionGraph = None



log = logging.getLogger(__name__)


def _is_loader(obj: Any) -> bool:
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__") and not isinstance(obj, (list, tuple))


def _unwrap_graph_items(items: Any) -> List[Any]:
    if items is None:
        return []

    if isinstance(items, (list, tuple)):
        seq = list(items)
    else:
        try:
            seq = list(items)
        except Exception:
            return []

    out = []
    for x in seq:
        # TransactionGraph wrapper
        if hasattr(x, "graph"):
            out.append(x.graph)
        else:
            out.append(x)
    return out


def _make_pyg_loader(
    items: Any,
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
):
    data_list = _unwrap_graph_items(items)
    if not data_list:
        return None
    return PyGDataLoader(
        data_list,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )



def _cfg_to_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if is_dataclass(cfg):
        return asdict(cfg)
    if hasattr(cfg, "items"):
        try:
            return dict(cfg.items())
        except Exception:
            pass
    if hasattr(cfg, "__dict__"):
        return {k: v for k, v in vars(cfg).items() if not k.startswith("_")}
    return {}


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        if isinstance(value, dict) and not isinstance(value, AttrDict):
            value = AttrDict(value)
            self[key] = value
        return value

    def __setattr__(self, key, value):
        self[key] = value


def _cfg_norm(cfg: Any) -> AttrDict:
    data = _cfg_to_dict(cfg)

    def _convert(v):
        if isinstance(v, dict):
            return AttrDict({kk: _convert(vv) for kk, vv in v.items()})
        if isinstance(v, list):
            return [_convert(x) for x in v]
        return v

    return AttrDict({k: _convert(v) for k, v in data.items()})


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


def compute_binary_metrics(
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


def _empty_binary_metrics() -> Dict[str, float]:
    return {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "roc_auc": 0.0,
        "pr_auc": 0.0,
        "loss": 0.0,
    }


def _safe_len(obj) -> Optional[int]:
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


def _prepare_level1_loader(
    data_or_loader,
    *,
    split_name: str,
    loader_builder=None,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
):
    # 1) already a loader
    if _is_loader(data_or_loader):
        return data_or_loader

    # 2) try explicit builder first
    if loader_builder is not None:
        builder_calls = [
            lambda: loader_builder(
                data_or_loader,
                split=split_name,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            ),
            lambda: loader_builder(
                data_or_loader,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
            ),
            lambda: loader_builder(data_or_loader, split_name),
            lambda: loader_builder(data_or_loader),
        ]

        last_err = None
        for fn in builder_calls:
            try:
                built = fn()
                if built is not None:
                    return built
            except TypeError as e:
                last_err = e
            except Exception as e:
                last_err = e

    # 3) automatic fallback: list of TransactionGraph or Data
    auto_loader = _make_pyg_loader(
        data_or_loader,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    if auto_loader is not None:
        return auto_loader

    raise TypeError(
        f"Unable to resolve {split_name} loader. "
        f"type={type(data_or_loader).__name__}, "
        f"loader_builder={type(loader_builder).__name__ if loader_builder is not None else None}. "
        "Pass `train_loader` directly or provide a compatible `loader_builder`."
    )



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


class Level1Trainer:
    def __init__(self, model, optimizer, cfg):
        self.model = model
        self.optimizer = optimizer
        self.cfg = _cfg_norm(cfg)

        self.device = str(_cfg_get(self.cfg, "device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.model.to(self.device)

        self.use_amp = bool(_cfg_get(self.cfg, "use_amp", False) and self.device.startswith("cuda"))
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

        pos_weight_value = _cfg_get(cfg, "pos_weight", None)
        self.pos_weight = None
        if pos_weight_value is not None:
            self.pos_weight = torch.tensor(
                [float(pos_weight_value)],
                device=self.device,
                dtype=torch.float32,
            )

        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

    def _move_batch(self, batch):
        return batch.to(self.device)

    def train_one_epoch(self, loader) -> Dict[str, float]:
        if loader is None or (_safe_len(loader) == 0):
            log.warning("[Level1Trainer] Empty train loader. Returning zero metrics.")
            return _empty_binary_metrics()

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
                    raise ValueError("Training batch must contain graph-level labels in batch.y")

                label = out.label.float().view_as(out.logits)
                loss = self.loss_fn(out.logits, label)
                scaled_loss = loss / grad_accum_steps

            self.scaler.scale(scaled_loss).backward()
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
            return _empty_binary_metrics()

        all_y = torch.cat(all_y, dim=0)
        all_score = torch.cat(all_score, dim=0)

        metrics = compute_binary_metrics(all_y, all_score)
        metrics["loss"] = total_loss / max(num_batches, 1)
        return metrics

    @torch.no_grad()
    def evaluate(self, loader) -> Dict[str, float]:
        if loader is None or (_safe_len(loader) == 0):
            log.warning("[Level1Trainer] Empty valid loader. Returning zero metrics.")
            return _empty_binary_metrics()

        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        all_y = []
        all_score = []

        for batch in loader:
            batch = self._move_batch(batch)
            out = self.model(batch)

            if out.label is None:
                raise ValueError("Evaluation batch must contain graph-level labels in batch.y")

            label = out.label.float().view_as(out.logits)
            loss = self.loss_fn(out.logits, label)

            total_loss += float(loss.item())
            num_batches += 1
            all_y.append(label.detach().cpu())
            all_score.append(out.score.detach().cpu())

        if not all_y:
            return _empty_binary_metrics()

        all_y = torch.cat(all_y, dim=0)
        all_score = torch.cat(all_score, dim=0)

        metrics = compute_binary_metrics(all_y, all_score)
        metrics["loss"] = total_loss / max(num_batches, 1)
        return metrics

    def fit(
        self,
        train_graphs=None,
        valid_graphs=None,
        label_dict=None,
        train_loader=None,
        valid_loader=None,
        loader_builder=None,
        **kwargs,
    ):
        train_source = train_loader
        if train_source is None:
            for key in ("loader", "dataloader", "train_dataloader", "train_loader", "graphs", "train_data", "data"):
                if key in kwargs and kwargs[key] is not None:
                    train_source = kwargs[key]
                    break
        if train_source is None:
            train_source = train_graphs

        valid_source = valid_loader
        if valid_source is None:
            for key in ("valid_loader", "eval_loader", "val_loader", "valid_dataloader", "eval_graphs", "valid_data"):
                if key in kwargs and kwargs[key] is not None:
                    valid_source = kwargs[key]
                    break
        if valid_source is None:
            valid_source = valid_graphs

        train_loader = _prepare_level1_loader(
            train_source,
            split="train",
            batch_size=int(_cfg_get(self.cfg, "batch_size", 16)),
            shuffle=True,
            label_dict=label_dict,
            loader_builder=loader_builder,
            **kwargs,
        )

        valid_loader = _prepare_level1_loader(
            valid_source,
            split="valid",
            batch_size=int(_cfg_get(self.cfg, "batch_size", 16)),
            shuffle=False,
            label_dict=label_dict,
            loader_builder=loader_builder,
            **kwargs,
        ) if valid_source is not None else None

        if train_loader is None or (_safe_len(train_loader) == 0):
            log.warning("[Level1Trainer] No training data. Skipping fit().")
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
                        "[Level1Trainer] Early stopping at epoch %d (metric=%s, best=%s)",
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
