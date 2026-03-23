from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .global_graph_loader import GlobalGraphData, GlobalGraphLoader
from .label_loader import LabelLoader, LabelRecord
from .transaction_loader import TransactionGraph, TransactionLoader

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _noneify(v: Any) -> Any:
    if isinstance(v, str) and v.strip().lower() in {"none", "null", ""}:
        return None
    return v


def _cfg_to_dict(cfg: Any) -> dict:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    if hasattr(cfg, "items"):
        try:
            return dict(cfg.items())
        except Exception:
            pass
    if hasattr(cfg, "__dict__"):
        return {
            k: v for k, v in vars(cfg).items()
            if not k.startswith("_")
        }
    return dict(cfg)


def _cfg_get(cfg: dict, *keys: str, default=None):
    for k in keys:
        if k in cfg:
            return _noneify(cfg[k])
    return default


def _resolve_path(
    path_value: Optional[str],
    *,
    base_dir: Path,
    must_exist: bool,
    label: str,
) -> Optional[Path]:
    if path_value is None:
        return None

    p = Path(path_value)
    if not p.is_absolute():
        p = (base_dir / p).resolve()
    else:
        p = p.resolve()

    if must_exist and not p.exists():
        raise FileNotFoundError(
            f"[FraudDataset] {label} path does not exist: {p}"
        )
    return p


def _looks_like_cache_root(path: Path, chain: Optional[str]) -> bool:
    """
    Detect whether a path points to the chunk-cache root rather than raw transactions.
    Examples:
      .../.cache/graphs
      .../.cache/graphs/polygon
    """
    p = path.resolve()

    candidates = [p]
    if chain:
        candidates.append(p / chain)

    for c in candidates:
        if (c / "_index.pkl").exists():
            return True
        if any(c.glob("chunk_*.pkl")):
            return True

    if p.name == "graphs" and p.parent.name == ".cache":
        return True

    return False


def _guess_transactions_root_from_cache_root(cache_root: Path) -> Optional[Path]:
    """
    If cache_root is something like:
      /.../dataset/.cache/graphs
    guess:
      /.../dataset/transactions
    """
    p = cache_root.resolve()
    candidate = p.parent.parent / "transactions"
    return candidate.resolve() if candidate.exists() else None


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

@dataclass
class DatasetConfig:
    transactions_root: str = "../_data/dataset/transactions"
    labels_path: str = "../_data/dataset/labels.csv"
    global_graph_root: str = "../_data/dataset/global_graph"

    # optional explicit graph cache root
    cache_root: Optional[str] = None

    chain: str = "polygon"
    auto_split: bool = True
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    split_seed: int = 42
    normalize_address: bool = True

    label_chain_col: Optional[str] = "Chain"
    label_address_col: Optional[str] = "Contract"
    label_label_col: Optional[str] = "Category"
    label_split_col: Optional[str] = None

    normal_categories: List[int] = field(default_factory=lambda: [0])
    fraud_categories: Optional[List[int]] = None

    load_global_graph: bool = False
    chunk_size: int = 100


# ---------------------------------------------------------------------------
# dataset facade
# ---------------------------------------------------------------------------

class FraudDataset:
    """
    transactions / labels / global_graph 세 소스를 한 번에 로드하고
    contract_id 기준으로 정렬·join한 뒤 split-aware 접근을 제공한다.

    Important:
      - graph loading is delegated to TransactionLoader
      - this supports chunked pickle cache format
    """

    def __init__(self, cfg: DatasetConfig) -> None:
        self.cfg = cfg

        self._tx_loader = TransactionLoader(
            root=self.cfg.transactions_root,
            chain=self.cfg.chain,
            cache_root=self.cfg.cache_root,
            chunk_size=self.cfg.chunk_size,
        )

        self._lbl_loader = LabelLoader(
            path=self.cfg.labels_path,
            chain=self.cfg.chain,
            chain_col=self.cfg.label_chain_col,
            address_col=self.cfg.label_address_col,
            label_col=self.cfg.label_label_col,
            split_col=self.cfg.label_split_col,
            normalize_address=self.cfg.normalize_address,
            normal_categories=self.cfg.normal_categories,
            fraud_categories=self.cfg.fraud_categories,
        )

        self._gg_loader = GlobalGraphLoader(
            root=self.cfg.global_graph_root,
            normalize_address=self.cfg.normalize_address,
            chain=self.cfg.chain,
        )

        # loaded state
        self.transaction_graphs: List[TransactionGraph] = []
        self.labels: Dict[str, int] = {}
        self.splits: Dict[str, str] = {}
        self.global_graph: Optional[GlobalGraphData] = None

        # compatibility attrs expected by benchmark / scripts
        self.train_graphs: List[TransactionGraph] = []
        self.valid_graphs: List[TransactionGraph] = []
        self.test_graphs: List[TransactionGraph] = []

        self._loaded = False

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        config_dir: Optional[str] = None,
        auto_load: bool = True,
    ) -> "FraudDataset":
        """
        Supports:
          - DatasetConfig
          - dict
          - OmegaConf DictConfig
          - full benchmark config containing "dataset"
        """
        if isinstance(cfg, DatasetConfig):
            ds = cls(cfg)
            return ds.load() if auto_load else ds

        raw = _cfg_to_dict(cfg)
        if "dataset" in raw and isinstance(raw["dataset"], dict):
            raw = _cfg_to_dict(raw["dataset"])

        base_dir = Path(config_dir).resolve() if config_dir else Path.cwd().resolve()

        log.info("[FraudDataset.from_config] base dir for paths: %s", base_dir)
        log.info("[FraudDataset.from_config] config keys: %s", list(raw.keys()))

        chain = _cfg_get(raw, "chain", "blockchain", "network", default="polygon")

        transactions_root = _resolve_path(
            _cfg_get(raw, "transactions_root", "tx_root", "data_root", "root"),
            base_dir=base_dir,
            must_exist=True,
            label="transactions_root",
        )

        labels_path = _resolve_path(
            _cfg_get(raw, "labels_path", "label_path", "labels_file", "label_file"),
            base_dir=base_dir,
            must_exist=True,
            label="labels_path",
        )

        global_graph_root = _resolve_path(
            _cfg_get(raw, "global_graph_root", "global_root", "global_path"),
            base_dir=base_dir,
            must_exist=False,
            label="global_graph_root",
        )

        cache_root = _resolve_path(
            _cfg_get(raw, "cache_root", "graph_cache_root", "graphs_cache_root"),
            base_dir=base_dir,
            must_exist=False,
            label="cache_root",
        )

        # auto-detect cache-root misuse:
        # if transactions_root itself points to .cache/graphs, reinterpret it.
        if transactions_root is not None and cache_root is None and _looks_like_cache_root(transactions_root, chain):
            detected_cache_root = transactions_root
            guessed_transactions_root = _guess_transactions_root_from_cache_root(detected_cache_root)

            log.info(
                "[FraudDataset.from_config] transactions_root appears to be a cache root. "
                "Using it as cache_root: %s",
                detected_cache_root,
            )

            cache_root = detected_cache_root
            if guessed_transactions_root is not None:
                log.info(
                    "[FraudDataset.from_config] guessed transactions_root from cache_root: %s",
                    guessed_transactions_root,
                )
                transactions_root = guessed_transactions_root

        cfg_obj = DatasetConfig(
            transactions_root=str(transactions_root),
            labels_path=str(labels_path),
            global_graph_root=str(global_graph_root) if global_graph_root is not None else "",
            cache_root=str(cache_root) if cache_root is not None else None,
            chain=chain,
            auto_split=bool(_cfg_get(raw, "auto_split", default=True)),
            train_ratio=float(_cfg_get(raw, "train_ratio", default=0.7)),
            val_ratio=float(_cfg_get(raw, "val_ratio", "valid_ratio", default=0.15)),
            split_seed=int(_cfg_get(raw, "split_seed", "seed", default=42)),
            normalize_address=bool(_cfg_get(raw, "normalize_address", default=True)),
            label_chain_col=_cfg_get(raw, "label_chain_col", "chain_col", default="Chain"),
            label_address_col=_cfg_get(raw, "label_address_col", "address_col", default="Contract"),
            label_label_col=_cfg_get(raw, "label_label_col", "label_col", default="Category"),
            label_split_col=_cfg_get(raw, "label_split_col", "split_col", default=None),
            normal_categories=_cfg_get(raw, "normal_categories", default=[0]),
            fraud_categories=_cfg_get(raw, "fraud_categories", default=None),
            load_global_graph=bool(_cfg_get(raw, "load_global_graph", default=False)),
            chunk_size=int(_cfg_get(raw, "chunk_size", default=100)),
        )

        log.info(
            "[FraudDataset.from_config] "
            "transactions_root=%s | cache_root=%s | labels_path=%s | chain=%s | "
            "auto_split=%s | train_ratio=%.2f | val_ratio=%.2f | seed=%d",
            cfg_obj.transactions_root,
            cfg_obj.cache_root,
            cfg_obj.labels_path,
            cfg_obj.chain,
            cfg_obj.auto_split,
            cfg_obj.train_ratio,
            cfg_obj.val_ratio,
            cfg_obj.split_seed,
        )

        ds = cls(cfg_obj)
        return ds.load() if auto_load else ds

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def load(self) -> "FraudDataset":
        if self._loaded:
            return self

        log.info("[FraudDataset] Loading transaction graphs via TransactionLoader …")
        self.transaction_graphs = self._tx_loader.load_all()

        if not self.transaction_graphs:
            raise RuntimeError(
                "[FraudDataset] TransactionLoader returned 0 graphs. "
                f"transactions_root={self.cfg.transactions_root}, "
                f"cache_root={self.cfg.cache_root}, chain={self.cfg.chain}"
            )

        log.info("[FraudDataset] Loading labels via LabelLoader …")
        records = self._lbl_loader.load()

        if not records:
            raise RuntimeError(
                "[FraudDataset] LabelLoader returned 0 label records. "
                f"labels_path={self.cfg.labels_path}, chain={self.cfg.chain}"
            )

        # keep only intersection between loaded graphs and labels
        graph_ids = {g.contract_id for g in self.transaction_graphs}
        records = [r for r in records if r.contract_id in graph_ids]

        self.labels = {r.contract_id: r.label for r in records}

        if not self.labels:
            sample_ids = sorted(list(graph_ids))[:10]
            raise RuntimeError(
                "[FraudDataset] No overlap between loaded graph contract_ids and label contract_ids. "
                f"sample graph ids={sample_ids}"
            )

        # drop unlabeled graphs
        self.transaction_graphs = [
            g for g in self.transaction_graphs
            if g.contract_id in self.labels
        ]

        has_split = any(r.split is not None for r in records)
        if has_split:
            self.splits = {
                r.contract_id: r.split
                for r in records
                if r.split is not None and r.contract_id in self.labels
            }
        elif self.cfg.auto_split:
            self.splits = self._auto_split(records)
        else:
            self.splits = {cid: "train" for cid in self.labels.keys()}

        if self.cfg.load_global_graph:
            log.info("[FraudDataset] Loading global graph …")
            self.global_graph = self._gg_loader.load()
        else:
            log.info("[FraudDataset] Skipping global graph load.")
            self.global_graph = None

        # mark loaded BEFORE building split views
        self._loaded = True

        self._refresh_split_views()

        log.info(
            "[FraudDataset] DONE | tx_graphs=%d | labels=%d | train=%d | valid=%d | test=%d | global_graph=%s",
            len(self.transaction_graphs),
            len(self.labels),
            len(self.train_graphs),
            len(self.valid_graphs),
            len(self.test_graphs),
            self.global_graph is not None,
        )


        return self

    def split_ids(self, split: str) -> List[str]:
        self._ensure_loaded()
        return [cid for cid, s in self.splits.items() if s == split]

    def split_graphs(self, split: str) -> List[TransactionGraph]:
        self._ensure_loaded()
        ids = set(self.split_ids(split))
        return [g for g in self.transaction_graphs if g.contract_id in ids]

    def get_labels_tensor(
        self,
        contract_ids: List[str],
    ) -> Tuple[torch.Tensor, List[str]]:
        self._ensure_loaded()
        valid_ids = [cid for cid in contract_ids if cid in self.labels]
        labels = torch.tensor(
            [self.labels[cid] for cid in valid_ids],
            dtype=torch.float,
        )
        return labels, valid_ids

    def fraud_rate(self, split: Optional[str] = None) -> float:
        self._ensure_loaded()
        if split:
            ids = self.split_ids(split)
            lbls = [self.labels[i] for i in ids if i in self.labels]
        else:
            lbls = list(self.labels.values())
        return sum(lbls) / max(len(lbls), 1)

    def summary(self) -> str:
        self._ensure_loaded()
        lines = [
            "=" * 50,
            "FraudDataset Summary",
            f"  Chain         : {self.cfg.chain}",
            f"  TX Graphs     : {len(self.transaction_graphs)}",
            f"  Labels        : {len(self.labels)} (fraud_rate={self.fraud_rate():.2%})",
            f"  Global Graph  : nodes={self.global_graph.num_nodes() if self.global_graph else 0}, "
            f"edges={self.global_graph.num_edges() if self.global_graph else 0}",
        ]
        for s in ("train", "valid", "test"):
            n = len(self.split_ids(s))
            fr = self.fraud_rate(s)
            lines.append(f"  {s:>5} split : {n} samples, fraud_rate={fr:.2%}")
        lines.append("=" * 50)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.load()

    def _refresh_split_views(self) -> None:
        if not self.splits:
            self.train_graphs = []
            self.valid_graphs = []
            self.test_graphs = []
            return

        train_ids = {cid for cid, s in self.splits.items() if s == "train"}
        valid_ids = {cid for cid, s in self.splits.items() if s == "valid"}
        test_ids  = {cid for cid, s in self.splits.items() if s == "test"}

        self.train_graphs = [g for g in self.transaction_graphs if g.contract_id in train_ids]
        self.valid_graphs = [g for g in self.transaction_graphs if g.contract_id in valid_ids]
        self.test_graphs  = [g for g in self.transaction_graphs if g.contract_id in test_ids]


    def _auto_split(self, records: List[LabelRecord]) -> Dict[str, str]:
        rng = random.Random(self.cfg.split_seed)

        fraud_ids = [r.contract_id for r in records if r.label == 1]
        normal_ids = [r.contract_id for r in records if r.label == 0]

        splits: Dict[str, str] = {}

        for ids in (fraud_ids, normal_ids):
            ids = ids[:]
            rng.shuffle(ids)
            n = len(ids)
            n_train = int(n * self.cfg.train_ratio)
            n_val = int(n * self.cfg.val_ratio)

            for cid in ids[:n_train]:
                splits[cid] = "train"
            for cid in ids[n_train:n_train + n_val]:
                splits[cid] = "valid"
            for cid in ids[n_train + n_val:]:
                splits[cid] = "test"

        return splits
