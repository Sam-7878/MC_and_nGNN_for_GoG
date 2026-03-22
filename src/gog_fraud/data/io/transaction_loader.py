# src/gog_fraud/data/io/transaction_loader.py

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import Data

from gog_fraud.data.io.cache_store import GraphCacheStore

log = logging.getLogger(__name__)


@dataclass
class TransactionGraph:
    contract_id: str
    graph: Data
    chain: str
    source_path: Optional[str] = None
    meta: Dict = field(default_factory=dict)


class TransactionLoader:
    """
    Graph cache (pickle) 우선 로드.
    cache가 없으면 RuntimeError 를 발생시켜 build_graph_cache.py 실행을 유도.
    """

    def __init__(
        self,
        root: str,
        chain: str,
        cache_root: Optional[str] = None,
        supported_extensions: Tuple[str, ...] = (".csv",),
        verbose: bool = True,
    ) -> None:
        self.root = Path(root)
        self.chain = chain
        self.chain_dir = self.root / chain
        self.exts = supported_extensions
        self.verbose = verbose

        # cache_root가 없으면 root 옆에 cache/ 폴더 사용
        resolved_cache = cache_root or str(self.root.parent / "cache" / "graphs")
        self.store = GraphCacheStore(cache_root=resolved_cache, chain=chain)

    def load_all(self) -> List[TransactionGraph]:
        if self.store.exists():
            log.info(f"[TransactionLoader] Loading from pickle cache: {self.store.cache_dir}")
            return self._load_from_cache()
        else:
            raise RuntimeError(
                f"Graph cache not found for chain={self.chain}.\n"
                f"Please run the preprocessing script first:\n\n"
                f"  PYTHONPATH=./src python -m gog_fraud.data.scripts.build_graph_cache \\\n"
                f"    --transactions_root {self.root} \\\n"
                f"    --cache_root {self.store.cache_dir.parent} \\\n"
                f"    --chain {self.chain}\n"
            )

    def _load_from_cache(self) -> List[TransactionGraph]:
        cached = self.store.load_all()
        results: List[TransactionGraph] = []

        for ctg in cached:
            try:
                self._assert_clean(ctg.graph, ctg.contract_id)
                results.append(
                    TransactionGraph(
                        contract_id=ctg.contract_id,
                        graph=ctg.graph,
                        chain=ctg.chain,
                        source_path=ctg.meta.get("source_csv"),
                        meta=ctg.meta,
                    )
                )
            except Exception as exc:
                log.warning(f"[TransactionLoader] Skip {ctg.contract_id} (cache): {exc}")

        log.info(
            f"[TransactionLoader] Loaded {len(results)} graphs from cache "
            f"for chain={self.chain}"
        )
        return results

    @staticmethod
    def _assert_clean(graph: Data, contract_id: str) -> None:
        if getattr(graph, "x", None) is None:
            raise ValueError("graph.x is missing")
        if torch.isnan(graph.x).any() or torch.isinf(graph.x).any():
            raise ValueError("graph.x contains NaN/Inf")
        if getattr(graph, "edge_attr", None) is not None:
            if torch.isnan(graph.edge_attr).any() or torch.isinf(graph.edge_attr).any():
                raise ValueError("edge_attr contains NaN/Inf")
