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
    name: Optional[str] = None


class TransactionLoader:
    """
    Chunk-based pickle cache 우선 로드.

    cache가 없으면 RuntimeError를 발생시켜
    build_graph_cache.py 실행을 유도합니다.
    """

    def __init__(
        self,
        root: str,
        chain: str,
        cache_root: Optional[str] = None,
        chunk_size: int = 100,
        verbose: bool = True,
    ) -> None:
        self.root = Path(root).resolve()
        self.chain = chain
        self.verbose = verbose

        # ---------------------------------------------------------------
        # 경로 결정 순서:
        #   1. 명시적으로 cache_root가 전달된 경우 → 그대로 resolve()
        #   2. None인 경우 → transactions root 옆에 .cache/graphs/ 사용
        # ---------------------------------------------------------------
        if cache_root is not None:
            resolved_cache = Path(cache_root).resolve()
        else:
            resolved_cache = self.root.parent / ".cache" / "graphs"
            resolved_cache = resolved_cache.resolve()

        self.store = GraphCacheStore(
            cache_root=str(resolved_cache),
            chain=chain,
            chunk_size=chunk_size,
        )

        log.debug(
            f"[TransactionLoader] cache_dir resolved → {self.store.cache_dir}"
        )

    def load_all(self) -> List[TransactionGraph]:
        if not self.store.exists():
            # 친절한 에러 메시지 (절대 경로로 표시)
            raise RuntimeError(
                f"\n"
                f"[TransactionLoader] Graph cache not found.\n"
                f"  Expected index : {self.store.cache_dir / '_index.pkl'}\n\n"
                f"Please build the cache first:\n\n"
                f"  PYTHONPATH=./src python -m gog_fraud.data.scripts.build_graph_cache \\\n"
                f"    --transactions_root {self.root} \\\n"
                f"    --cache_root        {self.store.cache_dir.parent} \\\n"
                f"    --chain             {self.chain}\n"
            )

        log.info(
            f"[TransactionLoader] Loading from chunk cache: {self.store.cache_dir}"
        )
        return self._load_from_cache()

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
                log.warning(
                    f"[TransactionLoader] Skip {ctg.contract_id} (cache integrity): {exc}"
                )

        log.info(
            f"[TransactionLoader] Ready: {len(results)} graphs "
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
            if (
                torch.isnan(graph.edge_attr).any()
                or torch.isinf(graph.edge_attr).any()
            ):
                raise ValueError("edge_attr contains NaN/Inf")
