# src/gog_fraud/data/io/cache_store.py

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class CachedTransactionGraph:
    contract_id: str
    chain: str
    graph: Any           # torch_geometric.data.Data
    meta: Dict = field(default_factory=dict)


class GraphCacheStore:
    """
    전처리된 PyG graph를 contract 단위로 pickle 캐시에 저장/로드.

    파일 구조:
        {cache_root}/{chain}/
            {contract_id}.pkl     ← 개별 contract graph
            _index.pkl            ← contract_id 목록 + meta summary
    """

    _INDEX_FILE = "_index.pkl"

    def __init__(self, cache_root: str, chain: str) -> None:
        self.cache_dir = Path(cache_root) / chain
        self.chain = chain

    def exists(self) -> bool:
        return (self.cache_dir / self._INDEX_FILE).exists()

    def save_all(
        self,
        graphs: List[CachedTransactionGraph],
        overwrite: bool = False,
    ) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        index: List[dict] = []
        saved = 0
        skipped = 0

        for ctg in graphs:
            pkl_path = self.cache_dir / f"{ctg.contract_id}.pkl"
            if pkl_path.exists() and not overwrite:
                skipped += 1
                index.append({"contract_id": ctg.contract_id, "path": str(pkl_path)})
                continue

            with open(pkl_path, "wb") as f:
                pickle.dump(ctg, f, protocol=pickle.HIGHEST_PROTOCOL)

            index.append({
                "contract_id": ctg.contract_id,
                "path": str(pkl_path),
                "meta": ctg.meta,
            })
            saved += 1

        index_path = self.cache_dir / self._INDEX_FILE
        with open(index_path, "wb") as f:
            pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)

        log.info(
            f"[GraphCacheStore] Saved chain={self.chain}: "
            f"saved={saved}, skipped={skipped}, total={len(index)}"
        )

    def load_all(self) -> List[CachedTransactionGraph]:
        index_path = self.cache_dir / self._INDEX_FILE
        if not index_path.exists():
            raise FileNotFoundError(
                f"Cache index not found: {index_path}. "
                f"Run build_graph_cache.py first."
            )

        with open(index_path, "rb") as f:
            index: List[dict] = pickle.load(f)

        results: List[CachedTransactionGraph] = []
        failed = 0

        for entry in index:
            pkl_path = Path(entry["path"])
            if not pkl_path.exists():
                log.warning(f"[GraphCacheStore] Missing pkl: {pkl_path}")
                failed += 1
                continue
            try:
                with open(pkl_path, "rb") as f:
                    ctg: CachedTransactionGraph = pickle.load(f)
                results.append(ctg)
            except Exception as exc:
                log.warning(f"[GraphCacheStore] Failed to load {pkl_path.name}: {exc}")
                failed += 1

        log.info(
            f"[GraphCacheStore] Loaded chain={self.chain}: "
            f"loaded={len(results)}, failed={failed}"
        )
        return results

    def load_one(self, contract_id: str) -> Optional[CachedTransactionGraph]:
        pkl_path = self.cache_dir / f"{contract_id}.pkl"
        if not pkl_path.exists():
            return None
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
