# src/gog_fraud/data/io/cache_store.py

from __future__ import annotations

import logging
import math
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
    전처리된 PyG graph를 chunk 단위 pickle 파일로 저장/로드.

    파일 구조:
        {cache_root}/{chain}/
            _index.pkl             ← 전체 contract_id 목록 + chunk 매핑
            chunk_000.pkl          ← 0번~99번 contract graphs
            chunk_001.pkl          ← 100번~199번 contract graphs
            ...
            chunk_023.pkl          ← 2300번~2353번 contract graphs

    청크 크기 기본값: 100
    """

    _INDEX_FILE = "_index.pkl"
    _CHUNK_PREFIX = "chunk_"

    def __init__(
        self,
        cache_root: str,
        chain: str,
        chunk_size: int = 100,
    ) -> None:
        self.cache_dir = Path(cache_root).resolve() / chain
        self.chain = chain
        self.chunk_size = chunk_size

    def exists(self) -> bool:
        index_path = self.cache_dir / self._INDEX_FILE
        exists = index_path.exists()
        if not exists:
            log.debug(f"[GraphCacheStore] Index not found at: {index_path}")
        return exists

    # ------------------------------------------------------------------
    # 저장
    # ------------------------------------------------------------------

    def save_all(
        self,
        graphs: List[CachedTransactionGraph],
        overwrite: bool = False,
    ) -> None:
        if not graphs:
            log.warning(f"[GraphCacheStore] Nothing to save for chain={self.chain}")
            return

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 기존 청크 파일 삭제 (overwrite 시)
        if overwrite:
            for old in self.cache_dir.glob(f"{self._CHUNK_PREFIX}*.pkl"):
                old.unlink()
            index_path = self.cache_dir / self._INDEX_FILE
            if index_path.exists():
                index_path.unlink()

        num_chunks = math.ceil(len(graphs) / self.chunk_size)
        log.info(
            f"[GraphCacheStore] Saving {len(graphs)} graphs "
            f"in {num_chunks} chunks (chunk_size={self.chunk_size}) "
            f"→ {self.cache_dir}"
        )

        index: List[Dict] = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, len(graphs))
            chunk_graphs = graphs[start:end]

            chunk_filename = f"{self._CHUNK_PREFIX}{chunk_idx:03d}.pkl"
            chunk_path = self.cache_dir / chunk_filename

            with open(chunk_path, "wb") as f:
                pickle.dump(chunk_graphs, f, protocol=pickle.HIGHEST_PROTOCOL)

            for g in chunk_graphs:
                index.append({
                    "contract_id": g.contract_id,
                    "chunk_file": chunk_filename,
                    "meta": g.meta,
                })

            log.debug(
                f"[GraphCacheStore] Saved chunk {chunk_idx:03d}: "
                f"{len(chunk_graphs)} graphs → {chunk_filename}"
            )

        # Index 저장
        index_path = self.cache_dir / self._INDEX_FILE
        with open(index_path, "wb") as f:
            pickle.dump(
                {
                    "version": 2,
                    "chain": self.chain,
                    "total": len(graphs),
                    "num_chunks": num_chunks,
                    "chunk_size": self.chunk_size,
                    "entries": index,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        log.info(
            f"[GraphCacheStore] Done: "
            f"total={len(graphs)}, chunks={num_chunks}, "
            f"index → {index_path}"
        )

    # ------------------------------------------------------------------
    # 전체 로드
    # ------------------------------------------------------------------

    def load_all(self) -> List[CachedTransactionGraph]:
        index_path = self.cache_dir / self._INDEX_FILE
        if not index_path.exists():
            raise FileNotFoundError(
                f"Cache index not found: {index_path}\n"
                f"Please run build_graph_cache.py first."
            )

        with open(index_path, "rb") as f:
            meta = pickle.load(f)

        # version 1 (구 포맷: entries가 list[dict])인 경우 호환
        if isinstance(meta, list):
            return self._load_legacy_v1(meta)

        num_chunks = int(meta.get("num_chunks", 0))
        results: List[CachedTransactionGraph] = []
        failed_chunks = 0

        log.info(
            f"[GraphCacheStore] Loading chain={self.chain}: "
            f"total={meta.get('total')}, chunks={num_chunks}"
        )

        for chunk_idx in range(num_chunks):
            chunk_filename = f"{self._CHUNK_PREFIX}{chunk_idx:03d}.pkl"
            chunk_path = self.cache_dir / chunk_filename

            if not chunk_path.exists():
                log.warning(
                    f"[GraphCacheStore] Missing chunk: {chunk_filename}"
                )
                failed_chunks += 1
                continue

            try:
                with open(chunk_path, "rb") as f:
                    chunk_graphs: List[CachedTransactionGraph] = pickle.load(f)
                results.extend(chunk_graphs)
                log.debug(
                    f"[GraphCacheStore] Loaded chunk {chunk_idx:03d}: "
                    f"{len(chunk_graphs)} graphs"
                )
            except Exception as exc:
                log.warning(
                    f"[GraphCacheStore] Failed to load chunk {chunk_filename}: {exc}"
                )
                failed_chunks += 1

        log.info(
            f"[GraphCacheStore] Loaded {len(results)} graphs "
            f"(failed_chunks={failed_chunks})"
        )
        return results

    # ------------------------------------------------------------------
    # 단건 로드 (contract_id 지정)
    # ------------------------------------------------------------------

    def load_one(self, contract_id: str) -> Optional[CachedTransactionGraph]:
        index_path = self.cache_dir / self._INDEX_FILE
        if not index_path.exists():
            return None

        with open(index_path, "rb") as f:
            meta = pickle.load(f)

        if isinstance(meta, list):
            # v1 호환
            entries = meta
        else:
            entries = meta.get("entries", [])

        # 어느 청크에 있는지 index에서 찾기
        target_chunk = None
        for entry in entries:
            if entry.get("contract_id") == contract_id:
                target_chunk = entry.get("chunk_file")
                break

        if target_chunk is None:
            return None

        chunk_path = self.cache_dir / target_chunk
        if not chunk_path.exists():
            return None

        with open(chunk_path, "rb") as f:
            chunk_graphs: List[CachedTransactionGraph] = pickle.load(f)

        for g in chunk_graphs:
            if g.contract_id == contract_id:
                return g

        return None

    # ------------------------------------------------------------------
    # 캐시 정보 출력
    # ------------------------------------------------------------------

    def print_info(self) -> None:
        index_path = self.cache_dir / self._INDEX_FILE
        if not index_path.exists():
            print(f"[GraphCacheStore] No cache found at {self.cache_dir}")
            return

        with open(index_path, "rb") as f:
            meta = pickle.load(f)

        if isinstance(meta, list):
            print(f"[GraphCacheStore] Legacy v1 cache, entries={len(meta)}")
            return

        print(f"[GraphCacheStore] Cache Info")
        print(f"  chain      : {meta.get('chain')}")
        print(f"  version    : {meta.get('version')}")
        print(f"  total      : {meta.get('total')}")
        print(f"  num_chunks : {meta.get('num_chunks')}")
        print(f"  chunk_size : {meta.get('chunk_size')}")
        print(f"  cache_dir  : {self.cache_dir}")

        chunk_files = sorted(self.cache_dir.glob(f"{self._CHUNK_PREFIX}*.pkl"))
        print(f"  chunk files: {len(chunk_files)}")

    # ------------------------------------------------------------------
    # v1 구 포맷 호환 로더
    # ------------------------------------------------------------------

    def _load_legacy_v1(
        self, entries: List[Dict]
    ) -> List[CachedTransactionGraph]:
        """
        이전 버전(파일 당 1개 pkl)의 캐시를 읽는 호환 레이어.
        """
        log.warning(
            "[GraphCacheStore] Legacy v1 cache detected (per-file pkl). "
            "Please rebuild with build_graph_cache.py --overwrite."
        )
        results: List[CachedTransactionGraph] = []
        for entry in entries:
            pkl_path = Path(entry.get("path", ""))
            if not pkl_path.exists():
                continue
            try:
                with open(pkl_path, "rb") as f:
                    ctg: CachedTransactionGraph = pickle.load(f)
                results.append(ctg)
            except Exception as exc:
                log.warning(f"[GraphCacheStore] v1 load failed {pkl_path.name}: {exc}")
        return results
