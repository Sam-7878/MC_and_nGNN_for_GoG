# src/gog_fraud/data/scripts/build_graph_cache.py
"""
Transaction CSV → log1p-normalized PyG graph → chunk-based pickle cache

사용법:
    PYTHONPATH=./src python -m gog_fraud.data.scripts.build_graph_cache \
        --transactions_root ../_data/dataset/transactions \
        --cache_root        ../_data/dataset/.cache/graphs \
        --chain             polygon \
        [--chunk_size       100] \
        [--overwrite]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm

from gog_fraud.data.preprocessing.graph_builder import build_graph_from_csv
from gog_fraud.data.io.cache_store import CachedTransactionGraph, GraphCacheStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def build_cache(
    transactions_root: str,
    cache_root: str,
    chain: str,
    chunk_size: int = 100,
    overwrite: bool = False,
) -> None:
    tx_dir = Path(transactions_root).resolve() / chain
    if not tx_dir.exists():
        raise FileNotFoundError(f"Transaction dir not found: {tx_dir}")

    # cache_root를 절대 경로로 고정
    resolved_cache = Path(cache_root).resolve()

    store = GraphCacheStore(
        cache_root=str(resolved_cache),
        chain=chain,
        chunk_size=chunk_size,
    )

    if store.exists() and not overwrite:
        log.info(
            f"[build_cache] Cache already exists for chain={chain}. "
            f"Use --overwrite to rebuild."
        )
        store.print_info()
        return

    csv_files = sorted(
        p for p in tx_dir.iterdir() if p.suffix.lower() == ".csv"
    )
    log.info(
        f"[build_cache] Found {len(csv_files)} CSV files for chain={chain}\n"
        f"  tx_dir     : {tx_dir}\n"
        f"  cache_dir  : {store.cache_dir}\n"
        f"  chunk_size : {chunk_size}\n"
        f"  num_chunks : ~{(len(csv_files) + chunk_size - 1) // chunk_size}"
    )

    graphs: list[CachedTransactionGraph] = []
    success = 0
    failed = 0

    for path in tqdm(
        csv_files,
        desc=f"Building {chain} graph cache",
        unit="contract",
    ):
        contract_id = path.stem.lower()
        try:
            graph, meta = build_graph_from_csv(path, contract_id, chain)

            # 최종 안전 확인
            assert not torch.isnan(graph.x).any(),        "x has NaN"
            assert not torch.isinf(graph.x).any(),        "x has Inf"
            assert not torch.isnan(graph.edge_attr).any(), "edge_attr has NaN"
            assert not torch.isinf(graph.edge_attr).any(), "edge_attr has Inf"

            graphs.append(
                CachedTransactionGraph(
                    contract_id=contract_id,
                    chain=chain,
                    graph=graph,
                    meta=meta,
                )
            )
            success += 1

        except Exception as exc:
            log.warning(f"[build_cache] Skip {path.name}: {exc}")
            failed += 1

    log.info(
        f"[build_cache] Build complete: "
        f"success={success}, failed={failed}, total={len(csv_files)}"
    )

    store.save_all(graphs, overwrite=overwrite)
    store.print_info()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build chunked PyG graph cache from transaction CSVs"
    )
    parser.add_argument(
        "--transactions_root", required=True,
        help="Root dir of raw transaction CSVs"
    )
    parser.add_argument(
        "--cache_root", required=True,
        help="Root dir for pickle cache output"
    )
    parser.add_argument(
        "--chain", required=True,
        help="Chain name (e.g. polygon, bsc, ethereum)"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=100,
        help="Number of graphs per chunk pkl file (default: 100)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing cache"
    )
    args = parser.parse_args()

    build_cache(
        transactions_root=args.transactions_root,
        cache_root=args.cache_root,
        chain=args.chain,
        chunk_size=args.chunk_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
