# src/gog_fraud/data/scripts/build_graph_cache.py
"""
Transaction CSV → log1p-normalized PyG graph → pickle cache

사용법:
    PYTHONPATH=./src python -m gog_fraud.data.scripts.build_graph_cache \
        --transactions_root ../_data/dataset/transactions \
        --cache_root        ../_data/cache/graphs \
        --chain             polygon \
        --overwrite
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
    overwrite: bool = False,
) -> None:
    tx_dir = Path(transactions_root) / chain
    if not tx_dir.exists():
        raise FileNotFoundError(f"Transaction dir not found: {tx_dir}")

    store = GraphCacheStore(cache_root=cache_root, chain=chain)

    if store.exists() and not overwrite:
        log.info(
            f"[build_cache] Cache already exists for chain={chain}. "
            f"Use --overwrite to rebuild."
        )
        return

    csv_files = sorted(p for p in tx_dir.iterdir() if p.suffix.lower() == ".csv")
    log.info(f"[build_cache] Found {len(csv_files)} CSV files for chain={chain}")

    graphs: list[CachedTransactionGraph] = []
    success = 0
    failed = 0

    for path in tqdm(csv_files, desc=f"Building {chain} graph cache", unit="contract"):
        contract_id = path.stem.lower()
        try:
            graph, meta = build_graph_from_csv(path, contract_id, chain)

            # 최종 NaN/Inf 확인
            assert not torch.isnan(graph.x).any(), "x has NaN"
            assert not torch.isinf(graph.x).any(), "x has Inf"
            assert not torch.isnan(graph.edge_attr).any(), "edge_attr has NaN"
            assert not torch.isinf(graph.edge_attr).any(), "edge_attr has Inf"

            ctg = CachedTransactionGraph(
                contract_id=contract_id,
                chain=chain,
                graph=graph,
                meta=meta,
            )
            graphs.append(ctg)
            success += 1

        except Exception as exc:
            log.warning(f"[build_cache] Skip {path.name}: {exc}")
            failed += 1

    log.info(
        f"[build_cache] Build complete: "
        f"success={success}, failed={failed}, total={len(csv_files)}"
    )

    store.save_all(graphs, overwrite=overwrite)
    log.info(f"[build_cache] Cache saved to: {store.cache_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PyG graph cache from transaction CSVs")
    parser.add_argument("--transactions_root", required=True, help="Root dir of raw transaction CSVs")
    parser.add_argument("--cache_root", required=True, help="Root dir for pickle cache output")
    parser.add_argument("--chain", required=True, help="Chain name (e.g. polygon, bsc, ethereum)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing cache")
    args = parser.parse_args()

    build_cache(
        transactions_root=args.transactions_root,
        cache_root=args.cache_root,
        chain=args.chain,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
