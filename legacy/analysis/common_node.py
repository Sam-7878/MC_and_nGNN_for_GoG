#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analysis/common_node.py

역할:
  - Level 2 (GoG): contract 간 공통 주소(common-node) 기반 edge 생성 (기존 유지)
  - Level 1 (graph_individual): contract 내부 address-level node/edge 생성 (신규 추가)

출력 파일:
  Level 2 (기존):
    _data/GoG/edges/{chain}_common_nodes_except_null_labels.csv
    _data/GoG/nodes/{chain}_node_frequency.csv
    _data/GoG/nodes/{chain}_global_common_nodes_list.csv

  Level 1 (신규):
    _data/level1/{chain}_level1_nodes.csv
    _data/level1/{chain}_level1_edges.csv
"""

import os
import re
import gc
import sys
import pickle
import logging
import argparse
import warnings
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ============================================================
# Logger
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# ============================================================
# Column inference helpers
# ============================================================

FROM_COLUMNS  = ["from_address", "from", "sender", "fromAddress", "From", "from_addr"]
TO_COLUMNS    = ["to_address",   "to",   "receiver", "toAddress", "To",   "to_addr"]
TIME_COLUMNS  = [
    "timestamp", "block_timestamp", "time", "ts",
    "block_number", "blockNumber", "block_time", "tx_index"
]
VALUE_COLUMNS = ["value", "amount", "tx_value", "quantity", "token_value"]
HASH_COLUMNS  = ["hash", "tx_hash", "transaction_hash", "txhash", "transactionHash"]

ZERO_ADDRESS  = "0x0000000000000000000000000000000000000000"


def _find_col(df_columns: list, candidates: list):
    lowered = {c.lower(): c for c in df_columns}
    for cand in candidates:
        if cand.lower() in lowered:
            return lowered[cand.lower()]
    return None


def _safe_read_csv(file_path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(
            file_path,
            dtype=str,
            low_memory=False,
            on_bad_lines="skip",
            na_values=["", "NA", "null", "None", "nan", "NaN"]
        )
        return df
    except Exception as e:
        logger.warning(f"[read_csv error] {file_path} : {e}")
        return None


def _normalize_address(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.lower()


# ============================================================
# Level 2: 기존 common-node 로직 (원본 보존 + 소폭 리팩토링)
# ============================================================

def get_common_node_file(file_path: str) -> tuple[str, set]:
    """
    한 contract CSV에서 고유 주소 집합(Set[str])을 뽑습니다.
    반환: (contract_id, set_of_addresses)
    """
    contract_id = Path(file_path).stem
    df = _safe_read_csv(file_path)

    if df is None or df.empty:
        return contract_id, set()

    from_col = _find_col(df.columns.tolist(), FROM_COLUMNS)
    to_col   = _find_col(df.columns.tolist(), TO_COLUMNS)

    nodes = set()

    if from_col:
        from_addrs = _normalize_address(df[from_col])
        nodes.update(a for a in from_addrs if a and a != ZERO_ADDRESS)

    if to_col:
        to_addrs = _normalize_address(df[to_col])
        nodes.update(a for a in to_addrs if a and a != ZERO_ADDRESS)

    return contract_id, nodes


def generate_pairwise_edges_and_save(
    contract_nodes: dict,
    label_df: pd.DataFrame,
    output_file: str,
    min_common: int = 1
):
    """
    contract 쌍 간 common-node 수 계산 후 저장. (Level 2 GoG edge)
    """
    logger.info(f"[Level2] Generating pairwise common-node edges -> {output_file}")

    contract_ids = list(contract_nodes.keys())
    rows = []

    for c1, c2 in tqdm(
        combinations(contract_ids, 2),
        total=len(contract_ids) * (len(contract_ids) - 1) // 2,
        desc="Pairwise edges"
    ):
        s1 = contract_nodes.get(c1, set())
        s2 = contract_nodes.get(c2, set())
        common = s1 & s2
        n_common = len(common)

        if n_common < min_common:
            continue

        union_size = len(s1 | s2)
        rows.append({
            "Contract1":       c1,
            "Contract2":       c2,
            "Common_Nodes":    n_common,
            "Unique_Addresses": union_size,
        })

    if not rows:
        logger.warning("[Level2] No common-node edges found.")
        df_out = pd.DataFrame(columns=[
            "Contract1", "Contract2",
            "Common_Nodes", "Unique_Addresses",
            "Label1", "Label2"
        ])
        df_out.to_csv(output_file, index=False)
        return df_out

    df_out = pd.DataFrame(rows)

    # label merge
    if label_df is not None and not label_df.empty:
        label_map = {}
        for col in ["Contract", "contract", "contract_address", "address", "graph_id"]:
            if col in label_df.columns:
                for col2 in ["label", "Label", "is_fraud", "fraud"]:
                    if col2 in label_df.columns:
                        label_map = dict(zip(
                            label_df[col].astype(str).str.lower().str.strip(),
                            label_df[col2]
                        ))
                        break
                if label_map:
                    break

        if label_map:
            df_out["Label1"] = df_out["Contract1"].map(label_map)
            df_out["Label2"] = df_out["Contract2"].map(label_map)

    # null label 제거 (파일명에 except_null_labels 포함)
    if "Label1" in df_out.columns and "Label2" in df_out.columns:
        df_out = df_out.dropna(subset=["Label1", "Label2"])

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_file, index=False)
    logger.info(f"[Level2] Saved {len(df_out)} edges -> {output_file}")
    return df_out


def analyze_frequencies(
    contract_nodes: dict,
    output_freq_file: str,
    output_global_file: str,
    top_k: int = 50
):
    """
    전체 contract에서 주소 등장 빈도 분석 및 global common node 저장.
    """
    addr_count = {}
    for nodes in contract_nodes.values():
        for addr in nodes:
            addr_count[addr] = addr_count.get(addr, 0) + 1

    freq_df = pd.DataFrame(
        list(addr_count.items()),
        columns=["address", "contract_count"]
    ).sort_values("contract_count", ascending=False)

    Path(output_freq_file).parent.mkdir(parents=True, exist_ok=True)
    freq_df.to_csv(output_freq_file, index=False)
    logger.info(f"[Level2] Saved frequency -> {output_freq_file}")

    global_df = freq_df.head(top_k)
    global_df.to_csv(output_global_file, index=False)
    logger.info(f"[Level2] Saved global common nodes -> {output_global_file}")

    return freq_df, global_df


# ============================================================
# Level 1: intra-graph node/edge 생성 (신규)
# ============================================================

def build_intra_graph_from_file(file_path: str) -> tuple[str, pd.DataFrame | None, pd.DataFrame | None]:
    """
    한 contract CSV에서 Level 1 intra-graph 데이터를 생성합니다.

    반환:
        (contract_id, node_df, edge_df)

    node_df 컬럼:
        graph_id, node_id,
        out_tx_count, out_degree, total_out_value, first_out_ts, last_out_ts,
        in_tx_count,  in_degree,  total_in_value,  first_in_ts,  last_in_ts

    edge_df 컬럼:
        graph_id, src, dst,
        tx_count, total_value, first_ts, last_ts
    """
    contract_id = Path(file_path).stem
    df = _safe_read_csv(file_path)

    if df is None or df.empty:
        return contract_id, None, None

    col_list = df.columns.tolist()

    from_col  = _find_col(col_list, FROM_COLUMNS)
    to_col    = _find_col(col_list, TO_COLUMNS)
    time_col  = _find_col(col_list, TIME_COLUMNS)
    value_col = _find_col(col_list, VALUE_COLUMNS)

    if from_col is None or to_col is None:
        logger.debug(f"[Level1] Skip {contract_id}: no from/to columns")
        return contract_id, None, None

    # ---- 기본 정리 ----
    work = pd.DataFrame()
    work["src"] = _normalize_address(df[from_col])
    work["dst"] = _normalize_address(df[to_col])

    # null / zero-address 제거
    valid_mask = (
        (work["src"] != "") &
        (work["dst"] != "") &
        (work["src"] != ZERO_ADDRESS) &
        (work["dst"] != ZERO_ADDRESS) &
        (work["src"] != "nan") &
        (work["dst"] != "nan")
    )
    work = work[valid_mask].copy().reset_index(drop=True)

    if work.empty:
        return contract_id, None, None

    # ---- timestamp ----
    if time_col and time_col in df.columns:
        ts_raw = pd.to_numeric(df.loc[valid_mask, time_col].reset_index(drop=True), errors="coerce")
        work["ts"] = ts_raw.fillna(work.index.to_series())
    else:
        # 원본 row 순서를 proxy timestamp로 사용
        work["ts"] = np.arange(len(work), dtype=np.float64)

    # ---- value ----
    if value_col and value_col in df.columns:
        val_raw = pd.to_numeric(df.loc[valid_mask, value_col].reset_index(drop=True), errors="coerce")
        work["value"] = val_raw.fillna(0.0)
    else:
        work["value"] = 0.0

    work["graph_id"] = contract_id

    # ============================================================
    # Edge aggregation: (graph_id, src, dst) 단위
    # ============================================================
    edge_df = (
        work.groupby(["graph_id", "src", "dst"], as_index=False)
        .agg(
            tx_count    = ("src",   "size"),
            total_value = ("value", "sum"),
            first_ts    = ("ts",    "min"),
            last_ts     = ("ts",    "max"),
        )
    )

    # ============================================================
    # Node feature aggregation
    # ============================================================

    # out-stats: src 기준
    out_stats = (
        work.groupby("src", as_index=False)
        .agg(
            out_tx_count    = ("dst",   "size"),
            out_degree      = ("dst",   "nunique"),
            total_out_value = ("value", "sum"),
            first_out_ts    = ("ts",    "min"),
            last_out_ts     = ("ts",    "max"),
        )
        .rename(columns={"src": "node_id"})
    )

    # in-stats: dst 기준
    in_stats = (
        work.groupby("dst", as_index=False)
        .agg(
            in_tx_count    = ("src",   "size"),
            in_degree      = ("src",   "nunique"),
            total_in_value = ("value", "sum"),
            first_in_ts    = ("ts",    "min"),
            last_in_ts     = ("ts",    "max"),
        )
        .rename(columns={"dst": "node_id"})
    )

    # 전체 node 합집합
    all_nodes_set = set(work["src"]).union(set(work["dst"]))
    node_base = pd.DataFrame({"node_id": sorted(all_nodes_set)})

    node_df = (
        node_base
        .merge(out_stats, on="node_id", how="left")
        .merge(in_stats,  on="node_id", how="left")
        .fillna(0)
    )

    # 파생 feature
    node_df["total_tx_count"]  = node_df["out_tx_count"]    + node_df["in_tx_count"]
    node_df["total_degree"]    = node_df["out_degree"]       + node_df["in_degree"]
    node_df["total_value"]     = node_df["total_out_value"]  + node_df["total_in_value"]
    node_df["net_value"]       = node_df["total_in_value"]   - node_df["total_out_value"]
    node_df["degree_ratio"]    = np.where(
        node_df["total_degree"] > 0,
        node_df["out_degree"] / node_df["total_degree"],
        0.0
    )
    node_df["active_duration"] = (
        node_df[["first_out_ts", "first_in_ts"]].min(axis=1) * -1 +
        node_df[["last_out_ts",  "last_in_ts" ]].max(axis=1)
    ).clip(lower=0.0)

    node_df.insert(0, "graph_id", contract_id)

    return contract_id, node_df, edge_df


def _worker_level1(file_path: str) -> tuple:
    """ProcessPoolExecutor worker (Level 1)"""
    return build_intra_graph_from_file(file_path)


def _worker_level2(file_path: str) -> tuple:
    """ProcessPoolExecutor worker (Level 2)"""
    return get_common_node_file(file_path)


# ============================================================
# Cache helpers
# ============================================================

def load_cache(cache_path: str):
    if Path(cache_path).exists():
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            logger.info(f"[Cache] Loaded -> {cache_path}")
            return data
        except Exception as e:
            logger.warning(f"[Cache] Load failed ({cache_path}): {e}")
    return None


def save_cache(data, cache_path: str):
    try:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"[Cache] Saved -> {cache_path}")
    except Exception as e:
        logger.warning(f"[Cache] Save failed ({cache_path}): {e}")


# ============================================================
# Main pipeline
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="common_node.py: Level1 intra-graph + Level2 GoG common-node edge generation"
    )
    parser.add_argument("--chain",        type=str, required=True,
                        help="blockchain name (e.g. bsc, eth, polygon)")
    parser.add_argument("--data_dir",     type=str, default=None,
                        help="raw transaction CSV directory (default: ./_data/dataset/local_graph/<chain>)")
    parser.add_argument("--output_dir",   type=str, default=None,
                        help="base output directory (default: ./_data)")
    parser.add_argument("--label_csv",    type=str, default=None,
                        help="label CSV (e.g. bsc_basic_metrics_processed.csv)")
    parser.add_argument("--min_common",   type=int, default=1,
                        help="Level2: min common-node count to create an edge")
    parser.add_argument("--top_k",        type=int, default=50,
                        help="Level2: top-k global common nodes to save")
    parser.add_argument("--max_workers",  type=int, default=4,
                        help="parallel workers")
    parser.add_argument("--skip_level1",  action="store_true",
                        help="Level1 intra-graph 생성 건너뜀")
    parser.add_argument("--skip_level2",  action="store_true",
                        help="Level2 common-node edge 생성 건너뜀")
    parser.add_argument("--use_cache",    action="store_true",
                        help="contract_nodes dict를 pickle 캐시로 관리")
    parser.add_argument("--cache_path",   type=str, default=None,
                        help="cache pickle 경로 (default: output_dir/cache/{chain}_contract_nodes.pkl)")
    return parser.parse_args()


def resolve_paths(args):
    chain = args.chain.lower()

    data_dir = Path(args.data_dir) if args.data_dir else \
        Path(f"./_data/dataset/local_graph/{chain}")

    output_dir = Path(args.output_dir) if args.output_dir else Path("./_data")

    # Level 2 출력 경로 (기존 구조 유지)
    gog_edge_dir  = output_dir / "GoG" / "edges"
    gog_node_dir  = output_dir / "GoG" / "nodes"

    level2_edge_file   = gog_edge_dir  / f"{chain}_common_nodes_except_null_labels.csv"
    level2_freq_file   = gog_node_dir  / f"{chain}_node_frequency.csv"
    level2_global_file = gog_node_dir  / f"{chain}_global_common_nodes_list.csv"

    # Level 1 출력 경로 (신규)
    level1_dir         = output_dir / "level1"
    level1_node_file   = level1_dir / f"{chain}_level1_nodes.csv"
    level1_edge_file   = level1_dir / f"{chain}_level1_edges.csv"

    # cache
    cache_path = args.cache_path or str(output_dir / "cache" / f"{chain}_contract_nodes.pkl")

    # label CSV 추론
    label_csv = args.label_csv
    if label_csv is None:
        candidates = [
            output_dir / "dataset" / "features" / f"{chain}_basic_metrics_processed.csv",
            output_dir / f"{chain}_basic_metrics_processed.csv",
            Path(f"./_data/dataset/features/{chain}_basic_metrics_processed.csv"),
            Path(f"./_data/{chain}_basic_metrics_processed.csv"),
        ]
        for c in candidates:
            if c.exists():
                label_csv = str(c)
                break

    return {
        "chain":              chain,
        "data_dir":           data_dir,
        "label_csv":          label_csv,
        "level2_edge_file":   str(level2_edge_file),
        "level2_freq_file":   str(level2_freq_file),
        "level2_global_file": str(level2_global_file),
        "level1_node_file":   str(level1_node_file),
        "level1_edge_file":   str(level1_edge_file),
        "cache_path":         cache_path,
    }


def get_csv_files(data_dir: Path) -> list:
    files = sorted([
        str(p) for p in data_dir.rglob("*.csv")
    ])
    if not files:
        logger.warning(f"[Data] No CSV files found in {data_dir}")
    else:
        logger.info(f"[Data] Found {len(files)} CSV files in {data_dir}")
    return files


def run_level1(csv_files, paths, max_workers):
    """
    Level 1: contract 내부 node/edge 생성 및 저장
    """
    logger.info("\n" + "=" * 60)
    logger.info("[Phase 1-L1] Building Level-1 intra-graph data...")
    logger.info("=" * 60)

    all_node_dfs = []
    all_edge_dfs = []

    skipped = 0
    success = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_worker_level1, fp): fp
            for fp in csv_files
        }

        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Level1 intra-graph"
        ):
            contract_id, node_df, edge_df = future.result()

            if node_df is not None and not node_df.empty:
                all_node_dfs.append(node_df)
                success += 1
            else:
                skipped += 1

            if edge_df is not None and not edge_df.empty:
                all_edge_dfs.append(edge_df)

    logger.info(f"\n[Level1] Success: {success}  /  Skipped: {skipped}")

    # 저장
    node_out = paths["level1_node_file"]
    edge_out = paths["level1_edge_file"]

    Path(node_out).parent.mkdir(parents=True, exist_ok=True)

    if all_node_dfs:
        node_merged = pd.concat(all_node_dfs, ignore_index=True)
        node_merged.to_csv(node_out, index=False)
        logger.info(f"[Level1] Saved {len(node_merged)} node rows -> {node_out}")
    else:
        logger.warning("[Level1] No node data generated.")

    if all_edge_dfs:
        edge_merged = pd.concat(all_edge_dfs, ignore_index=True)
        edge_merged.to_csv(edge_out, index=False)
        logger.info(f"[Level1] Saved {len(edge_merged)} edge rows -> {edge_out}")
    else:
        logger.warning("[Level1] No edge data generated.")

    return all_node_dfs, all_edge_dfs


def run_level2(csv_files, paths, args, max_workers):
    """
    Level 2: contract 간 common-node edge 생성 및 저장
    """
    logger.info("\n" + "=" * 60)
    logger.info("[Phase 1-L2] Building Level-2 common-node sets...")
    logger.info("=" * 60)

    # cache 시도
    contract_nodes = None
    if args.use_cache:
        contract_nodes = load_cache(paths["cache_path"])

    if contract_nodes is None:
        contract_nodes = {}

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_worker_level2, fp): fp
                for fp in csv_files
            }

            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Level2 common-node sets"
            ):
                contract_id, nodes = future.result()
                if nodes:
                    contract_nodes[contract_id] = nodes

        logger.info(f"[Level2] Collected node sets for {len(contract_nodes)} contracts")

        if args.use_cache:
            save_cache(contract_nodes, paths["cache_path"])

    # label 로드
    label_df = None
    if paths["label_csv"] and Path(paths["label_csv"]).exists():
        try:
            label_df = pd.read_csv(paths["label_csv"])
            logger.info(f"[Level2] Label CSV loaded -> {paths['label_csv']}")
        except Exception as e:
            logger.warning(f"[Level2] Label CSV load failed: {e}")

    # pairwise edge 생성
    generate_pairwise_edges_and_save(
        contract_nodes = contract_nodes,
        label_df       = label_df,
        output_file    = paths["level2_edge_file"],
        min_common     = args.min_common
    )

    # 빈도 분석
    analyze_frequencies(
        contract_nodes    = contract_nodes,
        output_freq_file  = paths["level2_freq_file"],
        output_global_file= paths["level2_global_file"],
        top_k             = args.top_k
    )

    return contract_nodes


def print_summary(paths, skip_level1, skip_level2):
    logger.info("\n" + "=" * 60)
    logger.info("[Summary] Output files")
    logger.info("=" * 60)

    if not skip_level1:
        logger.info(f"  [Level1 - graph_individual input]")
        logger.info(f"    node file : {paths['level1_node_file']}")
        logger.info(f"    edge file : {paths['level1_edge_file']}")

    if not skip_level2:
        logger.info(f"  [Level2 - GoG input]")
        logger.info(f"    edge file      : {paths['level2_edge_file']}")
        logger.info(f"    freq file      : {paths['level2_freq_file']}")
        logger.info(f"    global common  : {paths['level2_global_file']}")

    logger.info("=" * 60)
    logger.info("\n[Next Step]")

    if not skip_level1:
        logger.info(
            "  graph_individual/main.py 실행 시:\n"
            "    --node_csv   " + paths["level1_node_file"] + "\n"
            "    --edge_csv   " + paths["level1_edge_file"] + "\n"
            "    --graph_id_col graph_id\n"
            "    --node_id_col  node_id\n"
            "    --edge_mode    precomputed\n"
        )


def main():
    args = parse_args()
    paths = resolve_paths(args)

    chain      = paths["chain"]
    data_dir   = paths["data_dir"]
    max_workers = args.max_workers

    logger.info("=" * 60)
    logger.info(f" common_node.py  |  Chain: {chain.upper()}")
    logger.info("=" * 60)
    logger.info(f"  data_dir    : {data_dir}")
    logger.info(f"  label_csv   : {paths['label_csv']}")
    logger.info(f"  skip_level1 : {args.skip_level1}")
    logger.info(f"  skip_level2 : {args.skip_level2}")
    logger.info(f"  max_workers : {max_workers}")
    logger.info("=" * 60)

    if not Path(data_dir).exists():
        logger.error(f"data_dir not found: {data_dir}")
        sys.exit(1)

    csv_files = get_csv_files(Path(data_dir))
    if not csv_files:
        logger.error("No CSV files to process.")
        sys.exit(1)

    # ---- Level 1 ----
    if not args.skip_level1:
        run_level1(csv_files, paths, max_workers)
        gc.collect()

    # ---- Level 2 ----
    if not args.skip_level2:
        run_level2(csv_files, paths, args, max_workers)
        gc.collect()

    print_summary(paths, args.skip_level1, args.skip_level2)
    logger.info("[Done]")


if __name__ == "__main__":
    main()
