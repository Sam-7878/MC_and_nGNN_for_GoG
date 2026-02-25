"""
Common Node Analysis for Multi-Chain Blockchain Data (OOM Safe Version)
Author: Sam & Gemini
Description: Extract common nodes, analyze frequencies, and generate pairwise edge data for GoG.
Features: Multiprocessing, Data Caching, C-optimized Counting, Memory-Safe Chunk Streaming
"""

import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Set, Dict, Tuple
import sys
from datetime import datetime
import itertools
from collections import Counter
import pickle
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_common_node_file(file_path: str) -> Tuple[str, Set[str]]:
    try:
        df = pd.read_csv(
            file_path,
            dtype=str,              
            low_memory=False,       
            on_bad_lines='skip',    
            na_values=['', 'NA', 'null', 'None']  
        )
        
        nodes = set()
        
        from_columns = ['from_address', 'from', 'sender', 'fromAddress', 'From']
        to_columns = ['to_address', 'to', 'receiver', 'toAddress', 'To']
        
        from_col = next((col for col in from_columns if col in df.columns), None)
        if from_col:
            from_addrs = df[from_col].dropna().str.lower().str.strip()
            from_addrs = from_addrs[
                (from_addrs != '') & 
                (from_addrs != '0x0000000000000000000000000000000000000000')
            ]
            nodes.update(from_addrs.unique())
            
        to_col = next((col for col in to_columns if col in df.columns), None)
        if to_col:
            to_addrs = df[to_col].dropna().str.lower().str.strip()
            to_addrs = to_addrs[
                (to_addrs != '') & 
                (to_addrs != '0x0000000000000000000000000000000000000000')
            ]
            nodes.update(to_addrs.unique())
            
        return Path(file_path).stem, nodes
        
    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}")
        return Path(file_path).stem, set()


def load_all_contract_nodes(data_dir: Path, chain: str, use_cache: bool = True) -> Dict[str, Set[str]]:
    chain_dir = data_dir / chain
    cache_dir = data_dir.parent / '.cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{chain}_contract_nodes_cache.pkl"

    if use_cache and cache_file.exists():
        logger.info(f"Loading cached node data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    csv_files = list(chain_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {chain_dir}")
        return {}
        
    contract_nodes = {}
    max_workers = max(1, os.cpu_count() - 4)  # 시스템에 따라 조정 (예: CPU 코어 수 - 4)
    logger.info(f"Parsing CSV files using {max_workers} CPU cores...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(get_common_node_file, str(f)) for f in csv_files]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Loading CSV files ({chain})"):
            contract_address, nodes = future.result()
            if nodes:
                contract_nodes[contract_address] = nodes

    logger.info(f"Saving parsed data to cache ({cache_file})...")
    with open(cache_file, 'wb') as f:
        pickle.dump(contract_nodes, f)
            
    return contract_nodes


def generate_pairwise_edges_and_save(contract_nodes: Dict[str, Set[str]], output_file: Path) -> int:
    """
    [OOM 방지] 엣지를 메모리에 전부 담지 않고 청크(Chunk) 단위로 파일에 직접 씁니다.
    """
    contracts = list(contract_nodes.keys())
    
    # 총 조합 개수를 수학적으로 계산 (리스트로 변환하지 않음)
    total_pairs = len(contracts) * (len(contracts) - 1) // 2
    
    # 1. 대상 파일의 부모 디렉토리 생성 및 헤더 먼저 쓰기
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=['Contract1', 'Contract2', 'Common_Nodes', 'Unique_Addresses']).to_csv(output_file, index=False)
    
    edges_chunk = []
    chunk_size = 500_000  # 50만 개가 모일 때마다 디스크에 저장하고 메모리 비움
    total_valid_edges = 0
    
    # list()로 감싸지 않고 바로 이터레이터로 순회하여 메모리 점유 = 0
    pairs_iterator = itertools.combinations(contracts, 2)
    
    for c1, c2 in tqdm(pairs_iterator, total=total_pairs, desc="Calculating Pairwise Edges", unit="pair"):
        set1 = contract_nodes[c1]
        set2 = contract_nodes[c2]
        
        common_nodes = len(set1 & set2)
        
        if common_nodes > 0:
            unique_addresses = len(set1 | set2)
            edges_chunk.append({
                'Contract1': c1,
                'Contract2': c2,
                'Common_Nodes': common_nodes,
                'Unique_Addresses': unique_addresses
            })
            total_valid_edges += 1
            
            # 버퍼가 꽉 차면 파일에 이어쓰기(append) 하고 버퍼를 비움 (메모리 해제)
            if len(edges_chunk) >= chunk_size:
                pd.DataFrame(edges_chunk).to_csv(output_file, mode='a', header=False, index=False)
                edges_chunk.clear()
                
    # 남은 자투리 데이터 마저 쓰기
    if edges_chunk:
        pd.DataFrame(edges_chunk).to_csv(output_file, mode='a', header=False, index=False)
        edges_chunk.clear()
        
    return total_valid_edges


def analyze_frequencies(contract_nodes: Dict[str, Set[str]]) -> pd.DataFrame:
    all_nodes_iterator = itertools.chain.from_iterable(contract_nodes.values())
    node_counts = Counter(all_nodes_iterator)
    df = pd.DataFrame(node_counts.items(), columns=['node', 'frequency'])
    return df.sort_values('frequency', ascending=False).reset_index(drop=True)


def get_global_common_nodes(contract_nodes: Dict[str, Set[str]]) -> Set[str]:
    if not contract_nodes:
        return set()
    
    iterator = iter(contract_nodes.values())
    try:
        common_nodes = next(iterator).copy()
        for nodes in iterator:
            common_nodes &= nodes
            if not common_nodes:
                break
        return common_nodes
    except StopIteration:
        return set()


def main():
    parser = argparse.ArgumentParser(description='Analyze nodes and generate edges for GoG')
    parser.add_argument('--chain', type=str, required=True, help='Blockchain name (e.g., bsc, eth)')
    parser.add_argument('--clear-cache', action='store_true', help='Force reload CSVs ignoring cache')
    args = parser.parse_args()
    
    data_dir = Path(__file__).parent.parent.parent / '_data' / 'dataset' / 'transactions'
    output_dir = Path(__file__).parent.parent.parent / '_data' / 'graphs' / args.chain
    
    logger.info("="*70)
    logger.info(f"Starting OOM-Safe Node Analysis for {args.chain.upper()}")
    
    if args.clear_cache:
        logger.info("Cache clearing requested. Will re-parse all CSV files.")
    
    # 1. 파일 로드 (이전 단계에서 캐시가 생성되어 있다면 매우 빠르게 넘어갑니다)
    logger.info("\n[Phase 1] Loading CSV files (with Cache & Multiprocessing)...")
    contract_nodes = load_all_contract_nodes(data_dir, args.chain, use_cache=not args.clear_cache)
    if not contract_nodes:
        sys.exit(1)
        
    # 2. Pairwise Edge 생성 (청크 단위로 바로 파일에 저장하도록 변경)
    logger.info("\n[Phase 2] Generating Pairwise Edges (Streaming to disk)...")
    pairwise_file = output_dir / f"{args.chain}_common_nodes_except_null_labels.csv"
    total_valid_edges = generate_pairwise_edges_and_save(contract_nodes, pairwise_file)
    logger.info(f"Saved pairwise edges to {pairwise_file}")
    
    # 3. Frequency 분석
    logger.info("\n[Phase 3] Analyzing node frequencies...")
    frequency_df = analyze_frequencies(contract_nodes)
    frequency_file = output_dir / f"{args.chain}_node_frequency.csv"
    frequency_df.to_csv(frequency_file, index=False)
    
    # 4. Global 공통 노드 확인
    logger.info("\n[Phase 4] Checking global common nodes...")
    global_common = get_global_common_nodes(contract_nodes)
    common_nodes_file = output_dir / f"{args.chain}_global_common_nodes_list.csv"
    pd.DataFrame(sorted(list(global_common)), columns=['address']).to_csv(common_nodes_file, index=False)
    
    # 5. 요약 저장
    logger.info("\n[Saving Results Summary]")
    summary_file = output_dir / f"{args.chain}_analysis_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"Node Analysis Summary for {args.chain.upper()}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        f.write(f"[Pairwise Edge (GoG) Analysis]\n")
        f.write(f"Total valid edges (Common_Nodes > 0): {total_valid_edges:,}\n\n")
        f.write(f"[Frequency Analysis]\n")
        f.write(f"Total unique nodes across all files: {len(frequency_df):,}\n")
        if not frequency_df.empty:
            f.write(f"Max frequency: {frequency_df['frequency'].max()}\n")
    
    print("\n" + "="*70)
    print("Analysis Completed Successfully! (No Memory Leaks)")
    print(f"Total Contracts Processed: {len(contract_nodes):,}")
    print(f"Edges Generated (for global.py): {total_valid_edges:,}")
    print("="*70)

if __name__ == "__main__":
    main()