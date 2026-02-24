"""
Common Node Analysis for Multi-Chain Blockchain Data (Optimized Version)
Author: Sam & Gemini
Description: Extract common nodes, analyze frequencies, and generate pairwise edge data for GoG.
Features: Multiprocessing, Data Caching, C-optimized Counting
"""

import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Set, Dict, Tuple, List
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
    """
    단일 CSV 파일에서 노드(주소)를 추출 (병렬 처리를 위해 파일명도 함께 반환)
    """
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
    """
    멀티프로세싱 및 캐싱을 활용하여 컨트랙트별 노드 Set 로드
    """
    chain_dir = data_dir / chain
    cache_dir = data_dir.parent / 'caches'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{chain}_contract_nodes_cache.pkl"

    # 1. 캐시 확인
    if use_cache and cache_file.exists():
        logger.info(f"Loading cached node data from {cache_file}...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # 2. 캐시가 없으면 병렬로 CSV 파싱
    csv_files = list(chain_dir.glob("*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {chain_dir}")
        return {}
        
    contract_nodes = {}
    
    # 사용할 CPU 코어 수 결정 (너무 많이 쓰면 메모리 초과 발생 가능하므로 여유분 1~2개 남김)
    max_workers = max(1, os.cpu_count() - 1)
    logger.info(f"Parsing CSV files using {max_workers} CPU cores...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(get_common_node_file, str(f)) for f in csv_files]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Loading CSV files ({chain})"):
            contract_address, nodes = future.result()
            if nodes:
                contract_nodes[contract_address] = nodes

    # 3. 파싱 완료된 데이터 캐싱
    logger.info(f"Saving parsed data to cache ({cache_file})...")
    with open(cache_file, 'wb') as f:
        pickle.dump(contract_nodes, f)
            
    return contract_nodes


def generate_pairwise_edges(contract_nodes: Dict[str, Set[str]]) -> pd.DataFrame:
    """
    컨트랙트 쌍(Pairwise) 간의 교집합 크기를 계산 (단일 스레드 최적화)
    """
    edges = []
    contracts = list(contract_nodes.keys())
    pairs = list(itertools.combinations(contracts, 2))
    
    for c1, c2 in tqdm(pairs, desc="Calculating Pairwise Edges", unit="pair"):
        set1 = contract_nodes[c1]
        set2 = contract_nodes[c2]
        
        # 교집합 연산 (C 레벨에서 수행되므로 매우 빠름)
        common_nodes = len(set1 & set2)
        
        if common_nodes > 0:
            unique_addresses = len(set1 | set2)
            edges.append({
                'Contract1': c1,
                'Contract2': c2,
                'Common_Nodes': common_nodes,
                'Unique_Addresses': unique_addresses
            })
            
    return pd.DataFrame(edges)


def analyze_frequencies(contract_nodes: Dict[str, Set[str]]) -> pd.DataFrame:
    """
    C-optimized Counter를 사용하여 전체 노드 빈도 초고속 분석
    """
    # 모든 set의 원소를 하나의 거대한 이터레이터로 연결하여 Counter에 전달
    all_nodes_iterator = itertools.chain.from_iterable(contract_nodes.values())
    node_counts = Counter(all_nodes_iterator)
            
    df = pd.DataFrame(node_counts.items(), columns=['node', 'frequency'])
    return df.sort_values('frequency', ascending=False).reset_index(drop=True)


def get_global_common_nodes(contract_nodes: Dict[str, Set[str]]) -> Set[str]:
    """
    모든 파일에 존재하는 절대 공통 노드 추출
    """
    if not contract_nodes:
        return set()
    
    iterator = iter(contract_nodes.values())
    try:
        common_nodes = next(iterator).copy()
        for nodes in iterator:
            common_nodes &= nodes  # 교집합 연산자 사용 (더 빠름)
            if not common_nodes:
                break
        return common_nodes
    except StopIteration:
        return set()


def save_results(global_common_nodes: Set[str], 
                 frequency_df: pd.DataFrame, 
                 pairwise_df: pd.DataFrame,
                 output_dir: Path, 
                 chain: str) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}
    
    pairwise_file = output_dir / f"{chain}_common_nodes_except_null_labels.csv"
    if not pairwise_df.empty:
        pairwise_df.to_csv(pairwise_file, index=False)
    else:
        pd.DataFrame(columns=['Contract1', 'Contract2', 'Common_Nodes', 'Unique_Addresses']).to_csv(pairwise_file, index=False)
    
    common_nodes_file = output_dir / f"{chain}_global_common_nodes_list.csv"
    pd.DataFrame(sorted(list(global_common_nodes)), columns=['address']).to_csv(common_nodes_file, index=False)
    
    frequency_file = output_dir / f"{chain}_node_frequency.csv"
    if not frequency_df.empty:
        frequency_df.to_csv(frequency_file, index=False)
    else:
        pd.DataFrame(columns=['node', 'frequency']).to_csv(frequency_file, index=False)
    
    summary_file = output_dir / f"{chain}_analysis_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"Node Analysis Summary for {chain.upper()}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"[Pairwise Edge (GoG) Analysis]\n")
        f.write(f"Total valid edges (Common_Nodes > 0): {len(pairwise_df):,}\n\n")
        
        f.write(f"[Frequency Analysis]\n")
        f.write(f"Total unique nodes across all files: {len(frequency_df):,}\n")
        if not frequency_df.empty:
            f.write(f"Max frequency: {frequency_df['frequency'].max()}\n")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(description='Analyze nodes and generate edges for GoG')
    parser.add_argument('--chain', type=str, required=True, help='Blockchain name (e.g., bsc, eth)')
    parser.add_argument('--clear-cache', action='store_true', help='Force reload CSVs ignoring cache')
    args = parser.parse_args()
    
    data_dir = Path(__file__).parent.parent.parent / '_data' / 'dataset' / 'transactions'
    output_dir = Path(__file__).parent.parent.parent / '_data' / 'graphs' / args.chain
    
    logger.info("="*70)
    logger.info(f"Starting Optimized Node Analysis for {args.chain.upper()}")
    
    if args.clear_cache:
        logger.info("Cache clearing requested. Will re-parse all CSV files.")
    
    # 1. 멀티프로세싱 + 캐싱 적용된 파일 로드
    logger.info("\n[Phase 1] Loading CSV files (with Cache & Multiprocessing)...")
    contract_nodes = load_all_contract_nodes(data_dir, args.chain, use_cache=not args.clear_cache)
    if not contract_nodes:
        sys.exit(1)
        
    # 2. Pairwise Edge 생성
    logger.info("\n[Phase 2] Generating Pairwise Edges...")
    pairwise_df = generate_pairwise_edges(contract_nodes)
    
    # 3. 최적화된 Frequency 분석
    logger.info("\n[Phase 3] Analyzing node frequencies...")
    frequency_df = analyze_frequencies(contract_nodes)
    
    # 4. Global 공통 노드 확인
    logger.info("\n[Phase 4] Checking global common nodes...")
    global_common = get_global_common_nodes(contract_nodes)
    
    # 5. 저장
    logger.info("\n[Saving Results]")
    save_results(global_common, frequency_df, pairwise_df, output_dir, args.chain)
    
    print("\n" + "="*70)
    print("Analysis Completed Successfully!")
    print(f"Total Contracts Processed: {len(contract_nodes):,}")
    print(f"Edges Generated (for global.py): {len(pairwise_df):,}")
    print("="*70)

if __name__ == "__main__":
    main()