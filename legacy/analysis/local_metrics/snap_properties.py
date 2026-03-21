"""
SNAP Graph Properties Analysis - OPTIMIZED VERSION
- Multiprocessing for parallel file processing
- Vectorized operations (no iterrows)
- Checkpoint system for resume capability
- Progress tracking with time estimation
"""

import snap
import pandas as pd
from tqdm import tqdm
import os
import argparse
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import numpy as np

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_snap_graph_optimized(tx_df):
    """
    최적화된 SNAP 그래프 생성
    - iterrows() 대신 벡터화 처리
    - 메모리 효율적인 노드 ID 매핑
    
    Args:
        tx_df: Transaction DataFrame (columns: 'from', 'to')
    
    Returns:
        snap.TNGraph: 방향 그래프
    """
    G = snap.TNGraph.New()
    
    # ✅ 벡터화: iterrows() 대신 unique() + enumerate 사용
    # 모든 주소를 한 번에 추출
    all_addresses = pd.concat([tx_df['from'], tx_df['to']]).unique()
    
    # 주소 → 노드 ID 매핑 (딕셔너리 한 번에 생성)
    node_dict = {addr: idx for idx, addr in enumerate(all_addresses)}
    
    # 모든 노드 추가
    for node_id in range(len(all_addresses)):
        G.AddNode(node_id)
    
    # ✅ 벡터화: 엣지 추가 (iterrows 대신 values 사용)
    from_ids = tx_df['from'].map(node_dict).values
    to_ids = tx_df['to'].map(node_dict).values
    
    # 엣지 추가 (중복 제거 옵션)
    for from_id, to_id in zip(from_ids, to_ids):
        if from_id != to_id:  # self-loop 제거
            try:
                G.AddEdge(int(from_id), int(to_id))
            except:
                pass  # 중복 엣지는 무시
    
    return G


def compute_metrics(G, sample_size=100):
    """
    SNAP 그래프 메트릭 계산
    
    Args:
        G: SNAP graph
        sample_size: BFS 샘플 크기 (작을수록 빠름)
    
    Returns:
        tuple: (effective_diameter, clustering_coefficient)
    """
    try:
        # Effective Diameter (BFS 기반 근사)
        effective_diameter = snap.GetBfsEffDiam(G, sample_size, False)
    except:
        effective_diameter = 0.0
    
    try:
        # Clustering Coefficient (-1 = 전체 그래프)
        clustering_coefficient = snap.GetClustCf(G, -1)
    except:
        clustering_coefficient = 0.0
    
    return effective_diameter, clustering_coefficient


def process_single_contract(addr, chain, end_date, sample_size=100, debug=False):
    """
    단일 컨트랙트 처리 (병렬화 가능)
    
    Args:
        addr: Contract address
        chain: Chain name
        end_date: Cutoff date
        sample_size: BFS sample size
        debug: Debug mode
    
    Returns:
        dict: Metrics or None
    """
    try:
        # 1. 데이터 로드
        file_path = f'../../_data/dataset/transactions/{chain}/{addr}.csv'
        
        if not os.path.exists(file_path):
            if debug:
                logger.debug(f"File not found: {addr}")
            return None
        
        # 필요한 컬럼만 로드
        tx = pd.read_csv(
            file_path,
            dtype=str,
            usecols=['from', 'to', 'timestamp'],
            low_memory=False
        )
        
        if tx.empty:
            return None
        
        # 2. 타임스탬프 필터링 ✅ 수정된 부분
        # 먼저 문자열을 숫자로 변환 후 datetime으로 변환
        tx['timestamp'] = pd.to_datetime(
            pd.to_numeric(tx['timestamp'], errors='coerce'),  # 숫자 변환 추가
            unit='s', 
            errors='coerce'
        )
        
        # NaT (Not a Time) 제거
        tx = tx[tx['timestamp'].notna()]
        
        # 날짜 필터링
        tx = tx[tx['timestamp'] < end_date]
        
        if len(tx) == 0:
            return None
        
        # 3. 주소 정규화
        tx['from'] = tx['from'].str.lower().str.strip()
        tx['to'] = tx['to'].str.lower().str.strip()
        
        # Null 및 빈 주소 제거
        tx = tx[
            (tx['from'].notna()) &
            (tx['to'].notna()) &
            (tx['from'] != '') &
            (tx['to'] != '') &
            (tx['from'] != '0x0000000000000000000000000000000000000000') &
            (tx['to'] != '0x0000000000000000000000000000000000000000')
        ]
        
        if len(tx) == 0:
            return None
        
        # 4. SNAP 그래프 생성 (최적화된 버전)
        G = build_snap_graph_optimized(tx[['from', 'to']])
        
        # 빈 그래프 체크
        if G.GetNodes() == 0:
            return None
        
        # 5. 메트릭 계산
        effective_diameter, clustering_coefficient = compute_metrics(G, sample_size)
        
        # 6. 결과 반환
        return {
            'Contract': addr,
            'Num_Nodes': G.GetNodes(),
            'Num_Edges': G.GetEdges(),
            'Effective_Diameter': effective_diameter,
            'Clustering_Coefficient': clustering_coefficient
        }
        
    except Exception as e:
        if debug:
            logger.error(f"Error processing {addr}: {e}")
        return None



def process_contract_wrapper(addr, chain, end_date, sample_size, debug):
    """멀티프로세싱 래퍼"""
    return process_single_contract(addr, chain, end_date, sample_size, debug)


def load_checkpoint(checkpoint_file):
    """체크포인트 로드"""
    if os.path.exists(checkpoint_file):
        try:
            df = pd.read_csv(checkpoint_file)
            processed = set(df['Contract'].values)
            logger.info(f"Loaded checkpoint: {len(processed)} contracts already processed")
            return df, processed
        except:
            return None, set()
    return None, set()


def save_checkpoint(results, checkpoint_file):
    """체크포인트 저장"""
    if results:
        df = pd.DataFrame(results)
        df.to_csv(checkpoint_file, index=False)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate SNAP graph properties (OPTIMIZED)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast mode with 8 workers
  python local_metrics/snap_properties.py --chain bsc --workers 8

  # High precision mode (slower)
  python local_metrics/snap_properties.py --chain bsc --sample-size 500 --workers 4

  # Resume from checkpoint
  python local_metrics/snap_properties.py --chain bsc --resume
        """
    )
    parser.add_argument('--chain', type=str, required=True,
                       help='Chain name (bsc, eth, polygon)')
    parser.add_argument('--end-date', type=str, default='2024-03-01',
                       help='End date for transactions (YYYY-MM-DD)')
    parser.add_argument('--sample-size', type=int, default=100,
                       help='BFS sample size for diameter (default: 100)')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of contracts for testing')
    
    args = parser.parse_args()
    
    if args.debug:
        logger.setLevel(logging.DEBUG)
    
    # 워커 수 설정
    if args.workers is None:
        args.workers = max(1, cpu_count() - 1)
    
    # 날짜 파싱
    end_date = pd.Timestamp(args.end_date)
    
    # 출력 디렉토리 생성
    output_dir = '../../_data/results/analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 결과 파일명
    output_file = os.path.join(
        output_dir, 
        f'{args.chain}_snap_metrics_labels.csv'
        # f'{args.chain}_advanced_metrics_labels.csv'
    )
    checkpoint_file = output_file.replace('.csv', '_checkpoint.csv')
    
    logger.info("="*70)
    logger.info(f"SNAP Properties Analysis (OPTIMIZED) - {args.chain.upper()}")
    logger.info("="*70)
    logger.info(f"End date: {args.end_date}")
    logger.info(f"BFS sample size: {args.sample_size}")
    logger.info(f"Parallel workers: {args.workers}")
    logger.info(f"Resume mode: {args.resume}")
    logger.info("="*70)
    
    # 1. 레이블 로드
    try:
        labels_df = pd.read_csv('../../_data/dataset/labels.csv')
        chain_labels = labels_df.query('Chain == @args.chain')
        contracts = list(chain_labels['Contract'].values)
        
        if args.limit:
            contracts = contracts[:args.limit]
        
        logger.info(f"Total contracts to process: {len(contracts)}")
        
    except FileNotFoundError:
        logger.error("Labels file not found: ../../_data/dataset/labels.csv")
        return
    
    # 2. 체크포인트 로드
    checkpoint_df, processed_contracts = load_checkpoint(checkpoint_file) if args.resume else (None, set())
    
    # 미처리 컨트랙트만 필터링
    if args.resume and processed_contracts:
        contracts = [c for c in contracts if c not in processed_contracts]
        logger.info(f"Remaining contracts: {len(contracts)}")
    
    if len(contracts) == 0:
        logger.info("All contracts already processed!")
        return
    
    # 3. 시작 시간 기록
    start_time = time.time()
    
    # 4. 병렬 처리
    logger.info(f"\nStarting parallel processing with {args.workers} workers...")
    
    process_func = partial(
        process_contract_wrapper,
        chain=args.chain,
        end_date=end_date,
        sample_size=args.sample_size,
        debug=args.debug
    )
    
    results = []
    if checkpoint_df is not None:
        results = checkpoint_df.to_dict('records')
    
    # 배치 처리 (중간 저장용)
    batch_size = 100
    total_processed = 0
    
    with Pool(processes=args.workers) as pool:
        for i, result in enumerate(tqdm(
            pool.imap_unordered(process_func, contracts),
            total=len(contracts),
            desc=f"Processing {args.chain}"
        )):
            if result:
                results.append(result)
            
            total_processed += 1
            
            # 100개마다 체크포인트 저장
            if total_processed % batch_size == 0:
                save_checkpoint(results, checkpoint_file)
    
    # 최종 저장
    elapsed_time = time.time() - start_time
    
    # 5. 결과 저장
    logger.info("\n" + "="*70)
    logger.info("Processing Summary")
    logger.info("="*70)
    logger.info(f"Total contracts processed: {len(contracts)}")
    logger.info(f"Successful extractions: {len(results) - len(processed_contracts)}")
    logger.info(f"Failed extractions: {len(contracts) - (len(results) - len(processed_contracts))}")
    logger.info(f"Total time: {elapsed_time/3600:.2f} hours ({elapsed_time/60:.2f} minutes)")
    logger.info(f"Average time per contract: {elapsed_time/len(contracts):.2f} seconds")
    
    if results:
        df = pd.DataFrame(results)
        
        # 컬럼 순서 정렬
        cols = ['Contract', 'Num_Nodes', 'Num_Edges', 
                'Effective_Diameter', 'Clustering_Coefficient']
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
        
        df.to_csv(output_file, index=False)
        logger.info(f"\n✓ Results saved to: {output_file}")
        
        # 체크포인트 파일 삭제
        if os.path.exists(checkpoint_file):
            os.remove(checkpoint_file)
            logger.info(f"✓ Checkpoint file removed")
        
        # 통계
        logger.info("\n" + "="*70)
        logger.info("Statistics")
        logger.info("="*70)
        logger.info(f"Mean nodes: {df['Num_Nodes'].mean():.2f}")
        logger.info(f"Mean edges: {df['Num_Edges'].mean():.2f}")
        logger.info(f"Mean diameter: {df['Effective_Diameter'].mean():.2f}")
        logger.info(f"Mean clustering: {df['Clustering_Coefficient'].mean():.6f}")
        
        # 상위 그래프
        logger.info("\nTop 10 largest graphs:")
        top10 = df.nlargest(10, 'Num_Nodes')[
            ['Contract', 'Num_Nodes', 'Num_Edges', 'Effective_Diameter']
        ]
        print(top10.to_string(index=False))
        
    else:
        logger.error("\n❌ No results extracted!")


if __name__ == "__main__":
    main()
