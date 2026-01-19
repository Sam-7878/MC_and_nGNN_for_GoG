"""
NetworkX Graph Properties Analysis - OPTIMIZED VERSION
- Multiprocessing for parallel file processing
- Efficient diameter calculation (sampling for large graphs)
- Progress tracking with time estimation
"""

import pandas as pd
import networkx as nx
import os
import argparse
from tqdm import tqdm
import warnings
import numpy as np
import logging
from multiprocessing import Pool, cpu_count
from functools import partial
import time

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


def get_graph_properties(file_path, skip_diameter=False, max_nodes_for_diameter=1000, debug=False):
    """
    하나의 CSV 파일을 읽어 그래프를 생성하고,
    Local Metrics(NetworkX Properties)를 계산하여 반환합니다.
    
    Args:
        file_path: CSV 파일 경로
        skip_diameter: 직경 계산 건너뛰기 (성능 향상)
        max_nodes_for_diameter: 직경 계산할 최대 노드 수
        debug: 디버그 모드
    
    Returns:
        dict: 그래프 속성 딕셔너리 또는 None
    """
    try:
        # 1. 데이터 로드 (필요한 컬럼만)
        df = pd.read_csv(
            file_path, 
            dtype=str, 
            low_memory=False,
            usecols=lambda x: x in ['from', 'to', 'from_address', 'to_address']  # 필요한 컬럼만
        )
        
        if df.empty:
            return None

        # 컬럼 이름 찾기
        from_col = None
        to_col = None
        
        for col in ['from', 'from_address', 'sender', 'fromAddress']:
            if col in df.columns:
                from_col = col
                break
        
        for col in ['to', 'to_address', 'receiver', 'toAddress']:
            if col in df.columns:
                to_col = col
                break
        
        if not from_col or not to_col:
            return None
        
        # 주소 정규화 및 필터링
        df[from_col] = df[from_col].str.lower().str.strip()
        df[to_col] = df[to_col].str.lower().str.strip()
        
        df = df[
            (df[from_col].notna()) &
            (df[to_col].notna()) &
            (df[from_col] != '') &
            (df[to_col] != '') &
            (df[from_col] != '0x0000000000000000000000000000000000000000') &
            (df[to_col] != '0x0000000000000000000000000000000000000000')
        ]
        
        if len(df) == 0:
            return None

        # 2. 그래프 생성
        G = nx.from_pandas_edgelist(
            df, 
            source=from_col,
            target=to_col,
            create_using=nx.DiGraph() 
        )
        
        if G.number_of_nodes() == 0:
            return None

        # 3. 기본 지표 계산 (빠른 것들)
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        density = nx.density(G)

        # 4. Reciprocity (빠름)
        try:
            reciprocity = nx.reciprocity(G)
        except:
            reciprocity = 0.0

        # 5. Assortativity (빠름)
        try:
            assortativity = nx.degree_assortativity_coefficient(G)
            if np.isnan(assortativity):
                assortativity = 0.0
        except:
            assortativity = 0.0

        # 6. Clustering Coefficient (중간 속도)
        try:
            # 큰 그래프는 샘플링
            if num_nodes < 5000:
                G_undirected = G.to_undirected()
                clustering_coefficient = nx.average_clustering(G_undirected)
            else:
                # 샘플링: 랜덤 1000개 노드만 계산
                G_undirected = G.to_undirected()
                sample_nodes = np.random.choice(
                    list(G_undirected.nodes()), 
                    size=min(1000, num_nodes), 
                    replace=False
                )
                clustering_coefficient = nx.average_clustering(
                    G_undirected, 
                    nodes=sample_nodes
                )
        except:
            clustering_coefficient = 0.0
            
        # 7. Effective Diameter (매우 느림 - 옵션화)
        diameter = 0
        if not skip_diameter and num_nodes > 0 and num_nodes <= max_nodes_for_diameter:
            try:
                G_undirected = G.to_undirected()
                largest_cc = max(nx.connected_components(G_undirected), key=len)
                
                # 가장 큰 컴포넌트가 전체의 10% 이상인 경우만 계산
                if len(largest_cc) > num_nodes * 0.1:
                    subgraph = G_undirected.subgraph(largest_cc)
                    # 노드가 많으면 근사값 사용
                    if len(largest_cc) > 500:
                        # 샘플링으로 근사 직경 계산
                        sample_size = min(100, len(largest_cc))
                        sample_nodes = np.random.choice(
                            list(largest_cc), 
                            size=sample_size, 
                            replace=False
                        )
                        paths = []
                        for node in sample_nodes:
                            lengths = nx.single_source_shortest_path_length(
                                subgraph, node
                            )
                            if lengths:
                                paths.append(max(lengths.values()))
                        diameter = int(np.mean(paths)) if paths else 0
                    else:
                        diameter = nx.diameter(subgraph)
            except:
                diameter = 0

        return {
            'Contract': os.path.splitext(os.path.basename(file_path))[0],
            'Num_nodes': num_nodes,
            'Num_edges': num_edges,
            'Density': density,
            'Reciprocity': reciprocity,
            'Assortativity': assortativity,
            'Clustering_Coefficient': clustering_coefficient,
            'Effective_Diameter': diameter
        }

    except Exception as e:
        if debug:
            logger.error(f"Error processing {os.path.basename(file_path)}: {e}")
        return None


def process_file_wrapper(file_path, skip_diameter, max_nodes_for_diameter, debug):
    """멀티프로세싱을 위한 래퍼 함수"""
    return get_graph_properties(file_path, skip_diameter, max_nodes_for_diameter, debug)


def main():
    parser = argparse.ArgumentParser(
        description='Calculate NetworkX graph properties (OPTIMIZED)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fast mode (skip diameter, use all CPU cores)
  python local_metrics/nx_properties.py --chain bsc --skip-diameter --workers 8

  # Balanced mode (calculate diameter for small graphs only)
  python local_metrics/nx_properties.py --chain bsc --max-diameter-nodes 500 --workers 4

  # Full mode (slower but complete)
  python local_metrics/nx_properties.py --chain bsc --max-diameter-nodes 2000 --workers 2
        """
    )
    parser.add_argument('--data_dir', type=str, default='', 
                       help='Directory containing graph csv files')
    parser.add_argument('--output_dir', type=str, 
                       default='../../_data/results/analysis', 
                       help='Directory to save results')
    parser.add_argument('--chain', type=str, required=True,
                       help='Chain name for output file')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of files for testing')
    parser.add_argument('--skip-diameter', action='store_true',
                       help='Skip diameter calculation (much faster)')
    parser.add_argument('--max-diameter-nodes', type=int, default=2000,
                       help='Maximum nodes for diameter calculation (default: 2000)')
    parser.add_argument('--workers', type=int, default=8,
                       help='Number of parallel workers (default: CPU count - 1)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.data_dir == '':
        args.data_dir = f'../../_data/dataset/transactions/{args.chain}'
    
    if args.debug:
        logger.setLevel(logging.DEBUG)

    # 워커 수 설정
    if args.workers is None:
        args.workers = max(1, cpu_count() - 1)  # 1개 코어는 시스템용으로 남김
    
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        return
    
    files = [os.path.join(args.data_dir, f) 
             for f in os.listdir(args.data_dir) 
             if f.endswith('.csv')]
    
    if args.limit:
        files = files[:args.limit]
    
    logger.info("="*70)
    logger.info(f"NetworkX Properties Analysis (OPTIMIZED) - {args.chain.upper()}")
    logger.info("="*70)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Total files: {len(files)}")
    logger.info(f"Parallel workers: {args.workers}")
    logger.info(f"Skip diameter: {args.skip_diameter}")
    if not args.skip_diameter:
        logger.info(f"Max nodes for diameter: {args.max_diameter_nodes}")
    logger.info("="*70)
    
    # 시작 시간 기록
    start_time = time.time()
    
    logger.info(f"\nStarting parallel processing with {args.workers} workers...")
    
    # 멀티프로세싱으로 병렬 처리
    process_func = partial(
        process_file_wrapper,
        skip_diameter=args.skip_diameter,
        max_nodes_for_diameter=args.max_diameter_nodes,
        debug=args.debug
    )
    
    results = []
    with Pool(processes=args.workers) as pool:
        # imap_unordered: 완료되는 대로 결과 반환 (순서 무관)
        for result in tqdm(
            pool.imap_unordered(process_func, files),
            total=len(files),
            desc=f"Processing {args.chain}"
        ):
            if result:
                results.append(result)
    
    # 종료 시간 기록
    elapsed_time = time.time() - start_time
    
    # 결과 출력
    logger.info("\n" + "="*70)
    logger.info("Processing Summary")
    logger.info("="*70)
    logger.info(f"Total files processed: {len(files)}")
    logger.info(f"Successful extractions: {len(results)}")
    logger.info(f"Failed extractions: {len(files) - len(results)}")
    logger.info(f"Success rate: {len(results)/len(files)*100:.2f}%")
    logger.info(f"Total time: {elapsed_time/3600:.2f} hours ({elapsed_time/60:.2f} minutes)")
    logger.info(f"Average time per file: {elapsed_time/len(files):.2f} seconds")
    
    if results:
        df_res = pd.DataFrame(results)
        
        cols = ['Contract', 'Num_nodes', 'Num_edges', 'Density', 
                'Reciprocity', 'Assortativity', 'Clustering_Coefficient', 
                'Effective_Diameter']
        cols = [c for c in cols if c in df_res.columns]
        df_res = df_res[cols]

        # 출력 파일명에 모드 표시
        mode_suffix = "_fast" if args.skip_diameter else f"_maxd{args.max_diameter_nodes}"
        output_path = os.path.join(
            args.output_dir, 
            f'{args.chain}_basic_metrics{mode_suffix}.csv'
        )
        df_res.to_csv(output_path, index=False)
        
        logger.info(f"\n✓ Results saved to: {output_path}")
        
        # 통계
        logger.info("\n" + "="*70)
        logger.info("Statistics")
        logger.info("="*70)
        logger.info(f"Mean nodes: {df_res['Num_nodes'].mean():.2f}")
        logger.info(f"Mean edges: {df_res['Num_edges'].mean():.2f}")
        logger.info(f"Mean density: {df_res['Density'].mean():.6f}")
        logger.info(f"Mean reciprocity: {df_res['Reciprocity'].mean():.6f}")
        logger.info(f"Mean clustering: {df_res['Clustering_Coefficient'].mean():.6f}")
        
        if not args.skip_diameter:
            non_zero = df_res[df_res['Effective_Diameter'] > 0]
            if len(non_zero) > 0:
                logger.info(f"Mean diameter (non-zero): {non_zero['Effective_Diameter'].mean():.2f}")
        
        logger.info("\nTop 10 largest graphs:")
        top10 = df_res.nlargest(10, 'Num_nodes')[
            ['Contract', 'Num_nodes', 'Num_edges', 'Density']
        ]
        print(top10.to_string(index=False))
        
    else:
        logger.error("\n❌ No results extracted!")

if __name__ == "__main__":
    main()
