"""
NetworkX + SNAP Hybrid Graph Properties Analysis
- Multiprocessing for parallel file processing
- Uses NetworkX for Density, Reciprocity, Assortativity
- Uses SNAP for Effective_Diameter, Clustering_Coefficient (Ensures true float outputs)
"""

import pandas as pd
import networkx as nx
import snap
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


def get_graph_properties(file_path, skip_diameter=False, max_nodes_for_diameter=2000, debug=False):
    try:
        # 1. 데이터 로드
        df = pd.read_csv(
            file_path, 
            dtype=str, 
            low_memory=False,
            usecols=lambda x: x in ['from', 'to', 'from_address', 'to_address']
        )
        
        if df.empty:
            return None

        from_col = next((col for col in ['from', 'from_address', 'sender', 'fromAddress'] if col in df.columns), None)
        to_col = next((col for col in ['to', 'to_address', 'receiver', 'toAddress'] if col in df.columns), None)
        
        if not from_col or not to_col:
            return None
        
        # 주소 정규화
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

        # ---------------------------------------------------------
        # [Phase 1] NetworkX 그래프 생성 및 지표 계산
        # ---------------------------------------------------------
        G_nx = nx.from_pandas_edgelist(
            df, source=from_col, target=to_col, create_using=nx.DiGraph() 
        )
        
        num_nodes = G_nx.number_of_nodes()
        num_edges = G_nx.number_of_edges()
        
        if num_nodes == 0:
            return None

        density = float(nx.density(G_nx))

        try:
            reciprocity = float(nx.reciprocity(G_nx))
        except:
            reciprocity = 0.0

        try:
            assortativity = float(nx.degree_assortativity_coefficient(G_nx))
            if np.isnan(assortativity):
                assortativity = 0.0
        except:
            assortativity = 0.0

        # ---------------------------------------------------------
        # [Phase 2] SNAP 그래프 생성 및 지표 계산 (Float 보장)
        # ---------------------------------------------------------
        G_snap = snap.TNGraph.New()
        all_addresses = pd.concat([df[from_col], df[to_col]]).unique()
        node_dict = {addr: idx for idx, addr in enumerate(all_addresses)}
        
        # SNAP 노드 추가
        for node_id in range(len(all_addresses)):
            G_snap.AddNode(node_id)
        
        # SNAP 엣지 추가 (벡터화로 빠르게 처리)
        src_nodes = df[from_col].map(node_dict).values
        dst_nodes = df[to_col].map(node_dict).values
        for src, dst in zip(src_nodes, dst_nodes):
            G_snap.AddEdge(int(src), int(dst))

        # Clustering Coefficient (SNAP 사용)
        try:
            clustering_coefficient = float(snap.GetClustCf(G_snap, -1))
        except:
            clustering_coefficient = 0.0
            
        # Effective Diameter (SNAP 사용 - 통계적 근사값이라 온전한 Float이 나옴)
        diameter = 0.0
        if not skip_diameter and num_nodes > 0 and num_nodes <= max_nodes_for_diameter:
            try:
                # NTestNodes=10으로 90th percentile 도달거리를 근사 (snap_properties.py와 동일한 방식)
                diameter = float(snap.GetBfsEffDiam(G_snap, 10, False))
            except:
                diameter = 0.0

        return {
            'Contract': os.path.splitext(os.path.basename(file_path))[0],
            'Num_nodes': int(num_nodes),
            'Num_edges': int(num_edges),
            'Density': density,
            'Reciprocity': reciprocity,
            'Assortativity': assortativity,
            'Clustering_Coefficient': clustering_coefficient, # SNAP 값 매핑
            'Effective_Diameter': diameter                   # SNAP 값 매핑
        }

    except Exception as e:
        if debug:
            logger.error(f"Error processing {os.path.basename(file_path)}: {e}")
        return None


def process_file_wrapper(file_path, skip_diameter, max_nodes_for_diameter, debug):
    return get_graph_properties(file_path, skip_diameter, max_nodes_for_diameter, debug)


def main():
    parser = argparse.ArgumentParser(description='Calculate Hybrid Graph Properties (NetworkX + SNAP)')
    parser.add_argument('--data_dir', type=str, default='', help='Directory containing graph csv files')
    parser.add_argument('--output_dir', type=str, default='../../_data/results/analysis', help='Directory to save results')
    parser.add_argument('--chain', type=str, required=True, help='Chain name for output file')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files for testing')
    parser.add_argument('--skip-diameter', action='store_true', help='Skip diameter calculation')
    parser.add_argument('--max-diameter-nodes', type=int, default=2000, help='Max nodes for diameter calculation')
    parser.add_argument('--workers', type=int, default=8, help='Number of parallel workers')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    if args.data_dir == '':
        args.data_dir = f'../../_data/dataset/transactions/{args.chain}'
    
    if args.debug:
        logger.setLevel(logging.DEBUG)

    if args.workers is None:
        args.workers = max(1, cpu_count() - 1)
    
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory not found: {args.data_dir}")
        return
    
    files = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.csv')]
    
    if args.limit:
        files = files[:args.limit]
    
    logger.info("="*70)
    logger.info(f"Hybrid NetworkX+SNAP Properties Analysis - {args.chain.upper()}")
    
    start_time = time.time()
    logger.info(f"\nStarting parallel processing with {args.workers} workers...")
    
    process_func = partial(
        process_file_wrapper,
        skip_diameter=args.skip_diameter,
        max_nodes_for_diameter=args.max_diameter_nodes,
        debug=args.debug
    )
    
    results = []
    with Pool(processes=args.workers) as pool:
        for result in tqdm(pool.imap_unordered(process_func, files), total=len(files), desc=f"Processing {args.chain}"):
            if result:
                results.append(result)
    
    elapsed_time = time.time() - start_time
    
    if results:
        df_res = pd.DataFrame(results)
        
        cols = ['Contract', 'Num_nodes', 'Num_edges', 'Density', 
                'Reciprocity', 'Assortativity', 'Clustering_Coefficient', 
                'Effective_Diameter']
        cols = [c for c in cols if c in df_res.columns]
        df_res = df_res[cols]

        # Float 컬럼 강제화
        float_cols = ['Density', 'Reciprocity', 'Assortativity', 'Clustering_Coefficient', 'Effective_Diameter']
        for col in float_cols:
            if col in df_res.columns:
                df_res[col] = df_res[col].astype(float)

        output_path = os.path.join(args.output_dir, f'{args.chain}_basic_metrics.csv')
        df_res.to_csv(output_path, index=False)
        
        logger.info(f"\n✓ Results saved to: {output_path}")
        
    else:
        logger.error("\n❌ No results extracted!")

if __name__ == "__main__":
    main()