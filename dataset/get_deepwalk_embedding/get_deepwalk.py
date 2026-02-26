from itertools import chain

import numpy as np
import gc  
from utils import GraphDatasetGenerator
from deepwalk import DeepWalk
import networkx as nx
import logging
import multiprocessing  
import argparse
import os
import random
import warnings

warnings.filterwarnings("ignore", message=".*pyg-lib.*")

logging.basicConfig(level=logging.INFO)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def parameter_parser():
    parser = argparse.ArgumentParser(description="Run DeepWalk model for graph embeddings.")
    parser.add_argument('--embedding_dim', type=int, default=32, help='Dimension of embeddings.')
    parser.add_argument('--chain', type=str, default='polygon', help='Blockchain')
    parser.add_argument('--workers', type=int, default=0, help='Number of parallel workers (0 = auto).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    return parser.parse_args()

def process_graph(idx, data, embedding_dim, chain, seed, save_dir):
    current_seed = seed + idx
    seed_everything(current_seed)
    
    G = nx.Graph()
    if data.edge_index is not None and data.edge_index.numel() > 0:
        G.add_edges_from(data.edge_index.t().tolist())
    
    num_nodes = G.number_of_nodes()
    
    if num_nodes == 0:
        return

    # =====================================================================
    # ✅ [핵심 추가] 거대 그래프 메모리 폭발 방지 (동적 파라미터 스케일링)
    # =====================================================================
    walk_length = 20
    num_walks = 40
    
    if num_nodes > 100000:
        walk_length = 4
        num_walks = 2
        logging.warning(f"Graph {idx} is MASSIVE ({num_nodes} nodes). Drastically reducing walks to prevent OOM.")
    elif num_nodes > 30000:
        walk_length = 10
        num_walks = 5
        logging.warning(f"Graph {idx} is HUGE ({num_nodes} nodes). Reducing walks to prevent OOM.")
    elif num_nodes > 10000:
        walk_length = 10
        num_walks = 10
        logging.info(f"Graph {idx} is large ({num_nodes} nodes). Adjusting walks.")
    else:
        logging.info(f'Processing graph {idx} ({num_nodes} nodes, {G.number_of_edges()} edges)')

    print(f"Graph {idx}: num_n={num_nodes}, walk_length={walk_length}, num_walks={num_walks}")

    try:
        deepwalk = DeepWalk(G, walk_length=walk_length, num_walks=num_walks, embedding_dim=embedding_dim, seed=current_seed)
        walks = deepwalk.generate_walks()
        
        if not walks:
            return

        model = deepwalk.train(walks)
        
        node_embeddings = []
        for node in G.nodes():
            if str(node) in model.wv:
                node_embeddings.append(model.wv[str(node)])
            else:
                node_embeddings.append(np.zeros(embedding_dim))
                
        node_embeddings = np.array(node_embeddings)
        np.save(f'{save_dir}/{idx}.npy', node_embeddings)
        
    except MemoryError:
        # 워커가 죽는 것을 막고 0벡터로 저장 후 스킵
        logging.error(f"MemoryError on graph {idx} ({num_nodes} nodes). Saving zero embeddings instead of crashing.")
        node_embeddings = np.zeros((num_nodes, embedding_dim))
        np.save(f'{save_dir}/{idx}.npy', node_embeddings)
    except Exception as e:
        logging.error(f"Error on graph {idx}: {str(e)}")
        
    finally:
        del G
        if 'deepwalk' in locals(): del deepwalk
        if 'model' in locals(): del model
        if 'walks' in locals(): del walks
        if 'node_embeddings' in locals(): del node_embeddings
        gc.collect()

def worker_process(args):
    idx, embedding_dim, chain, seed = args
    output_dir = f'../../_data/dataset/Deepwalk/{chain}/'
    
    # 이어하기: 이미 생성된 파일은 1초 만에 패스
    if os.path.exists(f'{output_dir}/{idx}.npy'):
        return
    
    global data_list 
    data = data_list[idx]
    process_graph(idx, data, embedding_dim, chain, seed, output_dir)


import torch
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    args = parameter_parser()
    seed_everything(args.seed)
    
    graphs_directory = f"../../_data/GoG/{args.chain}/graphs/"
    output_dir = f'../../_data/dataset/Deepwalk/{args.chain}/'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading graphs from {graphs_directory}...")

    dataset_generator = GraphDatasetGenerator(graphs_directory)

    print("Generating PyG data list...")
    
    global data_list
    data_list = dataset_generator.get_pyg_data_list()
    embedding_dim = args.embedding_dim

    data_length = len(data_list)
    numbers = list(range(0, data_length))
    print(f"Total graphs to process: {data_length}")
    
    # ✅ 사용자가 --workers를 지정하면 그 값을 우선 사용하도록 수정
    if args.workers > 0:
        num_cores = args.workers
    else:
        num_cores = max(2, os.cpu_count() // 2)
        
    logging.info(f'Using {num_cores} cores for multiprocessing.')

    # OOM 방지용 청크 초기화 적용
    pool = multiprocessing.Pool(num_cores, maxtasksperchild=2)
    tasks = [(idx, embedding_dim, args.chain, args.seed) for idx in numbers]
    
    try:
        from tqdm import tqdm
        print(f"Processing {len(tasks)} graphs with embedding dimension {embedding_dim}...")
        
        for _ in tqdm(pool.imap_unordered(worker_process, tasks, chunksize=1), total=len(tasks)):
            pass 
            
    except Exception as e:
        logging.error(f"Error during multiprocessing: {str(e)}")
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()