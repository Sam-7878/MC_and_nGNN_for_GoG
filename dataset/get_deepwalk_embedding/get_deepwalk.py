import numpy as np
import gc  
from deepwalk import DeepWalk
import networkx as nx
import logging
import multiprocessing  
import argparse
import os
import random
import warnings
import json
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore", message=".*pyg-lib.*")

logging.basicConfig(level=logging.INFO)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def parameter_parser():
    parser = argparse.ArgumentParser(description="Run DeepWalk model for graph embeddings.")
    parser.add_argument('--embedding_dim', type=int, default=32, help='Dimension of embeddings.')
    parser.add_argument('--chain', type=str, default='bsc', help='Blockchain')
    parser.add_argument('--workers', type=int, default=0, help='Number of parallel workers (0 = auto).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    return parser.parse_args()

def process_graph(idx, edges, embedding_dim, seed, save_dir):
    current_seed = seed + idx
    seed_everything(current_seed)
    
    G = nx.Graph()
    if edges:
        G.add_edges_from(edges)
    
    num_nodes = G.number_of_nodes()
    
    if num_nodes == 0:
        return

    # 거대 그래프 메모리 폭발 방지 (동적 파라미터 스케일링)
    walk_length = 20
    num_walks = 40
    
    if num_nodes > 100000:
        walk_length = 5
        num_walks = 2
        logging.warning(f"Graph {idx} is MASSIVE ({num_nodes} nodes). Drastically reducing walks to prevent OOM.")
    elif num_nodes > 30000:
        walk_length = 10
        num_walks = 5
        logging.warning(f"Graph {idx} is HUGE ({num_nodes} nodes). Reducing walks to prevent OOM.")
    elif num_nodes > 10000:
        walk_length = 10
        num_walks = 10
    
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
    
    # 1. 이어하기: 이미 생성된 파일은 즉시 패스
    if os.path.exists(f'{output_dir}/{idx}.npy'):
        return
    
    graph_path = f"../../_data/GoG/{chain}/graphs/{idx}.json"
    if not os.path.exists(graph_path):
        return
        
    # 2. 지연 로딩 (Lazy Loading): 워커가 실행될 때 비로소 JSON을 읽음
    # 이렇게 하면 메인 프로세스는 메모리를 전혀 쓰지 않음!
    with open(graph_path, 'r') as f:
        data = json.load(f)
        edges = data.get('edges', [])
        
    process_graph(idx, edges, embedding_dim, seed, output_dir)
    
    # 처리 후 즉시 메모리 해제
    del data, edges
    gc.collect()

def main():
    args = parameter_parser()
    seed_everything(args.seed)
    
    graphs_directory = Path(f"../../_data/GoG/{args.chain}/graphs/")
    output_dir = f'../../_data/dataset/Deepwalk/{args.chain}/'
    os.makedirs(output_dir, exist_ok=True)

    # 전체 데이터 개수를 JSON 파일 개수로 파악 (데이터를 메모리에 올리지 않음)
    json_files = list(graphs_directory.glob("*.json"))
    numbers = sorted([int(f.stem) for f in json_files])
    
    if args.workers > 0:
        num_cores = args.workers
    else:
        num_cores = max(2, os.cpu_count() // 2)
        
    logging.info(f'Using {num_cores} cores for multiprocessing (Lazy Loading Mode).')

    pool = multiprocessing.Pool(num_cores, maxtasksperchild=4)
    tasks = [(idx, args.embedding_dim, args.chain, args.seed) for idx in numbers]
    
    try:
        print(f"Processing {len(tasks)} graphs with embedding dimension {args.embedding_dim}...")
        
        # tqdm 적용
        for _ in tqdm(pool.imap_unordered(worker_process, tasks, chunksize=1), total=len(tasks)):
            pass 
            
    except Exception as e:
        logging.error(f"Error during multiprocessing: {str(e)}")
    finally:
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()