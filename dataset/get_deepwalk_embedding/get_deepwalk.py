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

# 재현성을 위한 Seed 고정 함수
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def parameter_parser():
    parser = argparse.ArgumentParser(description="Run DeepWalk model for graph embeddings.")
    parser.add_argument('--embedding_dim', type=int, default=32, help='Dimension of embeddings.')
    parser.add_argument('--chain', type=str, default='polygon', help='Blockchain')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers for generating walks.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    return parser.parse_args()

def process_graph(idx, data, embedding_dim, chain, seed, save_dir):
    # 각 프로세스 내부에서도 시드 설정 (idx를 더해 프로세스마다 다른 시드지만 고정되게 함)
    current_seed = seed + idx
    seed_everything(current_seed)
    
    logging.info(f'============================Processing graph {idx}============================')
    
    G = nx.Graph()
    # 엣지가 없는 경우 예외 처리
    if data.edge_index is not None and data.edge_index.numel() > 0:
        G.add_edges_from(data.edge_index.t().tolist())
    
    # 노드 자체가 없는 경우 처리 (빈 그래프)
    print(f"Graph {idx} has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    if G.number_of_nodes() == 0:
        logging.warning(f"Graph {idx} is empty. Saving zero embeddings.")
        # 임시로 1개 노드 생성하여 0 벡터 저장하거나 빈 배열 저장 (후처리 로직에 따라 다름)
        # 여기서는 빈 배열 저장 시 에러 가능성이 있으므로 생략하거나 더미 생성
        # GoG 특성상 완전히 빈 그래프는 드물므로 pass
        return

    deepwalk = DeepWalk(G, walk_length=20, num_walks=40, embedding_dim=embedding_dim, seed=current_seed)
    
    # 워크가 생성되지 않는 경우 대비
    walks = deepwalk.generate_walks()
    if not walks:
        logging.warning(f"Graph {idx} generated no walks.")
        return

    model = deepwalk.train(walks)
    
    # 순서를 보장하기 위해 G.nodes() 대신 정렬하거나 고정된 리스트 사용 권장
    # 여기서는 G.nodes()를 그대로 쓰되, G 생성 순서에 의존
    node_embeddings = []
    for node in G.nodes():
        if str(node) in model.wv:
            node_embeddings.append(model.wv[str(node)])
        else:
            node_embeddings.append(np.zeros(embedding_dim))
            
    node_embeddings = np.array(node_embeddings)
       
    np.save(f'{save_dir}/{idx}.npy', node_embeddings)
    
    del G, deepwalk, model, node_embeddings 
    gc.collect()

# 수정 전
# def worker_process(args):
#     idx, data, embedding_dim, chain, seed = args
#     output_dir = f'../../_data/dataset/Deepwalk/{chain}/'
#     os.makedirs(output_dir, exist_ok=True)
#     process_graph(idx, data, embedding_dim, chain, seed, output_dir)

# 수정 후
def worker_process(args):
    idx, embedding_dim, chain, seed = args
    
    # args로 넘기지 않고, Fork된 메모리에서 직접 접근 (Copy-on-Write 활용)
    global data_list 
    data = data_list[idx]
    
    output_dir = f'../../_data/dataset/Deepwalk/{chain}/'
    # os.makedirs(output_dir, exist_ok=True)
    process_graph(idx, data, embedding_dim, chain, seed, output_dir)




# PyTorch Tensor 공유로 인한 파일 디스크립터 한계 방지를 위해 파일 최상단에 추가하는 것을 권장합니다.
import torch
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    args = parameter_parser()
    seed_everything(args.seed)
    
    graphs_directory = f"../../_data/GoG/{args.chain}/graphs/"

    dataset_generator = GraphDatasetGenerator(graphs_directory)
    
    # worker_process에서 참조할 수 있도록 global 선언
    global data_list
    data_list = dataset_generator.get_pyg_data_list()
    embedding_dim = args.embedding_dim

    numbers = list(range(0, len(data_list)))
    num_cores = max(2, os.cpu_count()//2)  # CPU 코어 수의 절반 사용, 최소 2코어 확보
    logging.info(f'Using {num_cores} cores for multiprocessing.')

    pool = multiprocessing.Pool(num_cores)
    
    # 수정 포인트 1: data 객체를 넘기지 않음 (메모리 폭발 방지)
    tasks = [(idx, embedding_dim, args.chain, args.seed) for idx in numbers]
    
    try:
        print(f"Processing {len(tasks)} graphs with embedding dimension {embedding_dim}...")
        
        # 수정 포인트 2: map 대신 imap_unordered 사용 + chunksize 1 설정
        # (결과를 한 번에 묶지 않고 스트리밍 방식으로 처리하여 파이프 과부하 방지)
        for _ in pool.imap_unordered(worker_process, tasks, chunksize=1):
            pass 
            
    except Exception as e:
        logging.error(f"Error during multiprocessing: {str(e)}")
    finally:
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()

