import numpy as np
import gc  
from utils import GraphDatasetGenerator
from deepwalk import DeepWalk
import networkx as nx
import logging
import multiprocessing  
import argparse
import os

logging.basicConfig(level=logging.INFO)

# Parameter Parser Configuration
def parameter_parser():
    """A method to parse up command line parameters."""
    parser = argparse.ArgumentParser(description="Run DeepWalk model for graph embeddings.")
    parser.add_argument('--embedding_dim', type=int, default=32, help='Dimension of embeddings.')
    parser.add_argument('--chain', type=str, default='polygon', help='Blockchain')
    parser.add_argument('--workers', type=int, default=4, help='Number of parallel workers for generating walks.')
    return parser.parse_args()

def process_graph(idx, data, embedding_dim, chain):
    logging.info(f'Processing graph {idx}')
    G = nx.Graph()
    G.add_edges_from(data.edge_index.t().tolist())
    deepwalk = DeepWalk(G, walk_length=20, num_walks=40, embedding_dim=embedding_dim)
    model = deepwalk.train(deepwalk.generate_walks())
    node_embeddings = np.array([model.wv[str(node)] for node in G.nodes()])
    np.save(f'../../_data/data/Deepwalk/{chain}/{idx}.npy', node_embeddings)
    del G, deepwalk, model, node_embeddings 
    gc.collect()

def worker_process(args):
    idx, data, embedding_dim, chain = args
    process_graph(idx, data, embedding_dim, chain)

def main():
    args = parameter_parser()
    
    # Setting the graph directory
    graphs_directory = f"../../_data/GoG/{args.chain}/"

    dataset_generator = GraphDatasetGenerator(graphs_directory)
    data_list = dataset_generator.get_pyg_data_list()
    embedding_dim = args.embedding_dim
    chain = args.chain

    os.makedirs(os.path.dirname(f'../../_data/data/Deepwalk/{chain}/'), exist_ok=True)

    numbers = list(range(0, len(data_list)))

    num_cores = max(2, multiprocessing.cpu_count() // 2)
    logging.info(f'Using {num_cores} cores.')

    pool = multiprocessing.Pool(num_cores)
    tasks = [(idx, data, embedding_dim, chain) for idx, data in enumerate(data_list) if idx in numbers]
    try:
        pool.map(worker_process, tasks)
    except Exception as e:
        logging.error(f"Error during multiprocessing: {str(e)}")
    finally:
        pool.close()
        pool.join()

if __name__ == "__main__":
    main()
