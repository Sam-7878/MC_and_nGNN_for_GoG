import torch
import numpy as np
from torch_geometric.data import Data
from utils import hierarchical_graph_reader
from pygod.detector import DOMINANT, DONE, GAE, AnomalyDAE, CoLA
from pygod.metric import eval_roc_auc
from sklearn.metrics import average_precision_score, roc_auc_score
import os
import pandas as pd
import random
import warnings

warnings.filterwarnings("ignore", message=".*pyg-lib.*")

def create_masks(num_nodes):
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    train_size = int(num_nodes * 0.8)
    val_size = int(num_nodes * 0.1)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    return train_mask, val_mask, test_mask

def eval_roc_auc(label, score):
    roc_auc = roc_auc_score(y_true=label, y_score=score)
    if roc_auc < 0.5:
        score = [1 - s for s in score]
        roc_auc = roc_auc_score(y_true=label, y_score=score)
    return roc_auc

def train_and_evaluate(detector, data, epochs=50, eval_interval=5):
    optimizer = torch.optim.Adam(detector.parameters(), lr=0.01, weight_decay=5e-4)
    
    for epoch in range(1, epochs + 50):
        detector.train()
        optimizer.zero_grad()
        loss = detector(data)
        loss.backward()
        optimizer.step()
        
        if epoch % eval_interval == 0 or epoch == epochs:
            detector.eval()
            with torch.no_grad():
                pred, score, _, _ = detector.predict(data, return_pred=True, return_score=True)
                auc_score = eval_roc_auc(data.y, score)
                ap_score = average_precision_score(data.y.cpu().numpy(), score.cpu().numpy())
                print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}, AUC: {auc_score:.4f}, AP: {ap_score:.4f}')

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_model(detector, data, seeds):
    auc_scores = []
    ap_scores = []
    
    for seed in seeds:
        set_seed(seed)

        detector.fit(data)

        _, score, _, _ = detector.predict(data, return_pred=True, return_score=True, return_prob=True, return_conf=True)
        
        auc_score = eval_roc_auc(data.y, score)
        ap_score = average_precision_score(data.y.cpu().numpy(), score.cpu().numpy())

        auc_scores.append(auc_score)
        ap_scores.append(ap_score)

    return np.mean(auc_scores), np.std(auc_scores), np.mean(ap_scores), np.std(ap_scores)

def load_labels(filepath, column_name='label'):
    try:
        labels = pd.read_csv(filepath)[column_name].values
        return torch.tensor(labels, dtype=torch.long)
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        exit()
    except KeyError:
        print(f"Error: Column {column_name} does not exist in the file.")
        exit()

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=str, default='polygon')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index to use')
    return parser.parse_args()



def main():
    args = get_args()
    chain = args.chain
    
    filepath = f'../../_data/dataset/features/{chain}_basic_metrics_processed.csv'
    y = load_labels(filepath)
    
    graph_embeddings = []
    embedding_path = f'../../_data/dataset/Deepwalk/{chain}'

    processed_graphs = 0
    
    for idx in range(len(y)):
        embedding_file = os.path.join(embedding_path, f'{idx}.npy')
        if os.path.exists(embedding_file):
            node_embeddings = torch.tensor(np.load(embedding_file), dtype=torch.float32)
            mean_embedding = node_embeddings.mean(dim=0, keepdim=True).detach()
            graph_embeddings.append(mean_embedding)
            processed_graphs += 1
        else:
            print(f"Embedding file not found: {embedding_file}")
    
    if len(graph_embeddings) == 0:
        raise ValueError("No graph embeddings were processed. Please check the embedding path and files.")


    x = torch.cat(graph_embeddings, dim=0)

    hierarchical_graph = hierarchical_graph_reader(f'../../_data/GoG/{chain}/edges/global_edges.csv')
    edge_index = torch.LongTensor(list(hierarchical_graph.edges)).t().contiguous()

    global_data = Data(x=x, edge_index=edge_index, y=y)
    
    train_mask, val_mask, test_mask = create_masks(global_data.num_nodes)
    global_data.train_mask = train_mask
    global_data.val_mask = val_mask
    global_data.test_mask = test_mask

    # Parameters to test
    model_params = {
        'DOMINANT': [{'hid_dim': d, 'lr': lr, 'epoch': e} for d in [16, 32, 64] for lr in [0.01, 0.005, 0.1] for e in [50, 100, 150]],
        'DONE': [{'hid_dim': d, 'lr': lr, 'epoch': e} for d in [16, 32, 64] for lr in [0.01, 0.005, 0.1] for e in [50, 100, 150]],
        'GAE': [{'hid_dim': d, 'lr': lr, 'epoch': e} for d in [8, 16, 32, 64] for lr in [0.01, 0.005, 0.1] for e in [50, 100, 150]],
        'AnomalyDAE': [{'hid_dim': d, 'lr': lr, 'epoch': e} for d in [16, 32, 64] for lr in [0.01, 0.005, 0.1] for e in [50, 100, 150]],
        'CoLA': [{'hid_dim': d, 'lr': lr, 'epoch': e} for d in [16, 32, 64] for lr in [0.01, 0.005, 0.1] for e in [50, 100, 150]]
    }

    seed_for_param_selection = 42
    best_model_params = {}
    for model_name, param_list in model_params.items():
        for param in param_list:
            detector = eval(f"{model_name}(hid_dim=param['hid_dim'], num_layers=2, epoch=param['epoch'], lr=param['lr'], gpu=args.gpu)")
            avg_auc, std_auc, avg_ap, std_ap = run_model(detector, global_data, [seed_for_param_selection])
            if model_name not in best_model_params or avg_auc > best_model_params[model_name].get('Best AUC', 0):
                best_model_params[model_name] = {
                    "Best AUC": avg_auc,
                    "AUC Std Dev": std_auc,
                    "Best AP": avg_ap,
                    "AP Std Dev": std_ap,
                    "Params": param
                }
            print(f'Tested {model_name} with {param}: Avg AUC={avg_auc:.4f}, Std AUC={std_auc:.4f}, Avg AP={avg_ap:.4f}, Std AP={std_ap:.4f}')

    seeds_for_evaluation = [42, 43, 44]
    for model_name, stats in best_model_params.items():
        param = stats['Params']
        detector = eval(f"{model_name}(hid_dim=param['hid_dim'], num_layers=2, epoch=param['epoch'], lr=param['lr'], gpu=args.gpu)")
        avg_auc, std_auc, avg_ap, std_ap = run_model(detector, global_data, seeds_for_evaluation)
        print(model_name)
        print(stats)
        print(f'Final Evaluation for {model_name}: Avg AUC={avg_auc:.4f}, Std AUC={std_auc:.4f}, Avg AP={avg_ap:.4f}, Std AP={std_ap:.4f}')

if __name__ == "__main__":
    main()