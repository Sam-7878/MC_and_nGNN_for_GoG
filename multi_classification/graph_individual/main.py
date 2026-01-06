import os
import argparse
import torch
import numpy as np
from train import train, evaluate
from model import GCN, GIN, GraphSAGE, GAT, ResidualGCN
from dataloader import TransactionDataset
from torch.utils.data import Dataset, Subset, SubsetRandomSampler, random_split
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from collections import Counter
from utils import * 

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--chain', type=str, default='polygon')
    parser.add_argument('--model', type=str, default='GCN', choices=['GCN', 'GIN', 'GraphSAGE', 'GAT', 'ResidualGCN'], help="Type of GNN model to use.")
    parser.add_argument("--device", default="cuda:1", help='CUDA device to use (default: 1)')
    parser.add_argument('--classification_type', type=str, default='multiclass', choices=['binary', 'multiclass'], help='Type of classification task: binary or multiclass')
    parser.add_argument('--split_type', type=str, default='temporal', choices=['random', 'temporal'], help='Type of split: random or temporal')
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--features', default=[0, 1, 2], nargs='+', help='List of features to include')
    return parser.parse_args()

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    seeds = [42, 43, 44]
    metrics = {'train': [], 'test': []}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        dataset = TransactionDataset(root=f'../../_data/data/GCN/{args.chain}')
        labels = [dataset.get_label(idx) for idx in range(len(dataset))] 
        dataset = select_features_index(dataset, index=args.features)
        num_classes = len(set(labels))
        
        if args.split_type == 'random':
            train_indices, test_indices = train_test_split(
                range(len(dataset)),
                test_size=0.2,
                random_state=seed,
                stratify=labels
            )
        elif args.split_type == 'temporal':
            train_indices_file = f"../../_data/GoG/node/{args.chain}_train_index_{num_classes}.txt"
            test_indices_file = f"../../_data/GoG/node/{args.chain}_test_index_{num_classes}.txt"
            with open(train_indices_file, 'r') as f:
                train_indices = [int(line.strip()) for line in f]
            with open(test_indices_file, 'r') as f:
                test_indices = [int(line.strip()) for line in f]

        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        print_class_ratios_loader(train_loader)

        num_features = len(args.features)  
        if args.model == 'GCN':
            model = GCN(num_features=num_features, hidden_dim = args.hidden_channels, num_classes=num_classes).to(device)
        elif args.model == 'GIN':
            model = GIN(num_features=num_features, hidden_dim = args.hidden_channels, num_classes=num_classes).to(device)
        elif args.model == 'GraphSAGE':
            model = GraphSAGE(num_features=num_features, hidden_dim = args.hidden_channels, num_classes=num_classes).to(device)
        elif args.model == 'GAT':
            model = GAT(num_features=num_features, hidden_dim = args.hidden_channels, num_classes=num_classes).to(device)
        elif args.model == "ResidualGCN":
            model = ResidualGCN(num_features=num_features, hidden_dim = args.hidden_channels, num_classes=num_classes).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(args.epochs):
            train_metrics = train(model, train_loader, optimizer, criterion, device, num_classes)
            
            if (epoch + 1) % 5 != 0:
                test_metrics = evaluate(model, test_loader, criterion, device, num_classes)
            else:
                test_metrics = evaluate(model, test_loader, criterion, device, num_classes, True)

                print(f'Epoch {epoch+1}/{args.epochs}, Test Loss: {test_metrics[0]}')

            metrics['train'].append(train_metrics)
            metrics['test'].append(test_metrics)

        # Summarize results
        train_mean_metrics = np.mean(metrics['train'], axis=0)
        train_std_metrics = np.std(metrics['train'], axis=0)

        test_mean_metrics = np.mean(metrics['test'], axis=0)
        test_std_metrics = np.std(metrics['test'], axis=0)

        print("Summary Metrics Across Seeds:")
        print('Train')
        print(f"Mean Train Metrics: Accuracy: {train_mean_metrics[1]:.4f}, F1-macro: {train_mean_metrics[4]:.4f}, F1-micro: {train_mean_metrics[5]:.4f}")
        print(f"Std Deviation Train Metrics: Accuracy: {train_std_metrics[1]:.4f}, F1-macro: {train_std_metrics[4]:.4f}, F1-micro: {train_std_metrics[5]:.4f}")

        print('Test')
        print(f"Mean Test Metrics: Accuracy: {test_mean_metrics[1]:.4f}, F1-macro: {test_mean_metrics[4]:.4f}, F1-micro: {test_mean_metrics[5]:.4f}")
        print(f"Std Deviation Test Metrics: Accuracy: {test_std_metrics[1]:.4f}, F1-macro: {test_std_metrics[4]:.4f}, F1-micro: {test_std_metrics[5]:.4f}")

if __name__ == "__main__":
    main()
