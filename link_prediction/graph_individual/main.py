# import argparse
# import torch
# import numpy as np
# from torch_geometric.loader import DataLoader
# from torch.nn import CrossEntropyLoss
# from torch.optim import Adam
# from model import GCN, GIN, GraphSage, GAT, ResidualGCN
# from train import train_model, evaluate_model 
# from dataset import TransactionEdgeDataset
# import warnings
# warnings.filterwarnings('ignore')

# def train_test(model_class, args):
#     all_train_metrics = []
#     all_test_metrics = []

#     seeds = [42, 43, 44]
#     for seed in seeds:
#         torch.manual_seed(seed)
#         np.random.seed(seed)
#         root_path = f'../../_data/GoG/edges/{args.chain}' # read in data folder

#         train_data = TransactionEdgeDataset(root=root_path, chain=args.chain, use_train=True)
#         test_data = TransactionEdgeDataset(root=root_path, chain=args.chain, use_train=False)

#         train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
#         test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

#         model = model_class(train_data.num_node_features, 2).to(args.device)
#         optimizer = Adam(model.parameters(), lr=0.01)
#         criterion = CrossEntropyLoss()

#         train_metrics_per_epoch = []
#         test_metrics_per_epoch = []

#         for epoch in range(args.epochs):
#             train_loss = train_model(model, train_loader, args.device, criterion, optimizer)
#             train_metrics_per_epoch.append(train_loss)

#             if (epoch + 1) % 5 == 0:
#                 test_metrics = evaluate_model(model, test_loader, args.device)
#                 test_metrics_per_epoch.append(test_metrics)
#                 print(f'Epoch {epoch+1}/{args.epochs}, Test Accuracy: {test_metrics[0]:.4f}, AUC: {test_metrics[1]:.4f}, F1-macro: {test_metrics[2]:.4f}')

#         all_train_metrics.append(np.mean(train_metrics_per_epoch))
#         all_test_metrics.append(np.mean(test_metrics_per_epoch, axis=0))

#     train_mean = np.mean(all_train_metrics)
#     train_std = np.std(all_train_metrics)

#     test_mean = np.mean(all_test_metrics, axis=0)
#     test_std = np.std(all_test_metrics, axis=0)

#     print("Summary Metrics Across Seeds:")
#     print(f"Train Loss: Mean: {train_mean:.4f}, Std Dev: {train_std:.4f}")
#     print(f"Test Accuracy: Mean: {test_mean[0]:.4f}, Std Dev: {test_std[0]:.4f}")
#     print(f"Test AUC: Mean: {test_mean[1]:.4f}, Std Dev: {test_std[1]:.4f}")
#     print(f"Test F1: Mean: {test_mean[2]:.4f}, Std Dev: {test_std[2]:.4f}")

#     return {
#         'train_loss_mean': train_mean, 
#         'train_loss_std': train_std, 
#         'test_accuracy_mean': test_mean[0], 
#         'test_accuracy_std': test_std[0],
#         'test_auc_mean': test_mean[1], 
#         'test_auc_std': test_std[1],
#         'test_f1_mean': test_mean[2], 
#         'test_f1_std': test_std[2]
#     }

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--chain', type=str, default='polygon')
#     parser.add_argument('--batch_size', type=int, default=32)
#     parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
#     parser.add_argument('--epochs', type=int, default=100)
#     args = parser.parse_args()
   
#     model_mapping = {
#     'GCN': GCN,
#     'GIN': GIN,
#     'GraphSage': GraphSage,
#     'GAT': GAT,
#     'ResidualGCN': ResidualGCN}

#     results = {}
#     for model_name, model_class in model_mapping.items():
#         print('>> start ', model_name)
#         results[model_name] = train_test(model_class, args)
#         print('>> end ', model_name)
#     print(results)

# if __name__ == "__main__":
#     main()


import argparse
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from model import GCN2, GIN2, GraphSage2, GAT2, ResidualGCN
from train import train_model, evaluate_model 
from dataset import TransactionEdgeDataset
import warnings
warnings.filterwarnings('ignore')

def train_test(model_class, args):
    all_train_metrics = []
    all_test_metrics = []

    seeds = [42, 43, 44]
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        root_path = f'../../_data/GoG/edges/{args.chain}' 

        train_data = TransactionEdgeDataset(root=root_path, chain=args.chain, use_train=True)
        test_data = TransactionEdgeDataset(root=root_path, chain=args.chain, use_train=False)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        model = model_class(train_data.num_node_features, 2).to(args.device)
        optimizer = Adam(model.parameters(), lr=0.01)
        criterion = CrossEntropyLoss()

        train_metrics_per_epoch = []
        test_metrics_per_epoch = []

        for epoch in range(args.epochs):
            train_loss = train_model(model, train_loader, args.device, criterion, optimizer)
            train_metrics_per_epoch.append(train_loss)

            if (epoch + 1) % 5 == 0:
                test_metrics = evaluate_model(model, test_loader, args.device)
                test_metrics_per_epoch.append(test_metrics)
                print(f'Epoch {epoch+1}/{args.epochs}, Test Accuracy: {test_metrics[0]:.4f}, AUC: {test_metrics[1]:.4f}, F1-macro: {test_metrics[2]:.4f}')

        all_train_metrics.append(np.mean(train_metrics_per_epoch))
        all_test_metrics.append(np.mean(test_metrics_per_epoch, axis=0))

    train_mean = np.mean(all_train_metrics)
    train_std = np.std(all_train_metrics)

    test_mean = np.mean(all_test_metrics, axis=0)
    test_std = np.std(all_test_metrics, axis=0)

    print("Summary Metrics Across Seeds:")
    print(f"Train Loss: Mean: {train_mean:.4f}, Std Dev: {train_std:.4f}")
    print(f"Test Accuracy: Mean: {test_mean[0]:.4f}, Std Dev: {test_std[0]:.4f}")
    print(f"Test AUC: Mean: {test_mean[1]:.4f}, Std Dev: {test_std[1]:.4f}")
    print(f"Test F1: Mean: {test_mean[2]:.4f}, Std Dev: {test_std[2]:.4f}")

    return {
        'train_loss_mean': train_mean, 
        'train_loss_std': train_std, 
        'test_accuracy_mean': test_mean[0], 
        'test_accuracy_std': test_std[0],
        'test_auc_mean': test_mean[1], 
        'test_auc_std': test_std[1],
        'test_f1_mean': test_mean[2], 
        'test_f1_std': test_std[2]
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=str, default='polygon')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
   
    model_mapping = {
    'GCN2': GCN2,
    'GIN2': GIN2,
    'GraphSage2': GraphSage2,
    'GAT2': GAT2,
    'ResidualGCN': ResidualGCN}

    results = {}
    for model_name, model_class in model_mapping.items():
        print('>> start ', model_name)
        results[model_name] = train_test(model_class, args)
        print('>> end ', model_name)
    print(results)

if __name__ == "__main__":
    main()
