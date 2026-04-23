import os
import argparse
import torch
import numpy as np
from train import train, evaluate
from dataloader import TransactionDataset
from torch.utils.data import Dataset, Subset, SubsetRandomSampler, random_split
from torch_geometric.loader import DataLoader
from sklearn.metrics import classification_report
import pandas as pd
from collections import Counter
import heapq
from torch_geometric.data import Data
import random

def remap_labels_to_binary(dataset):
    label_map = {9: 1}  # Fraud class
    new_label = 0
    for data in dataset:
        old_label = data.y.item()
        if old_label != 9 and old_label not in label_map:
            label_map[old_label] = new_label
            new_label = 0 

    print(f"Label map: {label_map}") 

    remapped_data_list = []
    for i in range(len(dataset)):
        data = dataset[i]
        old_label = data.y.item()
        if old_label in label_map:
            data.y = torch.tensor(label_map[old_label])
        remapped_data_list.append(data)

    for data in remapped_data_list:
        assert data.y.item() in [0, 1], f"Label {data.y.item()} out of range for binary classification."

    return remapped_data_list, 2  # Binary classification always has 2 classes

def remap_labels_to_multiclass(dataset, exclude_classes):
    label_map = {}
    new_label = 0
    for data in dataset:
        old_label = data.y.item()
        if old_label not in exclude_classes and old_label not in label_map:
            label_map[old_label] = new_label
            new_label += 1
    print(f"Label map: {label_map}") 
    remapped_data_list = []
    for i in range(len(dataset)):
        data = dataset[i]
        old_label = data.y.item()
        if old_label in label_map:
            data.y = torch.tensor(label_map[old_label])
            remapped_data_list.append(data)
    
    for data in remapped_data_list:
        assert 0 <= data.y.item() < len(label_map), f"Label {data.y.item()} out of range after remapping."

    return remapped_data_list, len(label_map)

def print_class_ratios(dataset):
    label_counts = {}
    for data in dataset:
        label = data.y.item()
        if label in label_counts:
            label_counts[label] += 1
        else:
            label_counts[label] = 1

    total_samples = sum(label_counts.values())
    class_ratios = {label: count / total_samples for label, count in label_counts.items()}

    print("Class Ratios in Dataset:")
    for label, ratio in class_ratios.items():
        print(f"Class {label}: {ratio:.4f} ({label_counts[label]} samples)")

def compute_class_weights(dataset, num_classes):
    label_counts = np.zeros(num_classes)
    for data in dataset:
        label_counts[data.y.item()] += 1
    class_weights = 1.0 / label_counts
    class_weights = class_weights / class_weights.sum() * num_classes
    return torch.tensor(class_weights, dtype=torch.float)

def resample_by_labels(dataset, ratio = 1):
    class_counts = Counter(data.y.item() for data in dataset)
    smallest_class_count = min(class_counts.values())

    print(f"Class counts: {class_counts}")
    print(f"Smallest class count: {smallest_class_count}")
    
    if ratio == 'balanced':
        average_class_count = int(np.mean(list(class_counts.values())))
        class_indices = {class_id: [] for class_id in class_counts}
        for i, data in enumerate(dataset):
            class_indices[data.y.item()].append(i)
        sampled_indices = []
        for class_id, indices in class_indices.items():
            np.random.shuffle(indices)
            if len(indices) > average_class_count:
                sampled_indices.extend(np.random.choice(indices, average_class_count, replace=False))
            else:
                sampled_indices.extend(np.random.choice(indices, average_class_count, replace=True))
    else:
        max_class_size = ratio * smallest_class_count

        class_indices = {class_id: [] for class_id in class_counts}
        for i, data in enumerate(dataset):
            class_indices[data.y.item()].append(i)
        
        sampled_indices = []
        for class_id, indices in class_indices.items():
            np.random.shuffle(indices)
            sampled_indices.extend(indices[:min(len(indices), max_class_size)])
    
    np.random.shuffle(sampled_indices)
    
    sampler = SubsetRandomSampler(sampled_indices)
    return DataLoader(dataset, batch_size=16, sampler=sampler)

def print_class_ratios_loader(dataloader):
    label_counts = {}
    total_samples = 0

    i = 0
    for data in dataloader:
        labels = data.y  
        
        for label in labels:
            label = label.item()  
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
        i += 1

    total_samples = sum(label_counts.values())
    print(f"Total samples processed: {total_samples}")
    class_ratios = {label: count / total_samples for label, count in label_counts.items()}

    print("Class Ratios in DataLoader:")
    for label, ratio in class_ratios.items():
        print(f"Class {label}: {ratio:.4f} ({label_counts[label]} samples)")


def select_features_index(dataset, index=[0, 1, 2], scale_indices=[3, 4], scale_factor=1e18):
    remapped_data_list = []
    for data in dataset:
        if max(scale_indices) < data.x.size(1):
            new_x = data.x.clone() 
            new_x[:, scale_indices] = new_x[:, scale_indices] / scale_factor
        
        new_x_tensor = new_x[:, index]
        
        new_data = Data(x=new_x_tensor, edge_index=data.edge_index, y=data.y)
        if hasattr(data, 'edge_attr'):
            new_data.edge_attr = data.edge_attr
        
        remapped_data_list.append(new_data)

    return remapped_data_list


def calculate_class_weights(dataset):
    class_counts = Counter(data.y.item() for data in dataset)
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    class_weights = {class_id: total_samples / count for class_id, count in class_counts.items()}
    
    total_weight = sum(class_weights.values())
    for class_id in class_weights:
        class_weights[class_id] = (class_weights[class_id] / total_weight) * num_classes

    weights_tensor = torch.tensor(list(class_weights.values()), dtype=torch.float)
    return weights_tensor


def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)