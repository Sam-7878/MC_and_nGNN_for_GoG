# src/gog_fraud/data/preprocessing/ngnn/serialization.py

import os
import json
import torch
from typing import List, Dict, Any
from torch_geometric.data import Data, Batch

def save_rooted_subgraphs(
    subgraphs: List[Data], 
    save_path: str, 
    metadata: Dict[str, Any]
):
    """
    Saves a list of rooted subgraphs representing a single Level 1 graph
    into a .pt file along with necessary extraction metadata.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # We combine them into a single Batch object for easier I/O and loading
    batched_data = Batch.from_data_list(subgraphs)
    
    # Attach serialization metadata
    batched_data.extraction_metadata = metadata
    
    torch.save(batched_data, save_path)
    
def load_rooted_subgraphs(load_path: str) -> Batch:
    """
    Loads precomputed rooted subgraphs from a .pt file.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Artifact {load_path} not found.")
    return torch.load(load_path, weights_only=False)

def write_manifest(manifest_path: str, manifest_data: List[Dict[str, Any]]):
    """
    Writes a manifest of precomputed artifacts to JSON.
    """
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=4)

def load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """
    Loads JSON manifest containing the precomputed sample registry.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest {manifest_path} not found.")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)
