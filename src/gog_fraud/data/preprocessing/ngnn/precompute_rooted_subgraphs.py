# src/gog_fraud/data/preprocessing/ngnn/precompute_rooted_subgraphs.py

import os
import gc
import json
import torch
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any, Optional

from gog_fraud.data.preprocessing.ngnn.subgraph_extraction import extract_rooted_subgraphs
from gog_fraud.data.preprocessing.ngnn.serialization import save_rooted_subgraphs, write_manifest

def run_precompute(
    dataset, # the original Level 1 dataset
    output_dir: str,
    chain: str,
    split: str,
    num_hops: int = 1,
    max_nodes_per_subgraph: int = 50,
    root_policy: str = "all",
    config_hash: Optional[str] = None
):
    """
    Iterates over the dataset, extracts rooted subgraphs for each Level 1 graph,
    and caches them as .pt files along with a manifest for immediate DataLoader use.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    manifest_data = []
    manifest_path = os.path.join(output_dir, "metadata.json")
    
    # Optional metadata tracking extraction settings
    extraction_config = {
        "num_hops": num_hops,
        "max_nodes_per_subgraph": max_nodes_per_subgraph,
        "root_policy": root_policy,
        "split": split,
        "chain": chain,
        "version": "1.0"
    }
    
    total_samples = len(dataset)
    print(f"[{chain}/{split}] Precomputing {total_samples} samples with num_hops={num_hops}...")
    
    for i in tqdm(range(total_samples)):
        data = dataset[i]
        
        # Determine identifiers
        sample_id = getattr(data, 'sample_id', i)
        contract_id = getattr(data, 'contract_id', f"contract_{i}")
        label = getattr(data, 'y', -1)
        if hasattr(data, 'label'):
             label = data.label

        # 1. Extraction
        subgraphs = extract_rooted_subgraphs(
            data=data,
            num_hops=num_hops,
            max_nodes_per_subgraph=max_nodes_per_subgraph,
            root_policy=root_policy
        )

        # 2. Serialization Destination
        # Using sample-per-file for simplicity and debugging, as per architecture doc recommendation
        artifact_filename = f"{sample_id}_{chain}_{split}.pt"
        artifact_path = os.path.join(output_dir, artifact_filename)

        # 3. Save
        save_rooted_subgraphs(subgraphs, artifact_path, extraction_config)

        # 4. Record in manifest
        manifest_data.append({
            "sample_id": sample_id,
            "contract_id": contract_id,
            "label": label if isinstance(label, int) else (label.item() if hasattr(label, 'item') else label),
            "artifact_filename": artifact_filename,
            "num_subgraphs": len(subgraphs),
            "split": split,
            "config_hash": config_hash
        })
        
        # Protect against OOM
        if i % 1000 == 0:
            gc.collect()

    write_manifest(manifest_path, manifest_data)
    print(f"Finished precomputing {chain}/{split}. Manifest saved at {manifest_path}")

if __name__ == "__main__":
    # Test stub for debugging
    print("Precompute Script CLI stub")
