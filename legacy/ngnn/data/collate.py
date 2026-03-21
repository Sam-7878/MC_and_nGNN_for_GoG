# data/collate.py

from __future__ import annotations

from typing import Dict, List

import torch
from torch_geometric.data import Batch


def hierarchical_collate_fn(samples: List[Dict]) -> Dict:
    """
    Collate function for GoG + nGNN MVP.

    Input sample format:
        {
          "local_graph": Data,
          "contract_id": int,
          "contract_name": str,
          "label": int,
          "contract_feature": FloatTensor[Fg],
          "global_edge_index": LongTensor[2, E],
          "global_features": FloatTensor[N, Fg],
        }

    Output batch format:
        {
          "local_batch": PyG Batch,
          "contract_ids": LongTensor [B],
          "contract_names": List[str],
          "labels": LongTensor [B],
          "global_edge_index": LongTensor [2, E],
          "global_features": FloatTensor [N, Fg],
        }
    """
    if len(samples) == 0:
        raise ValueError("Received empty samples list in hierarchical_collate_fn.")

    local_graphs = [s["local_graph"] for s in samples]
    contract_ids = torch.tensor([s["contract_id"] for s in samples], dtype=torch.long)
    labels = torch.tensor([s["label"] for s in samples], dtype=torch.long)
    contract_names = [s["contract_name"] for s in samples]

    local_batch = Batch.from_data_list(local_graphs)

    # Full-graph transductive setting:
    # Every sample in the batch shares the same full global graph.
    global_edge_index = samples[0]["global_edge_index"]
    global_features = samples[0]["global_features"]

    batch = {
        "local_batch": local_batch,
        "contract_ids": contract_ids,
        "contract_names": contract_names,
        "labels": labels,
        "global_edge_index": global_edge_index,
        "global_features": global_features,
    }

    return batch
