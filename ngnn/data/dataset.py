# data/dataset.py

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import dropout_edge


class HierarchicalDataset(Dataset):
    """
    MVP Hierarchical Dataset for GoG + nGNN.

    Raw local graph JSON schema:
        {
          "edges": [[src, dst], ...],
          "features": [[...], ...],              # local node features
          "contract_feature": [...],             # static contract/global feature
          "label": 0 or 1
        }

    Returned item:
        {
          "local_graph": Data,
          "contract_id": int,                    # global contract node id
          "contract_name": str,                  # json stem
          "label": int,
          "contract_feature": FloatTensor[Fg],
          "global_edge_index": LongTensor[2, E],
          "global_features": FloatTensor[N, Fg],
        }
    """

    def __init__(
        self,
        data_dir: str,
        contract_graph_path: Optional[str] = None,
        split: str = "train",
        split_seed: int = 42,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        edge_dropout: float = 0.0,
        force_reload_global_graph: bool = False,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.contract_graph_path = Path(contract_graph_path) if contract_graph_path else None
        self.split = split.lower()
        self.split_seed = split_seed
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.edge_dropout = edge_dropout
        self.force_reload_global_graph = force_reload_global_graph

        if self.split not in {"train", "val", "test", "all"}:
            raise ValueError(f"split must be one of ['train', 'val', 'test', 'all'], got {self.split}")

        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir does not exist: {self.data_dir}")

        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if not np.isclose(ratio_sum, 1.0):
            raise ValueError(
                f"train_ratio + val_ratio + test_ratio must be 1.0, got {ratio_sum}"
            )

        self.all_files = sorted(self.data_dir.glob("*.json"))
        if len(self.all_files) == 0:
            raise FileNotFoundError(f"No JSON files found in: {self.data_dir}")

        # ------------------------------------------------------------------
        # Step 1. Scan metadata from all JSON files
        # ------------------------------------------------------------------
        self.meta = self._scan_all_metadata(self.all_files)

        self.contract_names = [m["contract_name"] for m in self.meta]
        self.labels_all = np.array([m["label"] for m in self.meta], dtype=np.int64)

        # Dimensions inferred from actual raw JSON schema
        self.node_dim = int(self.meta[0]["node_dim"])
        self.edge_dim = 1  # raw JSON has no edge_attr -> synthesize constant 1-dim edge feature
        self.global_feat_dim = int(self.meta[0]["contract_feat_dim"])
        self.num_classes = int(len(sorted(set(self.labels_all.tolist()))))

        # ------------------------------------------------------------------
        # Step 2. Global contract indexing
        # ------------------------------------------------------------------
        self.contract_to_idx = {name: idx for idx, name in enumerate(self.contract_names)}
        self.idx_to_contract = {idx: name for name, idx in self.contract_to_idx.items()}

        # ------------------------------------------------------------------
        # Step 3. Build global feature matrix from raw JSON contract_feature
        #         shape: [N_contracts, global_feat_dim]
        # ------------------------------------------------------------------
        self.global_features = self._build_global_feature_matrix()

        # ------------------------------------------------------------------
        # Step 4. Load global contract graph edge_index
        #         If file is missing or unreadable, fallback to empty graph
        # ------------------------------------------------------------------
        self.global_edge_index = self._load_global_edge_index()

        # ------------------------------------------------------------------
        # Step 5. Split indices
        # ------------------------------------------------------------------
        self.indices = self._build_split_indices()

        # Convenience
        self.split_files = [self.all_files[i] for i in self.indices]
        self.split_labels = self.labels_all[self.indices]

    # ----------------------------------------------------------------------
    # Public helpers
    # ----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict:
        real_idx = self.indices[idx]
        meta = self.meta[real_idx]
        json_path = self.all_files[real_idx]

        local_graph = self._load_local_graph(json_path)

        # Optional data MC perturbation hook for future Phase 3/4
        if self.split == "train" and self.edge_dropout > 0.0:
            local_graph = self._apply_edge_dropout(local_graph, self.edge_dropout)

        contract_name = meta["contract_name"]
        contract_id = self.contract_to_idx[contract_name]
        label = int(meta["label"])
        contract_feature = self.global_features[contract_id]

        return {
            "local_graph": local_graph,
            "contract_id": contract_id,
            "contract_name": contract_name,
            "label": label,
            "contract_feature": contract_feature,
            "global_edge_index": self.global_edge_index,
            "global_features": self.global_features,
        }

    def get_metadata(self) -> Dict:
        return {
            "data_dir": str(self.data_dir),
            "num_total_contracts": len(self.all_files),
            "num_split_contracts": len(self.indices),
            "split": self.split,
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "global_feat_dim": self.global_feat_dim,
            "num_classes": self.num_classes,
            "num_global_edges": int(self.global_edge_index.size(1)),
            "edge_dropout": self.edge_dropout,
        }

    # ----------------------------------------------------------------------
    # Metadata scan
    # ----------------------------------------------------------------------
    def _scan_all_metadata(self, files: List[Path]) -> List[Dict]:
        meta = []

        for file_path in files:
            with open(file_path, "r") as f:
                data = json.load(f)

            features = data.get("features", [])
            edges = data.get("edges", [])
            contract_feature = data.get("contract_feature", [])
            label = int(data.get("label", 0))

            node_dim = self._infer_node_dim(features)
            contract_feat_dim = self._infer_contract_feat_dim(contract_feature)

            meta.append(
                {
                    "file_path": str(file_path),
                    "contract_name": file_path.stem,
                    "label": label,
                    "num_nodes": len(features),
                    "num_edges": len(edges),
                    "node_dim": node_dim,
                    "contract_feat_dim": contract_feat_dim,
                }
            )

        # Basic consistency checks
        node_dims = sorted(set(m["node_dim"] for m in meta))
        contract_feat_dims = sorted(set(m["contract_feat_dim"] for m in meta))

        if len(node_dims) != 1:
            raise ValueError(f"Inconsistent node feature dims found: {node_dims}")
        if len(contract_feat_dims) != 1:
            raise ValueError(f"Inconsistent contract feature dims found: {contract_feat_dims}")

        return meta

    @staticmethod
    def _infer_node_dim(features: List) -> int:
        if not features:
            raise ValueError("Empty 'features' found. Cannot infer local node feature dim.")
        if not isinstance(features[0], (list, tuple)):
            raise ValueError("Expected 'features' to be a list of lists.")
        return len(features[0])

    @staticmethod
    def _infer_contract_feat_dim(contract_feature: List) -> int:
        if contract_feature is None:
            return 0
        if not isinstance(contract_feature, (list, tuple)):
            raise ValueError("Expected 'contract_feature' to be a list.")
        return len(contract_feature)

    # ----------------------------------------------------------------------
    # Global features
    # ----------------------------------------------------------------------
    def _build_global_feature_matrix(self) -> torch.Tensor:
        """
        Build global static feature matrix from raw JSON 'contract_feature'.

        Output:
            global_features: FloatTensor [N_contracts, global_feat_dim]
        """
        num_contracts = len(self.all_files)
        feat_dim = self.global_feat_dim

        global_features = torch.zeros((num_contracts, feat_dim), dtype=torch.float)

        for i, file_path in enumerate(self.all_files):
            with open(file_path, "r") as f:
                data = json.load(f)

            contract_feature = data.get("contract_feature", [])
            if len(contract_feature) != feat_dim:
                raise ValueError(
                    f"Contract feature dim mismatch in {file_path.name}: "
                    f"expected {feat_dim}, got {len(contract_feature)}"
                )

            global_features[i] = torch.tensor(contract_feature, dtype=torch.float)

        return global_features

    # ----------------------------------------------------------------------
    # Global graph loader
    # ----------------------------------------------------------------------
    def _load_global_edge_index(self) -> torch.Tensor:
        """
        Load full contract-level edge_index from .pt file.

        Supported cases:
        - dict with keys like 'edge_index', 'edges', 'global_edge_index'
        - tensor itself
        - list/tuple containing edge_index

        If unavailable, fallback to empty edge_index of shape [2, 0].
        """
        if self.contract_graph_path is None or not self.contract_graph_path.exists():
            warnings.warn(
                f"[HierarchicalDataset] contract_graph_path not found: {self.contract_graph_path}. "
                f"Using empty global graph."
            )
            return torch.empty((2, 0), dtype=torch.long)

        obj = torch.load(self.contract_graph_path, map_location="cpu")
        edge_index = self._extract_edge_index(obj)

        if edge_index is None:
            warnings.warn(
                f"[HierarchicalDataset] Could not parse edge_index from: {self.contract_graph_path}. "
                f"Using empty global graph."
            )
            return torch.empty((2, 0), dtype=torch.long)

        edge_index = edge_index.long().contiguous()

        if edge_index.numel() > 0:
            max_idx = int(edge_index.max().item())
            if max_idx >= len(self.all_files):
                warnings.warn(
                    f"[HierarchicalDataset] global edge_index max node id={max_idx}, "
                    f"but num_contracts={len(self.all_files)}. "
                    f"Please verify that contract graph indexing matches sorted JSON filenames."
                )

        return edge_index

    def _extract_edge_index(self, obj) -> Optional[torch.Tensor]:
        # Case 1: direct tensor
        if torch.is_tensor(obj):
            return self._normalize_edge_index(obj)

        # Case 2: dict
        if isinstance(obj, dict):
            for key in ["edge_index", "global_edge_index", "contract_edge_index", "edges"]:
                if key in obj:
                    return self._extract_edge_index(obj[key])

        # Case 3: list/tuple
        if isinstance(obj, (list, tuple)):
            # direct edge list: [[u, v], ...]
            if len(obj) > 0 and isinstance(obj[0], (list, tuple)) and len(obj[0]) == 2:
                return self._normalize_edge_index(torch.tensor(obj, dtype=torch.long))

            for item in obj:
                edge_index = self._extract_edge_index(item)
                if edge_index is not None:
                    return edge_index

        return None

    @staticmethod
    def _normalize_edge_index(edge_index: torch.Tensor) -> Optional[torch.Tensor]:
        if edge_index.dim() != 2:
            return None

        # Already [2, E]
        if edge_index.size(0) == 2:
            return edge_index.long()

        # Possibly [E, 2]
        if edge_index.size(1) == 2:
            return edge_index.t().contiguous().long()

        return None

    # ----------------------------------------------------------------------
    # Split logic
    # ----------------------------------------------------------------------
    def _build_split_indices(self) -> np.ndarray:
        all_indices = np.arange(len(self.all_files))
        all_labels = self.labels_all

        if self.split == "all":
            return all_indices

        # First split: train vs temp(val+test)
        train_size = self.train_ratio
        temp_size = self.val_ratio + self.test_ratio

        try:
            train_idx, temp_idx, train_y, temp_y = train_test_split(
                all_indices,
                all_labels,
                test_size=temp_size,
                stratify=all_labels if self._can_stratify(all_labels) else None,
                random_state=self.split_seed,
            )
        except Exception:
            train_idx, temp_idx, train_y, temp_y = train_test_split(
                all_indices,
                all_labels,
                test_size=temp_size,
                stratify=None,
                random_state=self.split_seed,
            )

        if self.split == "train":
            return np.sort(train_idx)

        # Second split: val vs test on temp
        if len(temp_idx) == 0:
            return np.array([], dtype=np.int64)

        val_portion_in_temp = self.val_ratio / (self.val_ratio + self.test_ratio)

        try:
            val_idx, test_idx, _, _ = train_test_split(
                temp_idx,
                temp_y,
                test_size=(1.0 - val_portion_in_temp),
                stratify=temp_y if self._can_stratify(temp_y) else None,
                random_state=self.split_seed,
            )
        except Exception:
            val_idx, test_idx, _, _ = train_test_split(
                temp_idx,
                temp_y,
                test_size=(1.0 - val_portion_in_temp),
                stratify=None,
                random_state=self.split_seed,
            )

        if self.split == "val":
            return np.sort(val_idx)
        if self.split == "test":
            return np.sort(test_idx)

        raise ValueError(f"Unexpected split: {self.split}")

    @staticmethod
    def _can_stratify(labels: np.ndarray) -> bool:
        unique, counts = np.unique(labels, return_counts=True)
        return len(unique) >= 2 and np.all(counts >= 2)

    # ----------------------------------------------------------------------
    # Local graph loader
    # ----------------------------------------------------------------------
    def _load_local_graph(self, json_path: Path) -> Data:
        with open(json_path, "r") as f:
            raw = json.load(f)

        features = raw.get("features", [])
        edges = raw.get("edges", [])

        if len(features) == 0:
            raise ValueError(f"Empty features in {json_path}")

        x = torch.tensor(features, dtype=torch.float)  # [N, 4] in current raw GoG schema
        num_nodes = x.size(0)

        if len(edges) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()  # [2, E]

            # MVP decision:
            # raw JSON has no explicit edge feature, so we synthesize 1-dim constant edge_attr.
            edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
        )

        return data

    # ----------------------------------------------------------------------
    # Optional data perturbation hook for later phases
    # ----------------------------------------------------------------------
    @staticmethod
    def _apply_edge_dropout(data: Data, p: float) -> Data:
        if data.edge_index.numel() == 0:
            return data

        edge_index_dropped, edge_mask = dropout_edge(
            data.edge_index,
            p=p,
            force_undirected=False,
            training=True,
        )

        data.edge_index = edge_index_dropped

        if getattr(data, "edge_attr", None) is not None and data.edge_attr.size(0) == edge_mask.size(0):
            data.edge_attr = data.edge_attr[edge_mask]

        return data
