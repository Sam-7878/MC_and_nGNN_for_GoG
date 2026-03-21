# data/dataset.py

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data


def signed_log1p_tensor(x: torch.Tensor) -> torch.Tensor:
    """
    Signed log transform for heavy-tailed numeric features.
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def to_safe_float_tensor(
    array_like,
    clip_value: float = 1e12,
    apply_signed_log: bool = True,
) -> torch.Tensor:
    """
    Convert array-like input to a finite float tensor.

    Steps:
    1. cast to float32
    2. replace NaN / Inf
    3. clip extreme values
    4. optional signed-log transform
    """
    arr = np.asarray(array_like, dtype=np.float32)

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=clip_value,
        neginf=-clip_value,
    )
    arr = np.clip(arr, -clip_value, clip_value)

    t = torch.from_numpy(arr)

    if apply_signed_log:
        t = signed_log1p_tensor(t)

    return t


class HierarchicalDataset(Dataset):
    """
    Dataset for GoG + nGNN hierarchical learning.

    Each item represents ONE contract and returns:
        {
            "local_graph"   : PyG Data,
            "contract_id"   : int,
            "contract_name" : str,
            "label"         : int,
        }

    The dataset also stores FULL global graph components:
        self.global_edge_index : LongTensor [2, E]
        self.global_features   : FloatTensor [N, Fg]

    Expected per-contract JSON schema (robustly handled):
        - node features:
            "features" | "node_features" | "x"
        - local edges:
            "edges" | "edge_index"
        - optional local edge features:
            "edge_attr" | "edge_features"
        - contract/global feature:
            "contract_feature" | "contract_features" | "global_features"
        - label:
            "label" | "y" | "is_malicious" | "target"
        - name/id:
            "contract_name" | "address" | filename stem

    Expected global graph .pt schema (robustly handled):
        dict or PyG Data with:
            - edge_index
            - one of:
                "contract_to_idx"
                "addr_to_idx"
                "name_to_idx"
                "contract_names"
                "node_names"
    """

    def __init__(
        self,
        data_dir: str,
        contract_graph_path: str,
        split: str = "train",
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        clip_value: float = 1e12,
        apply_signed_log: bool = True,
    ):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.contract_graph_path = Path(contract_graph_path)
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed
        self.clip_value = clip_value
        self.apply_signed_log = apply_signed_log

        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")
        if not self.contract_graph_path.exists():
            raise FileNotFoundError(
                f"contract_graph_path not found: {self.contract_graph_path}"
            )

        if split not in {"train", "val", "test", "all"}:
            raise ValueError(f"split must be one of train/val/test/all, got {split}")

        # ------------------------------------------------------------
        # 1) Load all per-contract JSONs
        # ------------------------------------------------------------
        self.all_samples = self.load_contract_graphs()

        if len(self.all_samples) == 0:
            raise RuntimeError(f"No JSON files found in: {self.data_dir}")

        # ------------------------------------------------------------
        # 2) Load full global contract graph
        # ------------------------------------------------------------
        (
            self.global_edge_index,
            self.contract_to_idx,
            self.idx_to_contract,
        ) = self.load_global_contract_graph()

        # ------------------------------------------------------------
        # 3) Build full global feature matrix [N, Fg]
        # ------------------------------------------------------------
        self.global_features = self.build_global_feature_matrix()

        # ------------------------------------------------------------
        # 4) Assign contract_id to each sample
        # ------------------------------------------------------------
        for sample in self.all_samples:
            contract_name = sample["contract_name"]
            if contract_name not in self.contract_to_idx:
                raise KeyError(
                    f"Contract '{contract_name}' from JSON not found in global graph mapping."
                )
            sample["contract_id"] = self.contract_to_idx[contract_name]

        # ------------------------------------------------------------
        # 5) Apply split
        # ------------------------------------------------------------
        self.samples = self.apply_split(self.all_samples)

        # ------------------------------------------------------------
        # 6) Infer dimensions
        # ------------------------------------------------------------
        self.node_dim, self.edge_dim, self.global_feat_dim = self.infer_feature_dims()

    # ------------------------------------------------------------------
    # Required Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        local_graph = sample["local_graph"]
        contract_id = sample["contract_id"]
        contract_name = sample["contract_name"]
        label = sample["label"]

        return {
            "local_graph": local_graph,
            "contract_id": contract_id,
            "contract_name": contract_name,
            "label": label,
        }

    # ------------------------------------------------------------------
    # Load per-contract JSON files
    # ------------------------------------------------------------------
    def load_contract_graphs(self) -> List[Dict]:
        """
        Load all contract-level local transaction graphs from JSON files.
        """
        json_files = sorted(self.data_dir.glob("*.json"))

        samples: List[Dict] = []

        for json_path in json_files:
            with open(json_path, "r", encoding="utf-8") as f:
                obj = json.load(f)

            contract_name = self._extract_contract_name(obj, json_path)
            label = self._extract_label(obj)
            node_features = self._extract_node_features(obj, json_path)
            edge_index = self._extract_edge_index(obj)
            edge_attr = self._extract_edge_attr(obj, edge_index)
            contract_feature = self._extract_contract_feature(obj)

            local_graph = Data(
                x=node_features,                 # [num_nodes, node_dim]
                edge_index=edge_index,          # [2, num_edges]
                edge_attr=edge_attr,            # [num_edges, edge_dim]
            )

            samples.append(
                {
                    "contract_name": contract_name,
                    "label": label,
                    "local_graph": local_graph,
                    "contract_feature": contract_feature,   # [Fg]
                }
            )

        return samples

    # ------------------------------------------------------------------
    # Load full global contract graph
    # ------------------------------------------------------------------
    def load_global_contract_graph(
        self,
    ) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, str]]:
        """
        Load full contract-level graph and mapping from .pt file.
        """
        obj = torch.load(self.contract_graph_path, map_location="cpu")

        # edge_index
        if isinstance(obj, dict):
            if "edge_index" in obj:
                edge_index = obj["edge_index"]
            else:
                raise KeyError(
                    f"'edge_index' not found in global graph file: {self.contract_graph_path}"
                )
        else:
            if hasattr(obj, "edge_index"):
                edge_index = obj.edge_index
            else:
                raise KeyError(
                    f"'edge_index' not found in global graph object: {self.contract_graph_path}"
                )

        edge_index = torch.as_tensor(edge_index, dtype=torch.long)

        if edge_index.numel() == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        elif edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                f"global edge_index must have shape [2, E], got {tuple(edge_index.shape)}"
            )

        contract_to_idx = self._extract_contract_mapping(obj)

        idx_to_contract = {idx: name for name, idx in contract_to_idx.items()}

        return edge_index, contract_to_idx, idx_to_contract

    # ------------------------------------------------------------------
    # Build full global feature matrix
    # ------------------------------------------------------------------
    def build_global_feature_matrix(self) -> torch.Tensor:
        """
        Build [N, Fg] matrix aligned with global contract node indices.
        """
        # infer N from mapping
        if len(self.contract_to_idx) == 0:
            raise RuntimeError("contract_to_idx is empty.")

        num_contracts = max(self.contract_to_idx.values()) + 1

        # infer feature dim from first sample
        first_feat = self.all_samples[0]["contract_feature"]
        if first_feat.dim() != 1:
            raise ValueError(
                f"contract_feature must be 1D, got shape {tuple(first_feat.shape)}"
            )
        global_feat_dim = first_feat.size(0)

        global_features = torch.zeros(
            (num_contracts, global_feat_dim),
            dtype=torch.float32,
        )

        for sample in self.all_samples:
            name = sample["contract_name"]
            feat = sample["contract_feature"]

            if feat.dim() != 1:
                raise ValueError(
                    f"contract_feature for '{name}' must be 1D, got {tuple(feat.shape)}"
                )

            idx = self.contract_to_idx.get(name, None)
            if idx is None:
                continue

            global_features[idx] = feat

        global_features = torch.nan_to_num(
            global_features,
            nan=0.0,
            posinf=1e6,
            neginf=-1e6,
        )

        return global_features

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------
    def apply_split(self, samples: List[Dict]) -> List[Dict]:
        if self.split == "all":
            return samples

        n = len(samples)
        indices = list(range(n))

        rng = random.Random(self.seed)
        rng.shuffle(indices)

        train_ratio, val_ratio, test_ratio = self.split_ratio
        if not math.isclose(train_ratio + val_ratio + test_ratio, 1.0, rel_tol=1e-6):
            raise ValueError(
                f"split_ratio must sum to 1.0, got {self.split_ratio}"
            )

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        if self.split == "train":
            selected = train_idx
        elif self.split == "val":
            selected = val_idx
        else:
            selected = test_idx

        return [samples[i] for i in selected]

    # ------------------------------------------------------------------
    # Feature dimension inference
    # ------------------------------------------------------------------
    def infer_feature_dims(self) -> Tuple[int, int, int]:
        sample = self.samples[0]

        local_graph = sample["local_graph"]
        node_dim = local_graph.x.size(-1)

        if local_graph.edge_attr is None:
            edge_dim = 1
        else:
            edge_dim = local_graph.edge_attr.size(-1)

        global_feat_dim = self.global_features.size(-1)

        return node_dim, edge_dim, global_feat_dim

    # ------------------------------------------------------------------
    # Extractors
    # ------------------------------------------------------------------
    def _extract_contract_name(self, obj: Dict, json_path: Path) -> str:
        for key in ["contract_name", "address", "contract", "name"]:
            if key in obj and obj[key] is not None:
                return str(obj[key])
        return json_path.stem

    def _extract_label(self, obj: Dict) -> int:
        for key in ["label", "y", "target", "is_malicious", "fraud_label"]:
            if key in obj:
                value = obj[key]
                if isinstance(value, bool):
                    return int(value)
                return int(value)
        return 0

    def _extract_node_features(self, obj: Dict, json_path: Path) -> torch.Tensor:
        raw = None
        for key in ["features", "node_features", "x"]:
            if key in obj:
                raw = obj[key]
                break

        if raw is None:
            raise KeyError(f"No node feature key found in {json_path.name}")

        x = to_safe_float_tensor(
            raw,
            clip_value=self.clip_value,
            apply_signed_log=self.apply_signed_log,
        )

        if x.dim() != 2:
            raise ValueError(
                f"Node features in {json_path.name} must be 2D [num_nodes, node_dim], got {tuple(x.shape)}"
            )

        x = torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        return x

    def _extract_edge_index(self, obj: Dict) -> torch.Tensor:
        raw = None
        for key in ["edges", "edge_index"]:
            if key in obj:
                raw = obj[key]
                break

        if raw is None or len(raw) == 0:
            return torch.empty((2, 0), dtype=torch.long)

        edge_index = torch.as_tensor(raw, dtype=torch.long)

        # allow both [E, 2] and [2, E]
        if edge_index.dim() != 2:
            raise ValueError(f"edge_index must be 2D, got shape {tuple(edge_index.shape)}")

        if edge_index.size(0) == 2:
            pass
        elif edge_index.size(1) == 2:
            edge_index = edge_index.t().contiguous()
        else:
            raise ValueError(
                f"edge_index must have shape [2, E] or [E, 2], got {tuple(edge_index.shape)}"
            )

        return edge_index

    def _extract_edge_attr(self, obj: Dict, edge_index: torch.Tensor) -> torch.Tensor:
        num_edges = edge_index.size(1)

        raw = None
        for key in ["edge_attr", "edge_features"]:
            if key in obj:
                raw = obj[key]
                break

        if raw is None:
            # synthesize constant edge feature
            return torch.ones((num_edges, 1), dtype=torch.float32)

        edge_attr = to_safe_float_tensor(
            raw,
            clip_value=self.clip_value,
            apply_signed_log=self.apply_signed_log,
        )

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        if edge_attr.dim() != 2:
            raise ValueError(
                f"edge_attr must be 2D [E, edge_dim], got {tuple(edge_attr.shape)}"
            )

        if edge_attr.size(0) != num_edges:
            raise ValueError(
                f"edge_attr row count mismatch: edge_attr has {edge_attr.size(0)} rows, "
                f"but edge_index has {num_edges} edges"
            )

        edge_attr = torch.nan_to_num(edge_attr, nan=0.0, posinf=1e6, neginf=-1e6)
        return edge_attr

    def _extract_contract_feature(self, obj: Dict) -> torch.Tensor:
        raw = None
        for key in ["contract_feature", "contract_features", "global_features"]:
            if key in obj:
                raw = obj[key]
                break

        if raw is None:
            raise KeyError(
                "No contract-level feature key found. "
                "Expected one of: contract_feature / contract_features / global_features"
            )

        feat = to_safe_float_tensor(
            raw,
            clip_value=self.clip_value,
            apply_signed_log=self.apply_signed_log,
        )

        if feat.dim() == 2 and feat.size(0) == 1:
            feat = feat.squeeze(0)

        if feat.dim() != 1:
            raise ValueError(
                f"contract_feature must be 1D [Fg], got {tuple(feat.shape)}"
            )

        feat = torch.nan_to_num(feat, nan=0.0, posinf=1e6, neginf=-1e6)
        return feat

    def _extract_contract_mapping(self, obj) -> Dict[str, int]:
        """
        Try multiple common mapping formats.
        """
        # dict-style
        if isinstance(obj, dict):
            for key in ["contract_to_idx", "addr_to_idx", "name_to_idx"]:
                if key in obj and isinstance(obj[key], dict):
                    return {str(k): int(v) for k, v in obj[key].items()}

            for key in ["contract_names", "node_names", "contracts"]:
                if key in obj and isinstance(obj[key], (list, tuple)):
                    return {str(name): i for i, name in enumerate(obj[key])}

        # object-style
        for key in ["contract_to_idx", "addr_to_idx", "name_to_idx"]:
            if hasattr(obj, key):
                mapping = getattr(obj, key)
                if isinstance(mapping, dict):
                    return {str(k): int(v) for k, v in mapping.items()}

        for key in ["contract_names", "node_names", "contracts"]:
            if hasattr(obj, key):
                names = getattr(obj, key)
                if isinstance(names, (list, tuple)):
                    return {str(name): i for i, name in enumerate(names)}

        raise KeyError(
            "Could not extract contract-name-to-index mapping from global graph file. "
            "Expected one of: contract_to_idx / addr_to_idx / name_to_idx / contract_names / node_names"
        )
