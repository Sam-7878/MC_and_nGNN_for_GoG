# ngnn/contract_graph_builder.py

import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import networkx as nx

class ContractGraphBuilder:
    """
    Build contract-level graph from individual transaction graphs
    """
    def __init__(self, data_dir, method='hybrid', k_neighbors=5):
        """
        Args:
            data_dir: Path to JSON files
            method: 'knn', 'label', 'hybrid'
            k_neighbors: Number of neighbors for k-NN
        """
        self.data_dir = Path(data_dir)
        self.method = method
        self.k = k_neighbors
        self.contract_to_idx = {}  # filename -> contract_id
        self.idx_to_contract = {}


    def build(self, save_path=None):
        """Main pipeline to build contract graph"""
        print(f"🔨 Building Contract Graph (method={self.method})")
        
        # Step 1: Load all graphs and create contract index
        graphs, labels = self._load_all_graphs()
        print(f"✅ Loaded {len(graphs)} contract graphs")
        
        # Step 2: Compute graph embeddings
        embeddings = self._compute_graph_embeddings(graphs)
        print(f"✅ Computed embeddings: {embeddings.shape}")
        
        # Step 3: Build edges based on method
        if self.method == 'knn':
            edge_index = self._build_knn_edges(embeddings)
        elif self.method == 'label':
            edge_index = self._build_label_edges(labels)
        elif self.method == 'hybrid':
            edge_index = self._build_hybrid_edges(embeddings, labels)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        print(f"✅ Built contract graph: {edge_index.shape[1]} edges")
        
        # Step 4: Analyze and save
        self._analyze_contract_graph(edge_index, labels)
        
        if save_path:
            self._save_contract_graph(edge_index, self.method, graphs, embeddings, save_path)
        
        return edge_index, self.contract_to_idx


    def _load_all_graphs(self):
        """Load all JSON files and assign contract IDs"""
        json_files = sorted(list(self.data_dir.glob("*.json")))
        print(f"Loading graphs: {len(json_files)} files found")

        graphs = []
        labels = []
        
        for idx, file in enumerate(tqdm(json_files, desc="Loading graphs")):
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                # ✅ 1. Features 안전 처리 (핵심 수정)
                raw_features = data.get('features')
                
                # None이거나 비어있으면 빈 배열 처리
                if raw_features is None or len(raw_features) == 0:
                     features = np.array([], dtype=np.float32)
                else:
                    try:
                        # 강제로 float32로 변환 시도
                        features = np.array(raw_features, dtype=np.float32)
                        # 혹시 NaN이나 Inf가 있으면 0으로 치환 (선택 사항)
                        if np.isnan(features).any() or np.isinf(features).any():
                             features = np.nan_to_num(features)
                    except (ValueError, TypeError):
                        # 변환 실패 (Ragged Array, String 등) -> Feature가 없는 것으로 간주
                        features = np.array([], dtype=np.float32)
                
                # ✅ 2. Label 처리
                label = data.get('label', 0)
                
                # ✅ 3. Assign contract ID
                contract_name = file.stem
                self.contract_to_idx[contract_name] = idx
                self.idx_to_contract[idx] = contract_name
                
                graphs.append({
                    'edges': np.array(data['edges']),
                    'features': features,  # 안전하게 초기화된 배열
                    'num_nodes': data.get('num_nodes', 0),
                    'num_edges': data.get('num_edges', 0),
                    'label': label         # ✅ 미리 저장해둠 (나중에 다시 읽을 필요 없음)
                })
                labels.append(data.get('label', 0))
                
            except Exception as e:
                print(f"⚠️  Failed to load {file}: {e}")
                continue
        
        print(f"✅ Loaded {len(graphs)} contract graphs")
        return graphs, np.array(labels)


    def _compute_graph_embeddings(self, graphs):
        """
        Compute graph-level embeddings using graph statistics
        (간단한 버전 - 나중에 GNN pre-training으로 업그레이드 가능)
        """
        embeddings = []
        
        for graph in tqdm(graphs, desc="Computing embeddings"):
            # Graph statistics as features
            edges = graph['edges']
            num_nodes = graph['num_nodes']
            num_edges = graph['num_edges']
            
            # Build networkx graph for analysis
            G = nx.Graph()
            if num_nodes > 0:
                G.add_nodes_from(range(num_nodes))
            if len(edges) > 0:
                G.add_edges_from(edges)
            
            # Compute features
            features = [
                num_nodes,
                num_edges,
                num_edges / max(num_nodes, 1),  # Edge density
                nx.number_connected_components(G),
                np.mean([d for _, d in G.degree()]) if num_nodes > 0 else 0,
                np.std([d for _, d in G.degree()]) if num_nodes > 0 else 0,
                nx.diameter(G) if nx.is_connected(G) and num_nodes > 0 else 0,
                nx.average_clustering(G) if num_nodes > 0 else 0
            ]
            
            embeddings.append(features)
        
        return np.array(embeddings)


    def _build_knn_edges(self, embeddings):
        """Build k-NN graph based on embedding similarity"""
        # Normalize embeddings
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        embeddings_norm = scaler.fit_transform(embeddings)
        
        # Compute similarity matrix
        sim_matrix = cosine_similarity(embeddings_norm)
        
        # Build k-NN edges
        edges = []
        for i in range(len(embeddings)):
            # Get top-k most similar (excluding self)
            similarities = sim_matrix[i]
            similarities[i] = -1  # Exclude self
            top_k_indices = np.argsort(similarities)[-self.k:]
            
            for j in top_k_indices:
                if similarities[j] > 0.5:  # Threshold
                    edges.append([i, j])
        
        edge_index = np.array(edges).T if edges else np.zeros((2, 0), dtype=int)
        return torch.tensor(edge_index, dtype=torch.long)


    def _build_label_edges(self, labels):
        """Build edges between contracts with same label"""
        edges = []
        label_to_contracts = defaultdict(list)
        
        # Group contracts by label
        for idx, label in enumerate(labels):
            label_to_contracts[label].append(idx)
        
        # Connect contracts within same label group
        for label, contract_indices in label_to_contracts.items():
            # Create clique (fully connected)
            for i in range(len(contract_indices)):
                for j in range(i + 1, len(contract_indices)):
                    edges.append([contract_indices[i], contract_indices[j]])
                    edges.append([contract_indices[j], contract_indices[i]])  # Undirected
        
        edge_index = np.array(edges).T if edges else np.zeros((2, 0), dtype=int)
        return torch.tensor(edge_index, dtype=torch.long)


    def _build_hybrid_edges(self, embeddings, labels):
        """Combine k-NN and label-based edges"""
        # Get k-NN edges
        knn_edges = self._build_knn_edges(embeddings)
        
        # Get label edges (with reduced connectivity)
        label_to_contracts = defaultdict(list)
        for idx, label in enumerate(labels):
            label_to_contracts[label].append(idx)
        
        label_edges = []
        for label, contract_indices in label_to_contracts.items():
            # Only connect to k random neighbors within same label
            for i in contract_indices:
                if len(contract_indices) > 1:
                    neighbors = [c for c in contract_indices if c != i]
                    k_label = min(3, len(neighbors))  # Max 3 label neighbors
                    selected = np.random.choice(neighbors, k_label, replace=False)
                    for j in selected:
                        label_edges.append([i, j])
        
        label_edges = torch.tensor(np.array(label_edges).T if label_edges 
                                   else np.zeros((2, 0), dtype=int), dtype=torch.long)
        
        # Combine edges (remove duplicates)
        all_edges = torch.cat([knn_edges, label_edges], dim=1)
        edge_set = set(map(tuple, all_edges.t().numpy()))
        edge_index = torch.tensor(list(map(list, edge_set)), dtype=torch.long).t()
        
        return edge_index


    def _analyze_contract_graph(self, edge_index, labels):
        """Print statistics about the contract graph"""
        print("\n" + "=" * 60)
        print("📊 Contract Graph Statistics")
        print("=" * 60)
        
        num_contracts = len(self.contract_to_idx)
        num_edges = edge_index.shape[1]
        
        print(f"Number of contracts (nodes): {num_contracts}")
        print(f"Number of edges: {num_edges}")
        if num_contracts > 0:
            print(f"Average degree: {2 * num_edges / num_contracts:.2f}")
        
        # Degree distribution
        degrees = torch.zeros(num_contracts, dtype=torch.long)
        for i in range(num_edges):
            src, dst = edge_index[:, i]
            degrees[src] += 1
            degrees[dst] += 1
        
        print(f"Degree stats: min={degrees.min()}, max={degrees.max()}, "
              f"mean={degrees.float().mean():.2f}")
        
        # Label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        print(f"\nLabel distribution:")
        for label, count in zip(unique_labels, counts):
            print(f"  Label {label}: {count} contracts")


    def _compute_degree_features(self, num_nodes, edge_index):
        """
        Feature가 없을 때 대체제로 사용할 Node Degree 계산
        """
        degree = torch.zeros(num_nodes, 1)

        # 엣지가 없는 경우 바로 반환
        if edge_index.numel() == 0:
            return degree

        # edge_index의 각 엣지(src, dst)에 대해 차수 증가
        # JSON의 엣지가 [u, v] 형태로 저장되어 있고, 무방향성을 가정하여
        # 양쪽 노드 모두 카운트합니다.
        for src, dst in edge_index.t().tolist():
            degree[src] += 1
            degree[dst] += 1

        return degree
    

    def _save_contract_graph(self, edge_index, method, graphs, embeddings, save_path):
        from torch_geometric.data import Data
        """Save contract graph to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data_objects = []
        labels = []

        for graph_dict in graphs:
            # ✅ 1. 미리 저장해둔 Label 사용
            label = graph_dict.get('label', 0)
            labels.append(label)

            # PyG Data 객체 생성
            edge_index_tensor = torch.tensor(graph_dict['edges'], dtype=torch.long).t()
            num_nodes = graph_dict['num_nodes']
            
            # ✅ 2. Features 처리 (size 체크)
            # numpy array.size를 사용하여 0차원 배열 오류 방지
            feat_arr = graph_dict['features']
            
            # ✅ 수정: size가 0보다 크더라도 dtype이 object면 건너뜀 (안전 장치)
            if feat_arr.size > 0 and feat_arr.dtype != np.object_:
                try:
                    x = torch.tensor(feat_arr, dtype=torch.float)
                except Exception:
                    # 변환 실패 시 degree feature 사용
                    x = self._compute_degree_features(num_nodes, edge_index_tensor)
            else:
                # Feature가 없거나 깨진 경우 -> Degree를 Feature로 사용
                x = torch.zeros(num_nodes, 1) # 초기화
                # Degree 계산 로직 (간단화)
                degree = torch.zeros(num_nodes, 1)
                for src, dst in edge_index_tensor.t().tolist():
                    degree[src] += 1
                    degree[dst] += 1
                x = degree

            data_objects.append(Data(x=x, edge_index=edge_index_tensor, y=label))

        # 메타데이터 생성
        # 파일 리스트가 필요하면 다시 glob을 하거나, _load_graphs에서 파일명도 저장했어야 함
        # 여기서는 단순히 인덱스로 매핑
        json_files = sorted(Path(self.data_dir).glob("*.json"))
        contract_to_idx = {str(f): i for i, f in enumerate(json_files)}
        idx_to_contract = {i: str(f) for i, f in enumerate(json_files)}
        labels_tentor = torch.tensor(labels, dtype=torch.long)

        torch.save({
            'edge_index': edge_index,
            'contract_to_idx': contract_to_idx,
            'idx_to_contract': idx_to_contract,
            'method': method,
            'embeddings': embeddings,
            'labels': labels_tentor, # ✅ Tensor로 저장
            'num_nodes': len(graphs),
            'k': getattr(self, 'k', None)
            # 'k': self.k
        }, save_path)
        

        # 통계 출력
        print(f"\n📊 Contract Graph Statistics")
        print("="*60)
        label_counts = torch.bincount(labels_tentor)
        print(f"Label distribution:")
        for i, count in enumerate(label_counts):
            print(f"  Label {i}: {count.item()} contracts")
        print(f"✅ Saved contract graph to {save_path}")


# ============================================================
# CLI Script
# ============================================================
import argparse

if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=str, required=True, help='Chain name (e.g., polygon)')
    parser.add_argument('--method', type=str, default='hybrid', choices=['knn', 'label', 'hybrid'], help='Method to build contract graph')
    parser.add_argument('--k_neighbors', type=int, default=5, help='Number of neighbors for k-NN')
    args = parser.parse_args()
    
    # Build contract graph
    builder = ContractGraphBuilder(
        data_dir=f"../../_data/GoG/{args.chain}/graphs",
        method=args.method,
        k_neighbors=args.k_neighbors
    )
    
    edge_index, contract_to_idx = builder.build(save_path=f'../../_data/GoG/{args.chain}/{args.chain}_contract_graph.pt')
    
    print("\n✅ Contract graph construction complete!")
    print(f"Use this graph for Hierarchical GNN training")
