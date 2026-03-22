"""Data reading utils."""

import os
import json
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
from torch_geometric.data import Data

def hierarchical_graph_reader(path, label_path=None):
    """
    하위 호환성을 유지하는 그래프 리더 함수.
    
    - label_path가 None인 경우: (기존 방식) CSV 엣지 리스트를 읽어 NetworkX 그래프 반환
    - label_path가 존재하는 경우: (main_mc.py 방식) JSON 폴더에서 피처를, CSV에서 라벨을 읽어 반환
    """
    
    # 1. 기존 방식 (Global Edges CSV만 읽을 때)
    if label_path is None:
        edges = pd.read_csv(path).values.tolist()
        graph = nx.from_edgelist(edges)
        return graph

    # 2. 새로운 방식 (JSON 피처와 Label CSV를 함께 읽을 때)
    else:
        print(f"Loading labels from {label_path}...")
        labels_df = pd.read_csv(label_path)
        
        # 파일 경로에서 체인명(bsc, polygon 등) 자동 추출
        # 예: path가 '../../_data/GoG/bsc/graphs' 라면 chain은 'bsc'
        chain_name = os.path.basename(os.path.dirname(path))
        
        if 'Chain' in labels_df.columns:
            labels_df = labels_df[labels_df['Chain'] == chain_name].reset_index(drop=True)
            
        label_col = 'label' if 'label' in labels_df.columns else 'Category'
        
        # 💡 핵심 수정: Contract 주소가 아닌 0, 1, 2 등의 숫자 인덱스로 매핑 생성
        if 'contract_id' in labels_df.columns:
            label_mapping = dict(zip(labels_df['contract_id'].astype(str), labels_df[label_col]))
        else:
            label_mapping = {str(idx): row[label_col] for idx, row in labels_df.iterrows()}
        
        features = []
        labels = []
        contract_ids = []
        
        json_files = glob.glob(os.path.join(path, '*.json'))
        
        for jf in tqdm(json_files, desc="Parsing JSON features"):
            # 파일명(예: 0.json)에서 확장자를 제외한 ID(0) 추출
            contract_id = os.path.splitext(os.path.basename(jf))[0]
            
            # 라벨이 존재하는 컨트랙트만 처리
            if contract_id not in label_mapping:
                continue
                
            with open(jf, 'r') as f:
                data = json.load(f)
                
            # JSON 파일 내부에 생성해둔 전역 컨트랙트 피처 추출
            if 'contract_feature' in data:
                feat = data['contract_feature']
            else:
                # 만약 contract_feature가 없다면 노드 피처들의 평균을 사용 (Fallback)
                node_feats = np.array(data.get('features', []))
                if len(node_feats) > 0:
                    feat = np.mean(node_feats, axis=0).tolist()
                else:
                    feat = [0.0] * 7  # 임시 빈 벡터
                    
            features.append(feat)
            labels.append(label_mapping[contract_id])
            contract_ids.append(contract_id)
            
        print(f"✅ Successfully loaded {len(features)} valid graphs.")
        return np.array(features), np.array(labels), contract_ids


class GraphDatasetGenerator(object):
    def __init__(self, path):
        self.df = pd.read_csv(path, low_memory=False)
        self.number_of_features = 7 
        self._create_target()

    def _create_target(self):
        self.target = torch.LongTensor(self.df['label']) if 'label' in self.df.columns else None

    def get_pyg_data_list(self):
        data_list = []
        for idx, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            node_features = torch.tensor([row['Num_nodes'], row['Num_edges'], row['Density'],
                                          row['Assortativity'], row['Reciprocity'], 
                                          row['Effective_Diameter'], row['Clustering_Coefficient']], 
                                          dtype=torch.float)
            

            data = Data(x=node_features.unsqueeze(0))  
            if self.target is not None:
                data.y = self.target[idx]
            data_list.append(data)
        return data_list