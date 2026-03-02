#!/usr/bin/env python3
"""
Generate local transaction graphs for each contract
Uses JSON files to map contract IDs to real addresses
"""

import torch
import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from torch_geometric.data import Data
import argparse
import warnings
import concurrent.futures
import multiprocessing

warnings.filterwarnings('ignore')


class LocalGraphGenerator:
    """Generate transaction-level graphs for each contract"""
    
    def __init__(self, 
                 tx_data_root="../../_data/dataset/transactions",
                 gog_root="../../_data/GoG",
                 chain='polygon'):
        self.chain = chain
        self.tx_data_root = Path(tx_data_root)
        self.gog_root = Path(gog_root)
        
        self.tx_dir = self.tx_data_root / chain
        self.contract_graph_path = self.gog_root / chain / f"{chain}_hybrid_graph.pt"
        self.json_dir = self.gog_root / chain / "graphs" # JSON files directory
        self.output_dir = self.gog_root / chain / "local_graphs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📂 Transaction directory: {self.tx_dir}")
        print(f"📂 Contract graph: {self.contract_graph_path}")
        print(f"📂 JSON directory: {self.json_dir}")
        print(f"📂 Output directory: {self.output_dir}")
    
    def load_json_address_mapping(self):
        """
        Load global mapping JSON to create mapping from contract_id to real address
        
        Returns:
            dict: {contract_id: real_address}
        """
        mapping_file = self.tx_data_root.parent / "global_graph" / f"{self.chain}_contract_to_number_mapping.json"
        
        print(f"\n📥 Loading global mapping from {mapping_file}...")
        
        id_to_address = {}
        
        if not mapping_file.exists():
            print(f"❌ Error: Mapping file not found at {mapping_file}")
            return id_to_address
            
        try:
            with open(mapping_file, 'r') as f:
                address_to_id = json.load(f)
            
            for address, contract_id in address_to_id.items():
                id_to_address[int(contract_id)] = address.lower()
                
            print(f"✅ Extracted {len(id_to_address)} contract addresses from mapping file")
            
            print(f"\n📋 Sample mappings:")
            for i, (cid, addr) in enumerate(list(id_to_address.items())[:5]):
                print(f"   Contract {cid}: {addr}")
                
        except Exception as e:
            print(f"❌ Error reading mapping file: {e}")
        
        return id_to_address
    
    def load_contract_info(self):
        """Load contract graph"""
        print(f"\n📥 Loading contract graph from {self.contract_graph_path}")
        
        contract_data = torch.load(self.contract_graph_path, weights_only=False)
        
        self.num_contracts = contract_data['num_nodes']
        self.global_labels = contract_data.get('labels')
        
        print(f"✅ Found {self.num_contracts} contracts in graph")
        
        return contract_data
    
    def find_transaction_files(self):
        """Find all transaction CSV files"""
        print(f"\n🔍 Searching for transaction CSV files...")
        
        if not self.tx_dir.exists():
            raise FileNotFoundError(f"Transaction directory not found: {self.tx_dir}")
        
        csv_files = list(self.tx_dir.glob("*.csv"))
        
        if len(csv_files) == 0:
            raise FileNotFoundError(f"No CSV files found in {self.tx_dir}")
        
        print(f"✅ Found {len(csv_files)} CSV files")
        
        return csv_files
    
    def extract_contract_address_from_filename(self, filepath):
        """Extract contract address from CSV filename"""
        return filepath.stem.lower()
    
    def build_address_to_csv_mapping(self, csv_files):
        """Build mapping from address to CSV file"""
        address_to_csv = {}
        for csv_file in csv_files:
            address = self.extract_contract_address_from_filename(csv_file)
            address_to_csv[address] = csv_file
        return address_to_csv
    
    def extract_features(self, df):
        """Extract node features from transaction DataFrame"""
        feature_list = []
        for _, row in df.iterrows():
            features = []
            try:
                features.append(float(row.get('value', 0)))
            except:
                features.append(0.0)
            try:
                features.append(float(row.get('timestamp', 0)))
            except:
                features.append(0.0)
            try:
                features.append(float(row.get('block_number', 0)))
            except:
                features.append(0.0)
            feature_list.append(features)
        return torch.tensor(feature_list, dtype=torch.float32)
    
    def build_local_graph_from_df(self, df, contract_id):
        """Build transaction graph from DataFrame"""
        num_txs = len(df)
        
        if num_txs == 0:
            return Data(
                x=torch.zeros((1, 3)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 1)),
                num_nodes=1,
                contract_id=contract_id
            )
        
        x = self.extract_features(df)
        edge_index = []
        edge_attr = []
        
        for i in range(num_txs - 1):
            edge_index.append([i, i + 1])
            edge_attr.append([1.0])
        
        window_size = min(20, num_txs)
        
        if 'from' in df.columns:
            from_addresses = df['from'].tolist()
            for i in range(num_txs):
                for j in range(i + 1, min(i + window_size, num_txs)):
                    if from_addresses[i] == from_addresses[j]:
                        edge_index.append([i, j])
                        edge_index.append([j, i])
                        edge_attr.append([0.5])
                        edge_attr.append([0.5])
        
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)
        
        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=num_txs,
            contract_id=contract_id
        )

    def _process_single_contract(self, args):
        """
        단일 컨트랙트의 로컬 그래프 생성을 처리하는 독립 메서드 (다중 프로세싱용)
        args: (contract_id, csv_file_path)
        """
        contract_id, csv_file = args
        
        if csv_file is not None:
            try:
                # pyarrow가 설치되어 있다면 engine='pyarrow'를 추가하여 읽기 속도를 더 향상시킬 수 있습니다.
                # 예: df = pd.read_csv(csv_file, engine='pyarrow')
                df = pd.read_csv(csv_file)
                local_graph = self.build_local_graph_from_df(df, contract_id)
                
                output_path = self.output_dir / f"{contract_id}.pt"
                torch.save(local_graph, output_path)
                
                return contract_id, len(df), True  # id, 트랜잭션 수, 매칭 여부
                
            except Exception as e:
                # 에러 발생 시 빈 그래프 생성으로 Fallback
                pass
                
        # 매칭되는 CSV가 없거나 에러가 발생한 경우 빈 그래프 생성
        empty_graph = Data(
            x=torch.zeros((1, 3)),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 1)),
            num_nodes=1,
            contract_id=contract_id
        )
        output_path = self.output_dir / f"{contract_id}.pt"
        torch.save(empty_graph, output_path)
        
        return contract_id, 0, False

    def generate_all_local_graphs(self, max_workers=0):
        """Generate local graphs for all contracts using Multiprocessing"""
        
        # 1. Load contract info
        contract_data = self.load_contract_info()
        
        # 2. Load JSON files to get real addresses
        id_to_address = self.load_json_address_mapping()
        
        # 3. Find CSV files
        csv_files = self.find_transaction_files()
        
        # 4. Build address to CSV mapping
        address_to_csv = self.build_address_to_csv_mapping(csv_files)
        
        print(f"\n📊 Matching analysis:")
        print(f"   Contracts in graph: {self.num_contracts}")
        print(f"   Addresses from JSON: {len(id_to_address)}")
        print(f"   CSV files: {len(csv_files)}")
        
        # 5. Check overlap
        json_addresses = set(id_to_address.values())
        csv_addresses = set(address_to_csv.keys())
        overlap = json_addresses & csv_addresses
        print(f"   Overlap: {len(overlap)} addresses can be matched")
        
        # 6. Generate graphs in Parallel
        print(f"\n🔨 Generating local graphs with multiprocessing...")
        
        # 병렬 처리를 위한 작업 리스트 구성
        tasks = []
        for contract_id in range(self.num_contracts):
            real_address = id_to_address.get(contract_id)
            if real_address and real_address in address_to_csv:
                csv_file = address_to_csv[real_address]
            else:
                csv_file = None
            tasks.append((contract_id, csv_file))
            
        matched = 0
        generated_graphs = {}
        
        # workers 수가 0이거나 지정되지 않은 경우 가용한 모든 CPU 코어 사용
        if max_workers <= 0:
            max_workers = multiprocessing.cpu_count() // 2  # 시스템 안정성을 위해 전체 코어의 절반만 사용
            
        print(f"⚡ Using {max_workers} CPU cores for parallel processing...")

        # ProcessPoolExecutor를 이용한 병렬 처리 실행
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # tqdm을 씌워 전체 진행률을 확인
            results = list(tqdm(
                executor.map(self._process_single_contract, tasks), 
                total=len(tasks), 
                desc="Processing contracts"
            ))

        # 결과 취합
        for contract_id, tx_count, is_matched in results:
            generated_graphs[contract_id] = tx_count
            if is_matched:
                matched += 1
        
        # Statistics
        empty_count = self.num_contracts - matched
        
        print(f"\n📊 Final Statistics:")
        print(f"  Total contracts: {self.num_contracts}")
        print(f"  Contracts with transactions: {matched}")
        print(f"  Contracts without transactions: {empty_count}")
        
        if matched > 0:
            tx_counts = [count for count in generated_graphs.values() if count > 0]
            print(f"\n📈 Transaction statistics:")
            print(f"  Avg transactions per contract: {np.mean(tx_counts):.2f}")
            print(f"  Max transactions: {np.max(tx_counts)}")
            print(f"  Min transactions: {np.min(tx_counts)}")
            print(f"  Total transactions: {sum(tx_counts):,}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=str, default='polygon')
    # workers 인자 추가 (0이면 최대 코어 사용)
    parser.add_argument('--workers', type=int, default=0, help='병렬 처리에 사용할 프로세스 수 (0: 모든 CPU 코어 사용)')
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 Local Transaction Graph Generator (Multiprocessing Enabled)")
    print("="*60)
    
    generator = LocalGraphGenerator(
        tx_data_root="../../_data/dataset/transactions",
        gog_root=f"../../_data/GoG",
        chain=args.chain
    )
    
    # workers 설정값 전달
    generator.generate_all_local_graphs(max_workers=args.workers)
    
    print("\n" + "="*60)
    print("✅ Complete!")
    print("="*60)


if __name__ == "__main__":
    main()