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
        self.contract_graph_path = self.gog_root / chain / f"{chain}_hybrid.pt"
        self.json_dir = self.gog_root / chain  # JSON files directory
        self.output_dir = self.gog_root / chain / "local_graphs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📂 Transaction directory: {self.tx_dir}")
        print(f"📂 Contract graph: {self.contract_graph_path}")
        print(f"📂 JSON directory: {self.json_dir}")
        print(f"📂 Output directory: {self.output_dir}")
    
    def load_json_address_mapping(self):
        """
        Load JSON files to create mapping from contract_id to real address
        
        Returns:
            dict: {contract_id: real_address}
        """
        print(f"\n📥 Loading JSON files to extract real addresses...")
        
        json_files = list(self.json_dir.glob("[0-9]*.json"))
        print(f"Found {len(json_files)} JSON files")
        
        id_to_address = {}
        
        for json_file in tqdm(json_files, desc="Reading JSON files"):
            # Extract contract ID from filename (e.g., "0.json" -> 0)
            try:
                contract_id = int(json_file.stem)
            except ValueError:
                continue
            
            # Read JSON
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                
                # Try common field names for contract address
                address = None
                for key in ['address', 'contract_address', 'contractAddress', 'addr']:
                    if key in data:
                        address = data[key]
                        break
                
                # If not found, check if there's a field with address-like value
                if address is None:
                    for key, value in data.items():
                        if isinstance(value, str) and len(value) == 42 and value.startswith('0x'):
                            address = value
                            break
                
                if address:
                    id_to_address[contract_id] = address.lower()
                else:
                    print(f"⚠️ No address found in {json_file.name}")
                    
            except Exception as e:
                print(f"❌ Error reading {json_file.name}: {e}")
        
        print(f"✅ Extracted {len(id_to_address)} contract addresses from JSON files")
        
        # Show samples
        print(f"\n📋 Sample mappings:")
        for i, (cid, addr) in enumerate(list(id_to_address.items())[:5]):
            print(f"   Contract {cid}: {addr}")
        
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
                value = float(row.get('value', 0))
                features.append(value)
            except:
                features.append(0.0)
            
            try:
                timestamp = float(row.get('timestamp', 0))
                features.append(timestamp)
            except:
                features.append(0.0)
            
            try:
                block = float(row.get('block_number', 0))
                features.append(block)
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
        
        # Extract features
        x = self.extract_features(df)
        
        # Build edges
        edge_index = []
        edge_attr = []
        
        # Temporal sequence
        for i in range(num_txs - 1):
            edge_index.append([i, i + 1])
            edge_attr.append([1.0])
        
        # Same sender connections (within window)
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
        
        # Convert to tensors
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
    
    def generate_all_local_graphs(self):
        """Generate local graphs for all contracts"""
        
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
        
        # 6. Generate graphs
        print(f"\n🔨 Generating local graphs...")
        
        matched = 0
        generated_graphs = {}
        
        for contract_id in tqdm(range(self.num_contracts), desc="Processing contracts"):
            # Get real address from JSON
            real_address = id_to_address.get(contract_id)
            
            if real_address and real_address in address_to_csv:
                # Found matching CSV file
                csv_file = address_to_csv[real_address]
                
                try:
                    df = pd.read_csv(csv_file)
                    local_graph = self.build_local_graph_from_df(df, contract_id)
                    
                    output_path = self.output_dir / f"{contract_id}.pt"
                    torch.save(local_graph, output_path)
                    
                    generated_graphs[contract_id] = len(df)
                    matched += 1
                    
                except Exception as e:
                    print(f"\n⚠️ Error processing contract {contract_id}: {e}")
            else:
                # No matching CSV - create empty graph
                empty_graph = Data(
                    x=torch.zeros((1, 3)),
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    edge_attr=torch.empty((0, 1)),
                    num_nodes=1,
                    contract_id=contract_id
                )
                
                output_path = self.output_dir / f"{contract_id}.pt"
                torch.save(empty_graph, output_path)
                
                generated_graphs[contract_id] = 0
        
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
    args = parser.parse_args()
    
    print("="*60)
    print("🚀 Local Transaction Graph Generator (JSON-based)")
    print("="*60)
    
    generator = LocalGraphGenerator(
        tx_data_root="../../_data/dataset/transactions",
        gog_root="../../_data/GoG",
        chain=args.chain
    )
    
    generator.generate_all_local_graphs()
    
    print("\n" + "="*60)
    print("✅ Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
