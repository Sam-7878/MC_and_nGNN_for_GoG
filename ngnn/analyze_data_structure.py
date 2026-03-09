# ngnn/analyze_data_structure.py
# 먼저 현재 데이터가 hierarchical 구조를 지원하는지 확인이 필요합니다.
# Step 1.1: 데이터 구조 확인

import json
import os
from collections import defaultdict, Counter
from pathlib import Path

def analyze_gog_structure(data_dir="../../_data/GoG/polygon/graphs"):
    """현재 GoG 데이터 구조 분석"""
    
    json_files = list(Path(data_dir).glob("*.json"))
    n_json_files = len(json_files)
    print(f"📊 Total JSON files: {n_json_files}\n")
    
    # 샘플 파일 로드
    sample_file = json_files[0]
    with open(sample_file, 'r') as f:
        sample = json.load(f)
    
    print("=" * 60)
    print("🔍 Sample Data Structure")
    print("=" * 60)
    print(f"Keys: {sample.keys()}")
    # 수정 전: len(sample.get('nodes', []))
    # 수정 후: len(sample.get('features', []))
    print(f"\nNodes: {len(sample.get('features', []))} items")
    print(f"Edges: {len(sample.get('edges', []))} items")
    
    # 글로벌 엣지(Inter-contract edges)는 json 파일에서 찾지 않고 
    # 별도의 CSV에서 로드해야 한다는 점을 스크립트나 마인드셋에 반영해야 합니다.
    
    # Node 구조
    if sample.get('features'):
        print(f"\n📌 Sample Feature:")
        print(json.dumps(list(sample['features'])[0], indent=2))
    
    # Edge 구조
    if sample.get('edges'):
        print(f"\n📌 Sample Edge:")
        print(json.dumps(sample['edges'][0], indent=2))
    
    # 메타데이터
    print(f"\n📌 Metadata:")
    for key in ['Chain', 'Contract', 'Category', 'Split']:
        print(f"  {key}: {sample.get(key, 'N/A')}")
    
    print("\n" + "=" * 60)
    print("📈 Dataset Statistics")
    print("=" * 60)
    
    # 전체 통계
    contracts = set()
    categories = Counter()
    node_counts = []
    edge_counts = []
    
    for file in json_files[:1000]:  # 샘플 1000개만
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            contracts.add(data.get('Contract', 'unknown'))
            categories[data.get('Category', 'unknown')] += 1
            node_counts.append(len(data.get('features', [])))
            edge_counts.append(len(data.get('edges', [])))
        except:
            continue
    
    print(f"Unique Contracts: {len(contracts)}")
    print(f"\nCategory Distribution:")
    for cat, count in categories.most_common():
        print(f"  {cat}: {count}")
    
    import numpy as np
    print(f"\nGraph Size Statistics:")
    print(f"  Nodes: mean={np.mean(node_counts):.1f}, "
          f"median={np.median(node_counts):.1f}, "
          f"max={np.max(node_counts)}")
    print(f"  Edges: mean={np.mean(edge_counts):.1f}, "
          f"median={np.median(edge_counts):.1f}, "
          f"max={np.max(edge_counts)}")
    
    print("\n" + "=" * 60)
    print("🔗 Contract Relationship Analysis")
    print("=" * 60)
    
    # Contract 간 관계 분석
    contract_interactions = defaultdict(set)
    
    for file in json_files[:1000]:
        try:
            with open(file, 'r') as f:
                data = json.load(f)
            
            contract = data.get('Contract')
            # Edge에서 다른 컨트랙트 찾기 (있다면)
            for edge in data.get('edges', []):
                if 'to_contract' in edge:
                    contract_interactions[contract].add(edge['to_contract'])
        except:
            continue
    
    if contract_interactions:
        avg_interactions = np.mean([len(v) for v in contract_interactions.values()])
        print(f"Contracts with interactions: {len(contract_interactions)}")
        print(f"Average interactions per contract: {avg_interactions:.2f}")
    else:
        print("⚠️  No inter-contract edges found in current data")
        print("   → Need to build contract graph separately")
    
    return {
        'has_inter_contract_edges': bool(contract_interactions),
        'num_contracts': len(contracts),
        'categories': dict(categories)
    }


import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nested GNN Training with Enhanced Features")
    parser.add_argument('--chain', type=str, required=True)
    args = parser.parse_args()

    result = analyze_gog_structure(f"../../_data/GoG/{args.chain}/graphs")
    
    print("\n" + "=" * 60)
    print("✅ Analysis Complete")
    print("=" * 60)
    print(f"Inter-contract edges exist: {result['has_inter_contract_edges']}")
    
    if not result['has_inter_contract_edges']:
        print("\n💡 Next Step: Build contract graph from transaction patterns")
        print("   We'll create edges between contracts based on:")
        print("   - Direct fund transfers")
        print("   - Shared addresses")
        print("   - Temporal proximity")
