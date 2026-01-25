import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import pearsonr
import argparse
import igraph as ig
import sys

import warnings
warnings.filterwarnings('ignore')



parser = argparse.ArgumentParser(
    description='Analyze global_graph properties for a specified blockchain.',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
    Examples:
    python global.py --chain bsc
    python global.py --chain ethereum
    """
)
parser.add_argument(
    '--chain',
    type=str,
    required=True,
    help='Blockchain name (e.g., bsc, ethereum, polygon)'
)

args = parser.parse_args()
chain = args.chain


print(f'Loading labels for {chain}...')
labels = pd.read_csv(f'../../_data/dataset/labels.csv')
chain_labels = labels.query('Chain == @chain')
print(f'Found {len(chain_labels)} labels for {chain}')


print("Labels columns:", labels.columns.tolist())
print("Unique Chain values:", labels['Chain'].unique().tolist())
if chain_labels.empty:
    print(f"No labels for {chain}. Skipping detailed mapping.")
    class_mapping = {}
else:
    try:
        class_mapping = dict(zip(chain_labels['Contract'], chain_labels['Category']))
    except KeyError as e:
        print(f"Column error: {e}. Check labels.csv columns.")
        class_mapping = {}


print('Loading global_link...')
global_link = pd.read_csv(
    f'../../_data/graphs/{chain}/{chain}_common_nodes_except_null_labels.csv',
    dtype={'Common_Nodes': 'int64', 'Unique_Addresses': 'int64'}
)

print('Mapping classes...')
class_mapping = dict(zip(chain_labels['Contract'], chain_labels['Category']))
global_link['Class1'] = global_link['Contract1'].map(class_mapping).fillna('Unknown')
global_link['Class2'] = global_link['Contract2'].map(class_mapping).fillna('Unknown')

# Vectorized string shortening (optimized, no apply)
print('Shortening addresses...')
global_link['Contract1_short'] = global_link['Contract1'].str[:6] + '...' + global_link['Contract1'].str[-6:]
global_link['Contract2_short'] = global_link['Contract2'].str[:6] + '...' + global_link['Contract2'].str[-6:]

# Filter links
filtered_link = global_link[global_link['Class1'] != 'Unknown']
filtered_link = filtered_link[filtered_link['Class2'] != 'Unknown']

print('Building global_graph...')



# CLI 지원
parser = argparse.ArgumentParser(description='Global Graph Analysis with igraph')
parser.add_argument('--chain', type=str, default='BSC', help='Blockchain chain (e.g., BSC, bsc)')
args = parser.parse_args()
chain = args.chain.lower()  # 'BSC' -> 'bsc' (데이터셋 기준)

print(f'Loading labels for {chain}...')
labels_path = f'../../_data/dataset/labels.csv'
if not os.path.exists(labels_path):
    raise FileNotFoundError(f"labels.csv not found: {labels_path}")

labels = pd.read_csv(labels_path)
print("Labels columns:", labels.columns.tolist())
print("Unique Chain values:", labels['Chain'].unique().tolist())

chain_labels = labels.query('Chain == @chain')
print(f'Found {len(chain_labels)} labels for {chain}')

if chain_labels.empty:
    print(f"⚠️ No labels for {chain}. Exiting.")
    sys.exit(1)

# Class mapping (Contract/Category)
class_mapping = dict(zip(chain_labels['Contract'], chain_labels['Category']))

print('Loading global_link...')
global_link_path = f'../../_data/graphs/{chain}/{chain}_common_nodes_except_null_labels.csv'
if not os.path.exists(global_link_path):
    raise FileNotFoundError(f"Global link CSV not found: {global_link_path}")
global_link = pd.read_csv(
    global_link_path,
    dtype={'Common_Nodes': 'int64', 'Unique_Addresses': 'int64'}
)

print('Mapping classes...')
global_link['Class1'] = global_link['Contract1'].map(class_mapping).fillna('Unknown')
global_link['Class2'] = global_link['Contract2'].map(class_mapping).fillna('Unknown')

print('Shortening addresses...')
global_link['Contract1_short'] = global_link['Contract1'].str[:6] + '...' + global_link['Contract1'].str[-6:]
global_link['Contract2_short'] = global_link['Contract2'].str[:6] + '...' + global_link['Contract2'].str[-6:]

# Filter links
filtered_link = global_link[global_link['Class1'] != 'Unknown']
filtered_link = filtered_link[filtered_link['Class2'] != 'Unknown']
if filtered_link.empty:
    print("⚠️ No filtered links. Exiting.")
    sys.exit(1)





# ✅ 수정: 3-tuple로 weight 전달
print('Building global_graph with igraph (faster for large graphs)...')
sources = filtered_link['Contract1'].tolist()
targets = filtered_link['Contract2'].tolist()
common_nodes_orig = filtered_link['Common_Nodes'].tolist()

# ✅ Distance weight (shortest path용)
distance_weights = [1.0 / max(cn, 1) for cn in common_nodes_orig]

# 그래프 생성: distance weight를 'weight'로 사용 (igraph 기본)
edge_tuples = [(src, tgt, dist) for src, tgt, dist in zip(sources, targets, distance_weights)]
global_graph = ig.Graph.TupleList(edge_tuples, weights=True)

# ✅ 원본 Common_Nodes를 별도 edge attribute로 저장
global_graph.es['common_nodes'] = common_nodes_orig

print(f'Global graph built: {global_graph.vcount()} nodes, {global_graph.ecount()} edges')
print(f"Distance weight sample (for path): {global_graph.es['weight'][:5]}")
print(f"Common_Nodes sample (for centrality): {global_graph.es['common_nodes'][:5]}")





# Tx counts (병렬 처리)
cache_file = f'../../_data/transactions/{chain}_tx_counts.pkl'
chain_dir = f'../../_data/dataset/transactions/{chain}'

def count_tx(addr, chain_dir):
    try:
        tx_path = f'{chain_dir}/{addr}.csv'
        if os.path.exists(tx_path):
            tx = pd.read_csv(tx_path, dtype={'timestamp': 'int64'})
            tx['timestamp'] = pd.to_datetime(tx['timestamp'], unit='s')
            end_date = pd.Timestamp('2024-03-01')
            return addr, tx[tx['timestamp'] < end_date].shape[0]
    except Exception:
        pass
    return addr, 0

number_of_transactions = {}
if os.path.exists(cache_file):
    print("Loading cached tx counts...")
    with open(cache_file, 'rb') as f:
        number_of_transactions = pickle.load(f)
else:
    print("Computing tx counts in parallel (one-time)...")
    addrs = list(class_mapping.keys())
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 8)) as executor:
        futures = [executor.submit(count_tx, addr, chain_dir) for addr in tqdm(addrs, desc="Submit")]
        for future in tqdm(as_completed(futures), total=len(addrs), desc="Process"):
            addr, count = future.result()
            number_of_transactions[addr] = count
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'wb') as f:
        pickle.dump(number_of_transactions, f)
    print(f"Cached to {cache_file}")

# Add tx_count to nodes (igraph: vertex index로 매핑)
node_names = [v['name'] for v in global_graph.vs]
tx_counts = [number_of_transactions.get(name, 0) for name in node_names]
global_graph.vs['tx_count'] = tx_counts

# Metrics
density = global_graph.density()
print(f"Density: {density}")

# Avg shortest path (approx)
def approximate_average_shortest_path_length(G, num_landmarks=10):
    nodes = list(range(G.vcount()))
    landmarks = np.random.choice(nodes, min(num_landmarks, G.vcount()), replace=False)
    total_dist = 0
    count = 0
    for landmark in tqdm(landmarks, desc="Avg shortest path"):
        dists = G.distances(landmark, weights='weight')[0]  # ✅ weights='weight'
        finite_dists = [d for d in dists if np.isfinite(d)]
        total_dist += sum(finite_dists)
        count += len(finite_dists)
    return total_dist / count if count > 0 else 0

avg_shortest_path = approximate_average_shortest_path_length(global_graph)
print(f"Average shortest path length (approx): {avg_shortest_path}")

# Effective diameter
def approximate_effective_diameter(G, num_samples=10000, percentile=90):
    nodes = list(range(G.vcount()))
    path_lengths = []
    for _ in tqdm(range(num_samples), desc="Effective diameter"):
        if G.vcount() < 2:
            break
        src, dst = random.sample(nodes, 2)
        dist = G.distances(src, dst, weights='weight')[0][0]
        if np.isfinite(dist):
            path_lengths.append(dist)
    return np.percentile(path_lengths, percentile) if path_lengths else None

effective_diameter = approximate_effective_diameter(global_graph)
print(f"Effective diameter (approx): {effective_diameter}")

# Clustering & Diameter (igraph native - ultra fast!)
avg_clustering = global_graph.transitivity_avglocal_undirected()  # local clustering mean
print(f"Average clustering coefficient: {avg_clustering}")

print("Computing exact diameter (igraph fast)...")
diameter_exact = global_graph.diameter(weights='weight')
print(f"Exact weighted diameter: {diameter_exact}")




# Centrality
# Centrality: 두 종류 계산
unweighted_degree = global_graph.degree()
weighted_degree_distance = global_graph.strength(weights='weight')  # Distance weight 합
weighted_degree_common = global_graph.strength(weights='common_nodes')  # ✅ 원본 weight 합

top_5_unweighted = sorted(enumerate(unweighted_degree), key=lambda x: x[1], reverse=True)[:5]
top_5_weighted = sorted(enumerate(weighted_degree_common), key=lambda x: x[1], reverse=True)[:5]  # ✅ common_nodes 사용

print("Top 5 unweighted degree:", [(node_names[i], d) for i, d in top_5_unweighted])
print("Top 5 weighted degree (Common_Nodes):", [(node_names[i], d) for i, d in top_5_weighted])

# Correlation: unweighted vs weighted (common_nodes)
corr_degree, _ = pearsonr(unweighted_degree, weighted_degree_common)
print(f"Pearson correlation (unweighted vs weighted): {corr_degree}")






# Scatter plot: unweighted vs weighted (common_nodes)
plt.figure(figsize=(10, 8))
plt.scatter(unweighted_degree, weighted_degree_common, alpha=0.5)  # ✅ common_nodes
plt.xlabel('Unweighted Degree')
plt.ylabel('Weighted Degree (Common Nodes)')
plt.title(f'{chain.upper()}: Degree Centrality')
os.makedirs('../../_data/results/analysis', exist_ok=True)
plt.savefig(f'../../_data/results/analysis/{chain}_degree_scatter.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved: ../../_data/results/analysis/{chain}_degree_scatter.png")



# Edge weights histogram: 원본 Common_Nodes 사용
edge_weights_common = global_graph.es['common_nodes']  # ✅ 수정
plt.figure(figsize=(10, 6))
plt.hist(edge_weights_common, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Common Nodes (Edge Weight)')
plt.ylabel('Frequency')
plt.title(f'{chain.upper()}: Edge Weights Distribution')
plt.savefig(f'../../_data/results/analysis/{chain}_edge_weights.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Plot saved: ../../_data/results/analysis/{chain}_edge_weights.png")



# Monte Carlo Risk Estimation
def monte_carlo_risk_estimation(G, num_samples=1000):
    nodes = list(range(G.vcount()))
    risks = []
    for _ in range(num_samples):
        node = random.choice(nodes)
        tx_count = G.vs[node]['tx_count']
        degree = G.degree(node)
        risk = tx_count * degree
        risks.append((node, risk))
    risks_sorted = sorted(risks, key=lambda x: x[1], reverse=True)
    return {
        'mean_risk': np.mean([r for _, r in risks]),
        'p95_risk': np.percentile([r for _, r in risks], 95),
        'top_risk_nodes': [(node_names[i], r) for i, r in risks_sorted[:5]]
    }

mc_risk = monte_carlo_risk_estimation(global_graph)
print("MC Risk Estimation:")
print(f"  Mean risk: {mc_risk['mean_risk']}")
print(f"  P95 risk: {mc_risk['p95_risk']}")
print(f"  Top 5 risk nodes: {mc_risk['top_risk_nodes']}")

# Cache graph
graph_cache = f'../../_data/graphs/{chain}_global_graph.graphml'
global_graph.write_graphml(graph_cache)
print(f"Graph cached to {graph_cache}")

print("\n✅ Analysis complete!")
