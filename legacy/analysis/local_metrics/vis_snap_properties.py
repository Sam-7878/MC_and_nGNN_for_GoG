from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=str, default='polygon')
    return parser.parse_args()



args = get_args()
chain = args.chain

BASE_PATH = Path('../../_data/results/analysis/')
IMAGE_PATH = Path('../../_data/results/analysis/images/')

df = pd.read_csv(BASE_PATH / f'{chain}_snap_metrics_labels.csv')

# Edge/Node ratio 계산
df['Edge_Node_Ratio'] = df['Num_Edges'] / df['Num_Nodes']

# 토큰 유형 분류
def classify_token(row):
    ratio = row['Edge_Node_Ratio']
    diameter = row['Effective_Diameter']
    
    if ratio < 1.2:
        return 'Star (Hub)'
    elif ratio > 4:
        return 'Dense (Active)'
    elif diameter < 2.5:
        return 'Small-World'
    else:
        return 'Community'

df['Token_Type'] = df.apply(classify_token, axis=1)

# 분포 시각화
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Diameter 분포
df['Effective_Diameter'].hist(bins=50, ax=axes[0, 0])
axes[0, 0].set_title('Effective Diameter Distribution')
axes[0, 0].axvline(df['Effective_Diameter'].mean(), color='r', linestyle='--', label='Mean')
axes[0, 0].legend()

# 2. Clustering 분포
df['Clustering_Coefficient'].hist(bins=50, ax=axes[0, 1])
axes[0, 1].set_title('Clustering Coefficient Distribution')

# 3. Edge/Node Ratio
df['Edge_Node_Ratio'].hist(bins=50, ax=axes[1, 0], range=(0, 10))
axes[1, 0].set_title('Edge/Node Ratio Distribution')

# 4. Token Type 분포
df['Token_Type'].value_counts().plot(kind='bar', ax=axes[1, 1])
axes[1, 1].set_title('Token Type Distribution')
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(IMAGE_PATH / f'{chain}_snap_analysis.png', dpi=300)
print("✓ Visualization saved!")

# 통계 출력
print("\n=== Token Type Statistics ===")
print(df['Token_Type'].value_counts())
print("\n=== Token Type Metrics ===")
print(df.groupby('Token_Type')[['Effective_Diameter', 'Clustering_Coefficient']].mean())
