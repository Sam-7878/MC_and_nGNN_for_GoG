"""
Local Graph Analysis - Cross-Chain Comparison
Updated: 2026-01-15 (Category mapping added)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from pathlib import Path

warnings.filterwarnings("ignore")

# ==================== Configuration ====================

BASE_PATH = Path('../../_data/results/analysis/')
IMAGE_PATH = Path('../../_data/results/analysis/images/')
LABELS_PATH = Path('../../_data/dataset/labels.csv')

# ✅ Category 매핑 (프로젝트에 맞게 수정 필요!)
CATEGORY_MAPPING = {
    0: 'Normal',
    1: 'Scam',
    2: 'Pump_Dump',
    3: 'Ponzi',
    4: 'Phishing',
    5: 'Honeypot',
    6: 'Rug_Pull',
    7: 'Token_Abuse',
    8: 'Price_Manipulation',
    9: 'Wash_Trading',
    10: 'Front_Running',
    11: 'Sandwich_Attack',
    12: 'Flash_Loan_Attack',
    13: 'Oracle_Manipulation',
    14: 'Governance_Attack',
    15: 'MEV_Exploit',
    16: 'Liquidity_Theft',
    17: 'Exit_Scam',
    18: 'Fake_Token',
    19: 'Airdrop_Scam',
    20: 'Social_Engineering',
    # 필요한 만큼 추가
}

# 파일명 패턴
FILE_PATTERNS = {
    'nx': '{chain}_basic_metrics.csv',
    'snap': '{chain}_snap_metrics_labels.csv',
    'common': '{chain}_common_nodes.csv'
}

# 체인 이름 매핑
CHAIN_MAPPING = {
    'bsc': 'BSC',
    'eth': 'Ethereum',
    'ethereum': 'Ethereum',
    'polygon': 'Polygon',
    'matic': 'Polygon'
}

# ==================== Utility Functions ====================

def _normalize_contract_address(df: pd.DataFrame) -> pd.DataFrame:
    """Contract 주소 정규화"""
    if 'Contract' in df.columns:
        df['Contract'] = df['Contract'].astype(str).str.lower().str.strip()
    return df


def load_labels(labels_path: Path) -> pd.DataFrame:
    """
    Labels 파일 로드 및 Category 매핑
    
    Returns:
        pd.DataFrame: Contract, Chain, Label (텍스트) 정보
    """
    if not labels_path.exists():
        print(f"⚠️  Warning: Labels file not found at {labels_path}")
        return pd.DataFrame()
    
    try:
        labels_df = pd.read_csv(labels_path)
        
        # Contract 주소 정규화
        labels_df = _normalize_contract_address(labels_df)
        
        # ✅ Category → Label 매핑
        if 'Category' in labels_df.columns:
            # Category를 정수로 변환
            labels_df['Category'] = pd.to_numeric(labels_df['Category'], errors='coerce')
            
            # 매핑 적용
            labels_df['Label'] = labels_df['Category'].map(CATEGORY_MAPPING)
            
            # 매핑되지 않은 경우 Unknown_X 형식
            unmapped = labels_df['Label'].isna()
            if unmapped.any():
                labels_df.loc[unmapped, 'Label'] = labels_df.loc[unmapped, 'Category'].apply(
                    lambda x: f'Unknown_{int(x)}' if pd.notna(x) else 'Unknown'
                )
            
            print(f"✓ Labels loaded: {len(labels_df)} contracts")
            
            # 매핑 통계
            mapped_count = (~labels_df['Label'].str.startswith('Unknown', na=False)).sum()
            unmapped_count = (labels_df['Label'].str.startswith('Unknown', na=False)).sum()
            
            if unmapped_count > 0:
                print(f"⚠️  Warning: {unmapped_count} categories not mapped")
                print(f"   Unmapped categories: {sorted(labels_df[labels_df['Label'].str.startswith('Unknown', na=False)]['Category'].unique())}")
        
        # Chain 컬럼 정규화
        if 'Chain' in labels_df.columns:
            labels_df['Chain'] = labels_df['Chain'].str.lower()
        
        return labels_df[['Contract', 'Chain', 'Label', 'Category']]
        
    except Exception as e:
        print(f"❌ Error loading labels: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def find_file(base_path: Path, chain: str, file_type: str) -> Path:
    """파일 자동 탐지"""
    pattern = FILE_PATTERNS[file_type].format(chain=chain)
    file_path = base_path / pattern
    
    if file_path.exists():
        return file_path
    
    # Fallback
    possible_patterns = {
        'nx': [
            f'{chain}_basic_metrics.csv',
            f'{chain}_basic_metrics_maxd2000.csv',
        ],
        'snap': [
            f'{chain}_snap_metrics_labels.csv',
            f'{chain}_advanced_metrics_labels.csv',
        ]
    }
    
    if file_type in possible_patterns:
        for alt_pattern in possible_patterns[file_type]:
            alt_path = base_path / alt_pattern
            if alt_path.exists():
                return alt_path
    
    return None


def load_and_merge(chain_name: str, labels_df: pd.DataFrame, base_path: Path = BASE_PATH) -> pd.DataFrame:
    """NetworkX + SNAP + Labels 3-way merge"""
    chain_lower = chain_name.lower()
    chain_display = CHAIN_MAPPING.get(chain_lower, chain_name.capitalize())
    
    print(f"\n{'='*50}")
    print(f"Loading {chain_display}...")
    print(f"{'='*50}")
    
    # 파일 찾기
    nx_file = find_file(base_path, chain_lower, 'nx')
    snap_file = find_file(base_path, chain_lower, 'snap')
    
    if nx_file is None or snap_file is None:
        print(f"❌ Files not found for {chain_display}")
        return pd.DataFrame()
    
    print(f"✓ NetworkX: {nx_file.name}")
    print(f"✓ SNAP: {snap_file.name}")
    
    try:
        # 데이터 로드
        nx_df = pd.read_csv(nx_file)
        snap_df = pd.read_csv(snap_file)
        
        print(f"  - NetworkX rows: {len(nx_df)}")
        print(f"  - SNAP rows: {len(snap_df)}")
        
        # Contract 주소 정규화
        nx_df = _normalize_contract_address(nx_df)
        snap_df = _normalize_contract_address(snap_df)
        
        # NetworkX + SNAP 병합
        merged = pd.merge(nx_df, snap_df, on='Contract', how='inner', suffixes=('_nx', '_snap'))
        print(f"✓ NetworkX + SNAP merged: {len(merged)} rows")
        
        # Labels 병합
        if not labels_df.empty and 'Chain' in labels_df.columns:
            chain_labels = labels_df[labels_df['Chain'] == chain_lower].copy()
            
            if not chain_labels.empty:
                print(f"  - Labels for {chain_display}: {len(chain_labels)}")
                
                merged = pd.merge(
                    merged,
                    chain_labels[['Contract', 'Label', 'Category']],
                    on='Contract',
                    how='left'
                )
                
                # Label 없으면 Unknown
                if 'Label' in merged.columns:
                    merged['Label'] = merged['Label'].fillna('Unknown')
                    labeled_count = (merged['Label'] != 'Unknown').sum()
                    print(f"✓ Labeled contracts: {labeled_count}/{len(merged)} ({labeled_count/len(merged)*100:.1f}%)")
            else:
                merged['Label'] = 'Unknown'
        else:
            merged['Label'] = 'Unknown'
        
        # 컬럼 정리
        merged['Chain'] = chain_display
        merged['Class'] = merged.get('Label', 'Unknown')
        
        # Diameter 통일
        if 'Effective_Diameter_snap' in merged.columns:
            merged['Effective_Diameter'] = merged['Effective_Diameter_snap']
        elif 'Effective_Diameter_nx' in merged.columns:
            merged['Effective_Diameter'] = merged['Effective_Diameter_nx']
        
        # Clustering 통일
        if 'Clustering_Coefficient_snap' in merged.columns:
            merged['Clustering_Coefficient'] = merged['Clustering_Coefficient_snap']
        elif 'Clustering_Coefficient_nx' in merged.columns:
            merged['Clustering_Coefficient'] = merged['Clustering_Coefficient_nx']
        
        return merged
        
    except Exception as e:
        print(f"❌ Error loading {chain_display}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def print_statistics(df: pd.DataFrame, chain_name: str = "All"):
    """데이터 통계 출력"""
    print(f"\n{'='*50}")
    print(f"Statistics - {chain_name}")
    print(f"{'='*50}")
    
    if df.empty:
        print("No data available.")
        return
    
    print(f"Total Contracts: {len(df)}")
    
    if 'Class' in df.columns:
        print(f"\nClass Distribution (Top 10):")
        class_counts = df['Class'].value_counts().head(10)
        for cls, count in class_counts.items():
            print(f"  - {cls}: {count} ({count/len(df)*100:.1f}%)")
        
        # Unknown이 많으면 경고
        unknown_count = (df['Class'] == 'Unknown').sum()
        if unknown_count > len(df) * 0.1:
            print(f"\n⚠️  Warning: {unknown_count} ({unknown_count/len(df)*100:.1f}%) contracts have 'Unknown' label")
    
    # 메트릭 통계
    metrics = ['Num_nodes', 'Num_edges', 'Effective_Diameter', 'Clustering_Coefficient', 'Reciprocity']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if available_metrics:
        print(f"\nMetric Statistics:")
        stats = df[available_metrics].describe().loc[['mean', '50%', 'std']]
        print(stats.to_string())


# local.py 상단에 추가

def detect_category_scheme(labels_df: pd.DataFrame) -> dict:
    """
    실제 데이터의 Category 값을 분석하여 적절한 매핑 제안
    """
    unique_categories = sorted(labels_df['Category'].dropna().unique())
    num_categories = len(unique_categories)
    
    print(f"\n{'='*50}")
    print(f"Detected {num_categories} unique categories: {unique_categories}")
    print(f"{'='*50}")
    
    # 카테고리 수에 따라 매핑 제안
    if num_categories <= 5:
        print("→ Suggests: Basic classification (Normal, Phishing, Ponzi, Scam, Honeypot)")
        suggested_mapping = {
            0: 'Normal',
            1: 'Phishing',
            2: 'Ponzi',
            3: 'Scam',
            4: 'Honeypot',
        }
    elif num_categories <= 10:
        print("→ Suggests: Extended DeFi classification")
        suggested_mapping = {
            0: 'Normal',
            1: 'Phishing',
            2: 'Ponzi_Scheme',
            3: 'Honeypot',
            4: 'Balance_Disorder',
            5: 'Hidden_Transfer',
            6: 'Fake_Token',
            7: 'Rug_Pull',
            8: 'Pump_Dump',
            9: 'Other_Scam',
        }
    else:
        print("→ Suggests: Check original dataset documentation")
        suggested_mapping = {i: f'Category_{i}' for i in unique_categories}
    
    return suggested_mapping


# ==================== Main Analysis ====================
# main() 함수 시작 부분 수정
def main():
    print("="*70)
    print("Local Graph Analysis - Cross-Chain Comparison")
    print("="*70)
    
    # Labels 로드
    labels_df = load_labels(LABELS_PATH)
    
    if labels_df.empty:
        print("\n❌ Cannot load labels.csv")
        return
    
    # ✅ 자동 카테고리 감지 및 제안
    suggested_mapping = detect_category_scheme(labels_df)
    
    print("\n" + "="*50)
    print("Suggested Category Mapping:")
    print("="*50)
    for cat_id, cat_name in sorted(suggested_mapping.items()):
        count = len(labels_df[labels_df['Category'] == cat_id])
        print(f"  {cat_id}: {cat_name:20s} ({count:5d} contracts)")
    
    
    # Label 분포 미리 확인
    if 'Label' in labels_df.columns and 'Chain' in labels_df.columns:
        print("\n" + "="*50)
        print("Label Distribution (from labels.csv)")
        print("="*50)
        
        for chain in sorted(labels_df['Chain'].unique()):
            chain_data = labels_df[labels_df['Chain'] == chain]
            print(f"\n{chain.upper()}:")
            
            # 텍스트 라벨 분포
            label_dist = chain_data['Label'].value_counts().head(10)
            for label, count in label_dist.items():
                # Category 번호도 표시
                cat_num = chain_data[chain_data['Label'] == label]['Category'].iloc[0]
                print(f"  {label} (Cat {int(cat_num)}): {count}")
    
    # 각 체인 데이터 로드
    bsc_graphs = load_and_merge('bsc', labels_df, BASE_PATH)
    ethereum_graphs = load_and_merge('ethereum', labels_df, BASE_PATH)
    polygon_graphs = load_and_merge('polygon', labels_df, BASE_PATH)
    
    # 통계 출력
    print_statistics(bsc_graphs, "BSC")
    print_statistics(ethereum_graphs, "Ethereum")
    print_statistics(polygon_graphs, "Polygon")
    
    # 전체 병합
    all_dfs = [df for df in [bsc_graphs, ethereum_graphs, polygon_graphs] if not df.empty]
    
    if len(all_dfs) == 0:
        print("\n❌ No data loaded.")
        return
    
    graphs = pd.concat(all_dfs, ignore_index=True)
    print_statistics(graphs, "All Chains")
    
    # 시각화
    if 'Class' not in graphs.columns:
        print("\n⚠️  No class labels for visualization.")
        return
    
    # Unknown 제거
    graphs_labeled = graphs[graphs['Class'] != 'Unknown'].copy()
    
    if len(graphs_labeled) < 10:
        print(f"\n⚠️  Only {len(graphs_labeled)} labeled contracts.")
        graphs_labeled = graphs
    
    # Top 5 클래스
    top_classes = graphs_labeled['Class'].value_counts().head(5).index.tolist()
    graphs_filter = graphs_labeled.query('Class in @top_classes').copy()
    
    print(f"\n{'='*50}")
    print(f"Visualization - Top {len(top_classes)} Classes")
    print(f"{'='*50}")
    for i, cls in enumerate(top_classes, 1):
        count = len(graphs_filter[graphs_filter['Class'] == cls])
        print(f"{i}. {cls}: {count} contracts")
    
    # Figure 1: Class별 Boxplot
    metrics = ['Num_edges', 'Assortativity', 'Reciprocity',
               'Effective_Diameter', 'Clustering_Coefficient']
    available_metrics = [m for m in metrics if m in graphs_filter.columns]
    
    if len(available_metrics) == 0:
        print("\n❌ No metrics available.")
        return
    
    print(f"\nPlotting metrics: {available_metrics}")
    
    fig, axes = plt.subplots(nrows=1, ncols=len(available_metrics), 
                            figsize=(5*len(available_metrics), 5))
    
    if len(available_metrics) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    class_order = graphs_filter['Class'].value_counts().index
    
    for i, metric in enumerate(available_metrics):
        sns.boxplot(x='Class', y=metric, data=graphs_filter, 
                   order=class_order, ax=axes[i], palette='Set2')
        axes[i].set_title(f'{metric}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].tick_params(axis='x', rotation=45)
        
        if metric in ['Num_nodes', 'Num_edges', 'Effective_Diameter']:
            axes[i].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(IMAGE_PATH / 'class_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure 1 saved: {IMAGE_PATH / 'class_comparison.png'}")
    plt.show()
    
    # Figure 2: Chain별 Boxplot
    available_chains = [df for df in [bsc_graphs, ethereum_graphs, polygon_graphs] if not df.empty]
    
    if len(available_chains) < 2:
        print("\n⚠️  Skip Figure 2: Need at least 2 chains.")
        return
    
    def plot_grouped_boxplot(data_list, ax, metrics, labels, log=True):
        combined_data = pd.concat(data_list)
        valid_metrics = [m for m in metrics if m in combined_data.columns]
        
        if len(valid_metrics) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return
        
        melted_data = combined_data.melt(
            id_vars='Chain', 
            value_vars=valid_metrics, 
            var_name='Metric', 
            value_name='Value'
        )
        
        sns.boxplot(data=melted_data, x='Metric', y='Value', 
                   hue='Chain', ax=ax, palette='Set2')
        
        if log:
            ax.set_yscale('log')
        
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        valid_labels = [labels[i] for i, m in enumerate(metrics) if m in valid_metrics]
        ax.set_xticklabels(valid_labels, rotation=0)
        
        if ax.get_legend():
            ax.get_legend().remove()
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    plot_grouped_boxplot(available_chains, axes[0], 
                        ['Num_nodes', 'Num_edges'], 
                        ['Nodes', 'Edges'], True)
    axes[0].set_title('(a) Network Size', fontsize=14, fontweight='bold')
    
    plot_grouped_boxplot(available_chains, axes[1], 
                        ['Reciprocity', 'Clustering_Coefficient'], 
                        ['Reciprocity', 'Clustering'], False)
    axes[1].set_title('(b) Local Structure', fontsize=14, fontweight='bold')
    
    plot_grouped_boxplot(available_chains, axes[2], 
                        ['Assortativity'], 
                        ['Assortativity'], False)
    axes[2].set_title('(c) Mixing Pattern', fontsize=14, fontweight='bold')
    
    plot_grouped_boxplot(available_chains, axes[3], 
                        ['Effective_Diameter'], 
                        ['Diameter'], True)
    axes[3].set_title('(d) Global Distance', fontsize=14, fontweight='bold')
    
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', 
                  bbox_to_anchor=(0.5, 0.98), ncol=3, 
                  title='Blockchain', frameon=False)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(IMAGE_PATH / 'chain_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Figure 2 saved: {IMAGE_PATH / 'chain_comparison.png'}")
    plt.show()
    
    print("\n" + "="*70)
    print("Analysis completed successfully! ✓")
    print("="*70)


if __name__ == "__main__":
    main()
