import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=str, default='polygon')
    return parser.parse_args()


def main():
    args = get_args()
    chain = str(args.chain)
    
    graphs1 = pd.read_csv(f'../../_data/results/analysis/{chain}_basic_metrics.csv')
    graphs2 = pd.read_csv(f'../../_data/results/analysis/{chain}_snap_metrics_labels.csv')
    
    # ✅ 수정 포인트: Num_Nodes 등 대소문자가 다른 중복 컬럼 이름 통일
    graphs2.rename(columns={'Num_Nodes': 'Num_nodes', 'Num_Edges': 'Num_edges'}, inplace=True)
    
    # ✅ 수정 포인트: 병합 시 중복되는 컬럼(graphs2의 컬럼)에 '_drop' 접미사를 붙임
    features = pd.merge(graphs1, graphs2, on='Contract', suffixes=('', '_drop'))
    
    # ✅ '_drop' 접미사가 붙은 중복 컬럼들(graphs2에서 온 잉여 데이터)을 모두 삭제
    features = features.filter(regex='^(?!.*_drop)')
    
    labels = pd.read_csv('../../_data/dataset/labels.csv').query('Chain == @chain')
    labels['binary_category'] = labels['Category'].apply(lambda x: 1 if x == 0 else 0)
    label_dict = dict(zip(labels.Contract, labels.binary_category))
    
    features['label'] = features['Contract'].apply(lambda x: label_dict.get(x, 0))  # Default to 0 if not found
    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.fillna(0, inplace=True)

    ## ✅ 수정 포인트: 스케일링 전에 원본 데이터 저장 (필요 시)
    # features.to_csv(f'../../_data/dataset/features/{chain}_basic_metrics_before_processed.csv', index=False)

    scaler = StandardScaler()
    columns = ['Num_nodes', 'Num_edges', 'Density', 'Assortativity', 'Reciprocity', 'Effective_Diameter', 'Clustering_Coefficient']
    
    # 에러가 발생했던 스케일링 부분 (이제 _x, _y가 없으므로 정상 작동함)
    features[columns] = scaler.fit_transform(features[columns])

    features.to_csv(f'../../_data/dataset/features/{chain}_basic_metrics_processed.csv', index=False)
    print(f"Successfully processed and saved features for {chain} chain.")

if __name__ == '__main__':
    main()