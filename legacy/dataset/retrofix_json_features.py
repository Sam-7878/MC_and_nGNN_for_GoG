# retrofix_json_features.py
"""
기존 GoG JSON 파일의 features를 log1p 변환하는 일회성 스크립트.
gog.py를 개선 버전으로 재실행하는 것이 최선이지만,
임시방편으로 기존 파일을 in-place 수정합니다.
"""
import argparse
import json
import math
import os
from pathlib import Path
from tqdm import tqdm

CONTRACT_FEATURE_CLIP = 10.0


def safe_log1p(x):
    if not math.isfinite(x) or x < 0:
        return 0.0
    return math.log1p(x)


def already_transformed(features):
    """log1p 변환 여부 추정: 모든 값이 100 미만이면 이미 변환된 것으로 간주"""
    flat = [v for feat in features for v in feat]
    return all(abs(v) < 200 for v in flat if math.isfinite(v))


def fix_json_file(path: Path):
    with open(path, 'r') as f:
        data = json.load(f)

    features = data.get('features', [])
    if not features or already_transformed(features):
        return False  # 변환 불필요

    # log1p 변환
    new_features = []
    for feat in features:
        new_feat = [safe_log1p(v) for v in feat]
        new_feat = [0.0 if not math.isfinite(v) else v for v in new_feat]
        new_features.append(new_feat)
    data['features'] = new_features

    # contract_feature 클리핑
    cf = data.get('contract_feature', [])
    data['contract_feature'] = [
        0.0 if not math.isfinite(v)
        else max(-CONTRACT_FEATURE_CLIP, min(CONTRACT_FEATURE_CLIP, v))
        for v in cf
    ]

    with open(path, 'w') as f:
        json.dump(data, f)

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chain', type=str, default='polygon')
    args = parser.parse_args()

    chain = args.chain  
    GRAPHS_DIR = f"../../../_data/GoG/{chain}/graphs"
  
    json_files = list(Path(GRAPHS_DIR).glob("*.json"))
    print(f"Found {len(json_files)} JSON files in {GRAPHS_DIR}")

    fixed = 0
    for p in tqdm(json_files, desc="Fixing JSON files"):
        if fix_json_file(p):
            fixed += 1

    print(f"\n✅ Fixed {fixed} / {len(json_files)} files.")

