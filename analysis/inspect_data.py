"""
CSV 파일 구조 확인 스크립트
"""
import pandas as pd
from pathlib import Path

def inspect_csv_files(data_dir: Path, chain: str, num_samples: int = 3):
    """CSV 파일 구조 확인"""
    chain_dir = data_dir / chain
    print(f"Inspecting CSV files in directory: {chain_dir}")
    csv_files = list(chain_dir.glob("*.csv"))[:num_samples]
    
    print("="*70)
    print(f"Inspecting {len(csv_files)} sample CSV files from {chain_dir}")
    print("="*70)
    
    for i, csv_file in enumerate(csv_files, 1):
        print(f"\n[File {i}] {csv_file.name}")
        print("-"*70)
        
        try:
            # 첫 몇 줄만 읽기
            df = pd.read_csv(csv_file, nrows=5, dtype=str, low_memory=False)
            
            print(f"Shape: {df.shape}")
            print(f"\nColumns ({len(df.columns)}):")
            for idx, col in enumerate(df.columns):
                print(f"  {idx}: '{col}' - dtype: {df[col].dtype}")
            
            print(f"\nFirst 3 rows:")
            print(df.head(3).to_string())
            
            # from_address와 to_address 컬럼 확인
            if 'from_address' in df.columns:
                print(f"\n✓ 'from_address' column found")
                print(f"  Sample: {df['from_address'].iloc[0] if len(df) > 0 else 'N/A'}")
            else:
                print(f"\n✗ 'from_address' column NOT found")
                
            if 'to_address' in df.columns:
                print(f"✓ 'to_address' column found")
                print(f"  Sample: {df['to_address'].iloc[0] if len(df) > 0 else 'N/A'}")
            else:
                print(f"✗ 'to_address' column NOT found")
                
        except Exception as e:
            print(f"Error: {e}")
        
        print()

# 실행
data_dir = Path("../../_data/dataset/transactions")

inspect_csv_files(data_dir, "bsc", num_samples=3)
