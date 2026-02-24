"""
Common Node Analysis for Multi-Chain Blockchain Data
Author: Sam
Description: Extract and analyze common nodes across blockchain transaction files
"""

import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Set, Dict
import sys
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_common_node_file(file_path: str) -> Set[str]:
    """
    CSV 파일에서 노드(주소)를 추출
    
    Args:
        file_path: CSV 파일 경로
        
    Returns:
        정규화된 노드 주소의 집합
    """
    try:
        # ✅ 데이터 타입 문제 해결: 모든 컬럼을 문자열로 읽기
        df = pd.read_csv(
            file_path,
            dtype=str,              # 모든 컬럼을 문자열로 처리
            low_memory=False,       # 전체 파일을 한 번에 로드하여 타입 추론 일관성 유지
            on_bad_lines='skip',    # 잘못된 형식의 행 건너뛰기
            na_values=['', 'NA', 'null', 'None']  # 결측값 처리
        )
        
        nodes = set()
        
        # ✅ 다양한 컬럼 이름 지원
        from_columns = ['from_address', 'from', 'sender', 'fromAddress', 'From']
        to_columns = ['to_address', 'to', 'receiver', 'toAddress', 'To']
        
        # from 컬럼 찾기
        from_col = None
        for col in from_columns:
            if col in df.columns:
                from_col = col
                break
        
        if from_col:
            from_addrs = df[from_col].dropna()
            # ✅ 주소 정규화: 소문자 변환 및 공백 제거
            from_addrs = from_addrs.str.lower().str.strip()
            # ✅ 빈 문자열 및 0x0000... (mint/burn) 주소 제외
            from_addrs = from_addrs[
                (from_addrs != '') & 
                (from_addrs != '0x0000000000000000000000000000000000000000')
            ]
            nodes.update(from_addrs.unique())
        else:
            logger.debug(f"No 'from' column found in {Path(file_path).name}")
        
        # to 컬럼 찾기
        to_col = None
        for col in to_columns:
            if col in df.columns:
                to_col = col
                break
        
        if to_col:
            to_addrs = df[to_col].dropna()
            # ✅ 주소 정규화
            to_addrs = to_addrs.str.lower().str.strip()
            to_addrs = to_addrs[
                (to_addrs != '') & 
                (to_addrs != '0x0000000000000000000000000000000000000000')
            ]
            nodes.update(to_addrs.unique())
        else:
            logger.debug(f"No 'to' column found in {Path(file_path).name}")
        
        if not from_col and not to_col:
            logger.warning(f"No address columns found in {Path(file_path).name}")
            logger.warning(f"Available columns: {', '.join(df.columns.tolist())}")
        
        return nodes
        
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        return set()
    except pd.errors.EmptyDataError:
        logger.warning(f"Empty file: {file_path}")
        return set()
    except Exception as e:
        logger.error(f"Error reading {file_path}: {str(e)}")
        return set()


def find_common_nodes(data_dir: Path, chain: str) -> Set[str]:
    """
    특정 체인의 모든 CSV 파일에서 공통 노드 찾기
    
    Args:
        data_dir: 데이터 디렉토리 경로
        chain: 블록체인 이름 (bsc, eth, etc.)
        
    Returns:
        공통 노드 주소의 집합
    """
    chain_dir = data_dir / chain
    
    if not chain_dir.exists():
        logger.error(f"Chain directory not found: {chain_dir}")
        return set()
    
    # CSV 파일 목록 가져오기
    csv_files = list(chain_dir.glob("*.csv"))
    
    if not csv_files:
        logger.error(f"No CSV files found in {chain_dir}")
        return set()
    
    logger.info(f"Found {len(csv_files)} CSV files for chain: {chain}")
    
    # 첫 번째 파일의 노드로 초기화
    common_nodes = None
    
    # 진행률 표시와 함께 파일 처리
    for csv_file in tqdm(csv_files, desc=f"Finding common nodes ({chain})", unit="file"):
        file_nodes = get_common_node_file(str(csv_file))
        
        if common_nodes is None:
            common_nodes = file_nodes
        else:
            # ✅ 집합 교집합 연산으로 공통 노드 추출
            common_nodes = common_nodes.intersection(file_nodes)
        
        # ✅ 조기 종료하지 않음 - 공통 노드가 없어도 모든 파일 처리
        if not common_nodes:
            logger.debug(f"No common nodes remaining after {csv_file.name}")
    
    return common_nodes if common_nodes else set()


def analyze_node_frequency(data_dir: Path, chain: str) -> pd.DataFrame:
    """
    각 노드가 몇 개의 파일에 나타나는지 빈도 분석
    
    Args:
        data_dir: 데이터 디렉토리 경로
        chain: 블록체인 이름
        
    Returns:
        노드 빈도 DataFrame (node, frequency)
    """
    chain_dir = data_dir / chain
    csv_files = list(chain_dir.glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found for frequency analysis in {chain_dir}")
        return pd.DataFrame(columns=['node', 'frequency'])
    
    node_frequency = {}
    
    for csv_file in tqdm(csv_files, desc=f"Analyzing frequency ({chain})", unit="file"):
        file_nodes = get_common_node_file(str(csv_file))
        
        for node in file_nodes:
            node_frequency[node] = node_frequency.get(node, 0) + 1
    
    if not node_frequency:
        logger.warning(f"No nodes found in any files for {chain}")
        return pd.DataFrame(columns=['node', 'frequency'])
    
    # DataFrame 생성 및 정렬
    df = pd.DataFrame(
        list(node_frequency.items()),
        columns=['node', 'frequency']
    )
    df = df.sort_values('frequency', ascending=False).reset_index(drop=True)
    
    return df


def save_results(common_nodes: Set[str], frequency_df: pd.DataFrame, 
                 output_dir: Path, chain: str) -> Dict[str, Path]:
    """
    분석 결과를 파일로 저장
    
    Args:
        common_nodes: 공통 노드 집합
        frequency_df: 노드 빈도 DataFrame
        output_dir: 출력 디렉토리
        chain: 블록체인 이름
        
    Returns:
        저장된 파일 경로들의 딕셔너리
    """
    # 출력 디렉토리 생성
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 공통 노드 저장
#    common_nodes_file = output_dir / f"{chain}_common_nodes_{timestamp}.csv"
    common_nodes_file = output_dir / f"{chain}_common_nodes.csv"
    if common_nodes:
        pd.DataFrame(
            sorted(list(common_nodes)),
            columns=['address']
        ).to_csv(common_nodes_file, index=False)
        saved_files['common_nodes'] = common_nodes_file
        logger.info(f"Common nodes saved to: {common_nodes_file}")
    else:
        # 공통 노드가 없어도 빈 파일 생성
        pd.DataFrame(columns=['address']).to_csv(common_nodes_file, index=False)
        saved_files['common_nodes'] = common_nodes_file
        logger.info(f"Empty common nodes file created: {common_nodes_file}")
    
    # 빈도 분석 결과 저장
#    frequency_file = output_dir / f"{chain}_node_frequency_{timestamp}.csv"
    frequency_file = output_dir / f"{chain}_node_frequency.csv"
    if not frequency_df.empty:
        frequency_df.to_csv(frequency_file, index=False)
        saved_files['frequency'] = frequency_file
        logger.info(f"Frequency analysis saved to: {frequency_file}")
    else:
        # 빈도 데이터가 없어도 빈 파일 생성
        pd.DataFrame(columns=['node', 'frequency']).to_csv(frequency_file, index=False)
        saved_files['frequency'] = frequency_file
        logger.info(f"Empty frequency file created: {frequency_file}")
    
    # 요약 리포트 저장
#    summary_file = output_dir / f"{chain}_analysis_summary_{timestamp}.txt"
    summary_file = output_dir / f"{chain}_analysis_summary.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write(f"Common Node Analysis Summary for {chain.upper()}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"[Common Nodes]\n")
        f.write(f"Total common nodes: {len(common_nodes)}\n\n")
        
        if common_nodes:
            f.write("Sample common nodes (first 20):\n")
            for i, node in enumerate(sorted(list(common_nodes))[:20], 1):
                f.write(f"  {i}. {node}\n")
            if len(common_nodes) > 20:
                f.write(f"  ... and {len(common_nodes) - 20} more nodes\n")
        else:
            f.write("  No common nodes found across all files.\n")
        
        f.write("\n" + "-"*70 + "\n\n")
        
        f.write(f"[Frequency Analysis]\n")
        f.write(f"Total unique nodes: {len(frequency_df)}\n")
        
        if not frequency_df.empty:
            f.write(f"Max frequency: {frequency_df['frequency'].max()}\n")
            f.write(f"Mean frequency: {frequency_df['frequency'].mean():.2f}\n")
            f.write(f"Median frequency: {frequency_df['frequency'].median():.2f}\n\n")
            
            f.write("Top 20 most frequent nodes:\n")
            f.write(frequency_df.head(20).to_string(index=False))
        else:
            f.write("  No frequency data available.\n")
        
        f.write("\n\n" + "="*70 + "\n")
    
    saved_files['summary'] = summary_file
    logger.info(f"Summary report saved to: {summary_file}")
    
    return saved_files


def main():
    parser = argparse.ArgumentParser(
        description='Analyze common nodes in blockchain transaction data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python common_node.py --chain bsc
  python common_node.py --chain eth
  
Output files will be automatically saved to:
  ../../_data/results/graphs/{chain}/{chain}_common_nodes.csv
  ../../_data/results/graphs/{chain}/{chain}_node_frequency.csv
  ../../_data/results/graphs/{chain}/{chain}_analysis_summary.txt
        """
    )
    parser.add_argument(
        '--chain',
        type=str,
        required=True,
        help='Blockchain name (e.g., bsc, eth, polygon)'
    )
    
    args = parser.parse_args()
    
    # ✅ 자동 경로 설정 (chain 옵션만으로 모든 경로 결정)
    # transactions 폴더 사용 (processed 아님)
    data_dir = Path(__file__).parent.parent.parent / '_data'  / 'dataset' / 'transactions'
    output_dir = Path(__file__).parent.parent.parent / '_data' / 'graphs' / args.chain
    
    logger.info("="*70)
    logger.info(f"Starting Common Node Analysis")
    logger.info("="*70)
    logger.info(f"Chain: {args.chain.upper()}")
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*70)
    
    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        sys.exit(1)
    
    # ✅ Phase 1: 공통 노드 찾기 (Normal Mode)
    logger.info("\n[Phase 1/2] Finding common nodes...")
    common_nodes = find_common_nodes(data_dir, args.chain)
    
    if common_nodes:
        logger.info(f"✓ Found {len(common_nodes)} common nodes")
        print("\n" + "="*70)
        print(f"Common Nodes for {args.chain.upper()}")
        print("="*70)
        for i, node in enumerate(sorted(list(common_nodes)[:20]), 1):
            print(f"{i}. {node}")
        if len(common_nodes) > 20:
            print(f"... and {len(common_nodes) - 20} more nodes")
    else:
        # ✅ 공통 노드가 없어도 경고만 출력하고 계속 진행
        logger.warning("✗ No common nodes found across all files")
        logger.info("This is normal - it means no single address appears in ALL token files")
        logger.info("Continuing to frequency analysis...")
    
    # ✅ Phase 2: 빈도 분석 (Frequency Mode) - 공통 노드 여부와 관계없이 항상 실행
    logger.info("\n[Phase 2/2] Analyzing node frequency...")
    frequency_df = analyze_node_frequency(data_dir, args.chain)
    
    if not frequency_df.empty:
        logger.info(f"✓ Analyzed {len(frequency_df)} unique nodes")
        
        # 상위 노드 출력
        print("\n" + "="*70)
        print(f"Top Frequent Nodes for {args.chain.upper()}")
        print("="*70)
        print(frequency_df.head(20).to_string(index=False))
        
        # 통계 정보
        print("\n" + "="*70)
        print("Statistics")
        print("="*70)
        print(f"Total unique nodes: {len(frequency_df):,}")
        print(f"Max frequency: {frequency_df['frequency'].max():,}")
        print(f"Mean frequency: {frequency_df['frequency'].mean():.2f}")
        print(f"Median frequency: {frequency_df['frequency'].median():.2f}")
        
        # 분포 정보
        print(f"\nFrequency Distribution:")
        print(f"  Appears in > 50% files: {len(frequency_df[frequency_df['frequency'] > len(list((data_dir / args.chain).glob('*.csv'))) / 2]):,}")
        print(f"  Appears in > 25% files: {len(frequency_df[frequency_df['frequency'] > len(list((data_dir / args.chain).glob('*.csv'))) / 4]):,}")
        print(f"  Appears in > 10% files: {len(frequency_df[frequency_df['frequency'] > len(list((data_dir / args.chain).glob('*.csv'))) / 10]):,}")
    else:
        logger.warning("✗ No frequency data available")
    
    # ✅ 결과 저장
    logger.info("\n[Saving Results]")
    saved_files = save_results(common_nodes, frequency_df, output_dir, args.chain)
    
    # 최종 요약
    print("\n" + "="*70)
    print("Analysis Completed Successfully!")
    print("="*70)
    print(f"Chain: {args.chain.upper()}")
    print(f"Common nodes: {len(common_nodes):,}")
    print(f"Unique nodes: {len(frequency_df):,}")
    print("\nSaved files:")
    for file_type, file_path in saved_files.items():
        print(f"  - {file_type}: {file_path.name}")
        print(f"    {file_path}")
    print("="*70)
    
    logger.info("All tasks completed successfully!")


if __name__ == "__main__":
    main()
