import os
import glob

def merge_markdown_files(base_dir, output_filepath):
    """
    지정된 디렉토리 내에서 00부터 10으로 시작하는 폴더를 찾아
    내부의 모든 .md 파일을 하나의 파일로 병합합니다.
    """
    # 00부터 10까지의 접두사 리스트 생성 (['00-', '01-', ..., '10-'])
    # target_prefixes = [f"{i:02d}-" for i in range(11)]
    target_prefixes = [f"{i:02d}-" for i in range(11, 17)]  # 11부터 16까지로 수정
    
    # 병합할 문서들을 열기 (덮어쓰기 모드)
    with open(output_filepath, 'w', encoding='utf-8') as outfile:
        outfile.write("# 00 ~ 10 병합된 작업 보고서\n\n")
        
        # 디렉토리 내의 항목들을 알파벳/숫자 순으로 정렬하여 탐색
        for folder_name in sorted(os.listdir(base_dir)):
            folder_path = os.path.join(base_dir, folder_name)
            
            # 항목이 디렉토리인지, 그리고 타겟 접두사(00~10)로 시작하는지 확인
            if os.path.isdir(folder_path) and any(folder_name.startswith(p) for p in target_prefixes):
                
                # 가독성을 위해 폴더명을 1단계 제목으로 추가
                outfile.write(f"\n# Directory: {folder_name}\n\n")
                
                # 해당 폴더 내의 모든 .md 파일 검색 및 정렬
                md_files = sorted(glob.glob(os.path.join(folder_path, "*.md")))
                
                if not md_files:
                    outfile.write("*이 폴더에는 markdown 파일이 없습니다.*\n\n")
                    continue
                
                for md_file in md_files:
                    file_name = os.path.basename(md_file)
                    
                    # 파일명을 2단계 제목으로 추가
                    outfile.write(f"## File: {file_name}\n\n")
                    
                    # 개별 md 파일의 내용을 읽어서 병합 파일에 쓰기
                    with open(md_file, 'r', encoding='utf-8') as infile:
                        content = infile.read()
                        outfile.write(content)
                        
                    # 파일 간 구분을 위한 줄바꿈 및 구분선 추가
                    outfile.write("\n\n---\n\n")
                    
    print(f"✅ 성공적으로 병합되었습니다! 출력 파일: {output_filepath}")

# 실행 설정
if __name__ == "__main__":
    # 작업 디렉토리 설정 (필요에 따라 절대 경로로 수정 가능)
    WORK_DIR = "docs/work_reports" 
    # OUTPUT_FILE = "docs/work_reports/Merged_Reports_00_to_10.md"
    OUTPUT_FILE = "docs/work_reports/Merged_Reports_11_to_16.md"
    
    merge_markdown_files(WORK_DIR, OUTPUT_FILE)