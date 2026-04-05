from pathlib import Path
import argparse
import sys
import os
import fnmatch


def count_lines_in_file(file_path: Path) -> int:
    """
    파일의 전체 라인 수를 반환합니다.
    인코딩 이슈가 있더라도 최대한 읽을 수 있도록 errors='ignore' 사용.
    """
    try:
        with file_path.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"[WARN] 파일 읽기 실패: {file_path} ({e})", file=sys.stderr)
        return 0


def parse_gitignore(root: Path) -> list[str]:
    """
    루트 디렉토리의 .gitignore 파일을 읽어서 패턴 리스트를 반환합니다.
    """
    gitignore_path = root / ".gitignore"
    patterns = [".git"]  # 기본적으로 Git 내부 폴더는 제외
    
    if gitignore_path.exists():
        with gitignore_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 빈 줄이거나 주석(#)인 경우 무시
                if line and not line.startswith("#"):
                    patterns.append(line)
    return patterns


def should_ignore(rel_path: str, is_dir: bool, patterns: list[str]) -> bool:
    """
    주어진 상대 경로가 .gitignore 패턴에 매칭되는지 확인합니다.
    """
    for p in patterns:
        p_clean = p.strip('/')
        name = rel_path.split('/')[-1]

        # 1. 파일명/폴더명 자체가 패턴과 일치하는 경우 (예: *.csv, .venv)
        if fnmatch.fnmatch(name, p_clean):
            return True

        # 2. 전체 상대 경로와 정확히 일치하는 경우 (예: docs/work_reports)
        if fnmatch.fnmatch(rel_path, p_clean):
            return True

        # 3. 특정 디렉토리 하위의 모든 파일 매칭 (예: dataset/test/ 하위 파일들)
        if '/' in p_clean and rel_path.startswith(p_clean + '/'):
            return True

        # 4. **/__pycache__ 와 같은 다중 디렉토리 와일드카드 처리
        if p.startswith("**/"):
            rule = p[3:].strip('/')
            if fnmatch.fnmatch(name, rule) or fnmatch.fnmatch(rel_path, rule):
                return True

    return False


def collect_target_files(root: Path, extensions: set[str]) -> list[Path]:
    """
    root 이하의 모든 하위 폴더를 탐색하며 .gitignore 규칙을 적용해 대상 파일을 수집합니다.
    """
    files = []
    patterns = parse_gitignore(root)

    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = Path(dirpath).relative_to(root).as_posix()
        if rel_dir == ".":
            rel_dir = ""

        # 디렉토리 필터링: 무시할 폴더는 미리 리스트에서 제외하여 하위 탐색을 방지 (성능 최적화)
        valid_dirs = []
        for d in dirnames:
            d_rel = f"{rel_dir}/{d}".strip("/") if rel_dir else d
            if not should_ignore(d_rel, is_dir=True, patterns=patterns):
                valid_dirs.append(d)
        dirnames[:] = valid_dirs  # os.walk의 in-place 수정 기능 활용

        # 파일 필터링
        for f in filenames:
            f_rel = f"{rel_dir}/{f}".strip("/") if rel_dir else f
            if not should_ignore(f_rel, is_dir=False, patterns=patterns):
                p = Path(dirpath) / f
                if p.suffix.lower() in extensions:
                    files.append(p)

    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="지정한 폴더 이하의 .py, .yaml 파일 총 라인 수를 계산합니다. (.gitignore 적용)"
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=".",
        help="집계할 대상 폴더 경로 (기본값: 현재 working folder)"
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="파일별 출력 없이 총합만 출력"
    )

    args = parser.parse_args()
    root = Path(args.folder).resolve()

    if not root.exists():
        print(f"[ERROR] 폴더가 존재하지 않습니다: {root}", file=sys.stderr)
        sys.exit(1)

    if not root.is_dir():
        print(f"[ERROR] 디렉터리가 아닙니다: {root}", file=sys.stderr)
        sys.exit(1)

    extensions = {".py", ".yaml", ".md", ".json", ".sh", ".txt", ".ts", ".tsx", ".jsx", ".java", ".go", ".cpp", ".c", ".h", ".rb", ".php", ".rs", ".html", ".htm", ".css", ".scss", ".less", ".xml", ".yml"}
    files = collect_target_files(root, extensions)

    total_lines = 0

    if not args.summary_only:
        print(f"대상 폴더: {root}")
        print("집계 대상 확장자: .py, .yaml, .md, .json, .sh, .txt, .ts, .tsx, .jsx, .java, .go, .cpp, .c, .h, .rb, .php, .rs, .html, .htm, .css, .scss, .less, .xml, .yml")
        print("적용된 필터: .gitignore 규칙 반영")
        print("-" * 60)

    for file_path in files:
        line_count = count_lines_in_file(file_path)
        total_lines += line_count

        if not args.summary_only:
            rel_path = file_path.relative_to(root)
            print(f"{rel_path}: {line_count}")

    if not args.summary_only:
        print("-" * 60)
        print(f"대상 파일 수: {len(files)}")

    print(f"총 line count: {total_lines}")


if __name__ == "__main__":
    main()