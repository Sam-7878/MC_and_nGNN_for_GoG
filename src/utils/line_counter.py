from pathlib import Path
import argparse
import sys


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


def collect_target_files(root: Path, extensions: set[str]) -> list[Path]:
    """
    root 이하의 모든 하위 폴더를 포함하여 대상 확장자 파일을 수집합니다.
    """
    files = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            files.append(path)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="지정한 폴더 이하의 .py, .yaml 파일 총 라인 수를 계산합니다."
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

    extensions = {".py", ".yaml"}
    files = collect_target_files(root, extensions)

    total_lines = 0

    if not args.summary_only:
        print(f"대상 폴더: {root}")
        print("집계 대상 확장자: .py, .yaml")
        print("-" * 60)

    for file_path in files:
        line_count = count_lines_in_file(file_path)
        total_lines += line_count

        if not args.summary_only:
            rel_path = file_path.relative_to(root)
            print(f"{rel_path}: {line_count}")

    if not args.summary_only:
        print("-" * 60)
        print(f"파일 수: {len(files)}")

    print(f"총 line count: {total_lines}")


if __name__ == "__main__":
    main()
