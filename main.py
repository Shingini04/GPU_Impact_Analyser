from __future__ import annotations

import argparse
import sys
from pathlib import Path

from analyzer import analyze_profile
from extractor import NsightExtractor
from html_report import render_html_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nsight Systems Analyzer: build one self-contained HTML report from an exported Nsight Systems SQLite database."
    )
    parser.add_argument("--sqlite", required=True, type=Path, help="Path to profile.sqlite exported with: nsys export -t sqlite profile.nsys-rep")
    parser.add_argument("--output", required=True, type=Path, help="Output HTML file, for example report.html")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sqlite_path = args.sqlite.expanduser()
    output_path = args.output.expanduser()

    if not sqlite_path.exists():
        print(f"Error: SQLite file does not exist: {sqlite_path}", file=sys.stderr)
        return 2
    if sqlite_path.is_dir():
        print(f"Error: SQLite path is a directory, not a file: {sqlite_path}", file=sys.stderr)
        return 2

    try:
        extraction = NsightExtractor(sqlite_path).extract()
        analysis = analyze_profile(extraction)
        html = render_html_report(analysis)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Report generated: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
