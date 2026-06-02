"""Command-line entry point for GPU Impact Analyser."""

from __future__ import annotations

import argparse
from pathlib import Path

from analyzer import analyze, write_outputs
from extractor import extract
from visualizer import generate_timeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse an exported Nsight Systems SQLite database.")
    parser.add_argument("--sqlite", required=True, help="Path to SQLite file exported from Nsight Systems.")
    parser.add_argument("--outdir", default=".", help="Output directory. Defaults to the current directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    extraction = extract(args.sqlite)
    for warning in extraction.warnings:
        print(f"warning: {warning}")

    result = analyze(extraction)
    csv_path, report_path = write_outputs(result, outdir)
    timeline_path = generate_timeline(result.events, outdir)

    print("GPU Impact Analyser complete.")
    print(f"CSV: {csv_path}")
    print(f"Timeline: {timeline_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
