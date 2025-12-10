#!/usr/bin/env python3
"""
CLI to download datasets using Downloader classes in `src/data_pipeline/downloader.py`.

Usage examples:
  python scripts/download_data.py --dataset earnings_transcripts --output data/earnings_transcripts.csv --years 2021,2022 --tickers AAPL,MSFT
  python scripts/download_data.py --dataset earnings_qa --output data/earnings_qa.csv --quarters 1,2
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional


# Ensure `src` is on sys.path so we can import the package
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

try:
    from data_pipeline.downloader import (
        EarningsTranscriptsDownloader,
        EarningsQADownloader,
    )
except Exception as e:  # pragma: no cover - defensive import error
    raise ImportError(f"Unable to import downloader classes: {e}")


def _parse_int_list(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [int(p) for p in parts]


def _parse_str_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts or None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Download earnings datasets and save CSVs")
    p.add_argument(
        "--dataset",
        required=True,
        choices=["earnings_transcripts", "earnings_qa"],
        help="Which dataset to download: 'earnings_transcripts' or 'earnings_qa'",
    )
    p.add_argument(
        "--output",
        required=False,
        default=None,
        help="Output CSV file path (default: data/<dataset>.csv)",
    )
    p.add_argument(
        "--quarters",
        required=False,
        help="Comma-separated list of fiscal quarters (1,2,3,4)",
    )
    p.add_argument(
        "--years",
        required=False,
        help="Comma-separated list of years (e.g. 2020,2021)",
    )
    p.add_argument(
        "--tickers",
        required=False,
        help="Comma-separated list of tickers to filter (e.g. AAPL,MSFT)",
    )
    return p


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    dataset = args.dataset
    output = args.output
    if output is None:
        output = f"data/{dataset}.csv"

    output_path = Path(output)

    quarters = _parse_int_list(args.quarters)
    years = _parse_int_list(args.years)
    tickers = _parse_str_list(args.tickers)

    if dataset == "earnings_transcripts":
        downloader = EarningsTranscriptsDownloader()
    else:
        downloader = EarningsQADownloader()

    try:
        downloader.download_and_save_dataset(
            output_path, quarters=quarters, years=years, tickers=tickers
        )
    except Exception as exc:
        print(f"Error while downloading/saving dataset: {exc}")
        raise


if __name__ == "__main__":
    main()
