#!/usr/bin/env python3
"""
Evaluate earnings Q&A alignment using LLMEvaluator and earnings_qa.csv.

This script:
- Loads earnings_qa.csv from the data directory
- Optionally filters by ticker, year, quarter, etc.
- Randomly samples a subset of questions (default 100)
- Uses LLMEvaluator.context_answer_alignment to check each Q/A pair
- Calculates and prints alignment rate (% of pairs that align)

Usage examples:
  python scripts/evaluate.py --api-key YOUR_MISTRAL_KEY
  python scripts/evaluate.py --api-key YOUR_MISTRAL_KEY --sample-size 50
  python scripts/evaluate.py --api-key YOUR_MISTRAL_KEY --ticker AAPL --year 2023
  python scripts/evaluate.py --api-key YOUR_MISTRAL_KEY --ticker AAPL,MSFT --quarter 1,2 --sample-size 25
"""
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional
import random
import time

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Make repo root importable so `src` package imports work
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from src.evaluation.evaluator import LLMEvaluator
    from src.text_embedding.text_embedder import TextEmbedder
    from src.vector_db.vector_db import VectorDB
except Exception as e:  # pragma: no cover
    raise ImportError(f"Unable to import required modules. Run from repo root: {e}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate earnings Q&A alignment rate using LLMEvaluator with vector DB context",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python scripts/evaluate.py --vectordb data/chroma_db --collection earnings
            python scripts/evaluate.py --vectordb data/chroma_db --collection earnings --sample-size 50 --ticker AAPL
            python scripts/evaluate.py --vectordb data/chroma_db --collection earnings --ticker AAPL,MSFT --year 2023
        """,
    )
    p.add_argument("--vectordb", required=True, help="Path to the vector DB directory")
    p.add_argument("--collection", required=True, help="Collection name in the vector DB")
    p.add_argument("--csv-file", default="data/earnings_qa.csv", help="CSV filename (default: earnings_qa.csv)")
    p.add_argument("--sample-size", type=int, default=100, help="Number of Q/A pairs to sample (default: 100)")
    p.add_argument("--ticker", help="Comma-separated tickers to filter (e.g. AAPL,MSFT)")
    p.add_argument("--year", help="Comma-separated years to filter (e.g. 2022,2023)")
    p.add_argument("--quarter", help="Comma-separated quarters to filter (e.g. 1,2)")
    p.add_argument("--mistral-model", default="mistral-large-latest", help="Mistral model to use")
    p.add_argument("--embedding-model", default="FinLang/finance-embeddings-investopedia", help="Embedding model for query context")
    p.add_argument("--context-results", type=int, default=5, help="Number of context chunks to retrieve from vector DB (default: 5)")
    p.add_argument(
        "--device",
        default=None,
        help="Device to load embeddings model on (e.g. 'cuda' or 'cpu'). If omitted, auto-detects GPU if available.",
    )
    p.add_argument("--seed", type=int, default=None, help="Random seed for reproducible sampling")
    p.add_argument("--verbose", action="store_true", help="Print details of each evaluation")
    return p


def parse_filter_list(value: Optional[str]) -> Optional[List]:
    """Parse comma-separated filter values"""
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    # Try to convert to int if all parts are numeric
    try:
        return [int(p) for p in parts]
    except ValueError:
        return parts or None


def chunks_to_context_string(chunks) -> str:
    """Transform list of EarningsTranscriptChunk objects into a formatted context string"""
    if not chunks:
        return ""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"Excerpt {i}:\n{chunk.text}")
    return "\n\n".join(context_parts)


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load CSV
    print(f"Loading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise RuntimeError(f"Error reading CSV: {e}")

    if df.empty:
        raise ValueError("CSV is empty")

    print(f"Loaded {len(df)} Q/A pairs")

    # Parse filters
    tickers = parse_filter_list(args.ticker)
    years = parse_filter_list(args.year)
    quarters = parse_filter_list(args.quarter)

    # Apply filters
    filtered_df = df.copy()

    if tickers is not None:
        if "ticker" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["ticker"].isin(tickers)]
        elif "symbol" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["symbol"].isin(tickers)]
        else:
            print("Warning: No ticker/symbol column found for filtering")

    if years is not None:
        if "year" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["year"].isin(years)]
        else:
            print("Warning: No year column found for filtering")

    if quarters is not None:
        if "quarter" in filtered_df.columns:
            filtered_df = filtered_df[filtered_df["quarter"].isin(quarters)]
        else:
            print("Warning: No quarter column found for filtering")

    print(f"After filtering: {len(filtered_df)} Q/A pairs")

    if filtered_df.empty:
        raise ValueError("No Q/A pairs remain after filtering")

    # Sample
    sample_size = min(args.sample_size, len(filtered_df))
    if args.seed is not None:
        random.seed(args.seed)
    sampled_df = filtered_df.sample(n=sample_size, random_state=args.seed)
    print(f"Sampled {len(sampled_df)} Q/A pairs")

    # Initialize evaluator, embedder, and vector DB
    print(f"Initializing LLMEvaluator, TextEmbedder (device={args.device}), and VectorDB...")
    print(os.getenv("MISTRAL_API_KEY"))
    evaluator = LLMEvaluator(mistral_api_key=os.getenv("MISTRAL_API_KEY"), mistral_model=args.mistral_model)
    embedder = TextEmbedder(model_name=args.embedding_model, device=args.device)
    vdb = VectorDB(vector_db_path=args.vectordb)

    # Evaluate alignment
    print(f"Evaluating {len(sampled_df)} Q/A pairs...")
    aligned_count = 0
    errors = 0

    for idx, row in sampled_df.iterrows():
        question = row.get("question", "")
        answer = row.get("answer", "")

        if not question or not answer:
            print(f"  Row {idx}: Skipped (missing question or answer)")
            continue

        # Query vector DB to get context
        try:
            ticker = row.get("ticker", None)
            try:
                fiscal_quarter = row.get("q", None)
                year = int(fiscal_quarter[:4])
                quarter = int(fiscal_quarter[6])
            except:
                year = None
                quarter = None

            question_embedding = embedder.embed_texts([question])[0]
            context_chunks = vdb.search(
                query_embedding=question_embedding,
                collection_name=args.collection,
                ticker=ticker,
                year=year,
                quarter=quarter,
                n_results=args.context_results
            )
            context = chunks_to_context_string(context_chunks)
            
            if not context:
                print(f"  Row {idx}: No context found in vector DB for query")
                continue

            try:
                is_aligned = evaluator.context_answer_alignment(
                    question=question,
                    answer=answer,
                    context=context
                )
            except:
                print("f  Mistral rate limit exceeded. Sleeping for one minute...")
                time.sleep(60)

            if is_aligned:
                aligned_count += 1
            if args.verbose:
                status = "✓ ALIGNED" if is_aligned else "✗ NOT ALIGNED"
                print(f"  [{idx}] {status} | Q: {question[:50]}...")
        except Exception as e:
            errors += 1
            print(f"  Row {idx}: Error during evaluation: {e}")
            continue

    # Calculate and print alignment rate
    evaluated = len(sampled_df) - errors
    if evaluated == 0:
        print("No Q/A pairs were successfully evaluated")
        sys.exit(1)

    alignment_rate = (aligned_count / evaluated) * 100
    print(f"\n" + "="*80)
    print(f"ALIGNMENT RATE: {alignment_rate:.2f}%")
    print(f"  Aligned: {aligned_count}/{evaluated}")
    print(f"  Errors: {errors}")
    print(f"="*80)


if __name__ == "__main__":
    main()
