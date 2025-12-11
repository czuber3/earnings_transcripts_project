#!/usr/bin/env python3
"""
Interactive CLI to search the vector DB using a natural-language query.

Requirements:
 - Project `src` must be importable (script adds `src` to sys.path automatically)
 - Dependencies: chromadb, sentence-transformers (see project's requirements.txt)

Usage:
  python scripts/search_vectordb.py --vectordb data/chroma_db --collection earnings_transcripts

Run the script and type queries at the prompt. Empty input exits.
"""
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
import pprint


# Ensure `src` is importable
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

try:
    from text_embedding.text_embedder import TextEmbedder
    from vector_db.vector_db import VectorDB
except Exception as e:  # pragma: no cover - provide friendly error
    raise ImportError(f"Unable to import project modules. Make sure you're running from the repo root and have installed requirements: {e}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Interactive search over vector DB")
    p.add_argument("--vectordb", required=True, help="Path to the vector DB directory")
    p.add_argument("--collection", required=True, help="Collection name to query")
    p.add_argument("--embedding-model", default="FinLang/finance-embeddings-investopedia", help="Embedding model to use for queries")
    p.add_argument(
        "--device",
        default=None,
        help="Device to load embeddings model on (e.g. 'cuda' or 'cpu'). If omitted, auto-detects GPU if available.",
    )
    p.add_argument("--n-results", type=int, default=5, help="Number of results to return")
    p.add_argument("--ticker", required=False, help="Optional ticker filter (e.g. AAPL)")
    p.add_argument("--year", type=int, required=False, help="Optional year filter (e.g. 2023)")
    p.add_argument("--quarter", type=int, required=False, help="Optional quarter filter (1-4)")
    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    vectordb_path = Path(args.vectordb)
    collection_name = args.collection
    ticker = args.ticker
    year = args.year
    quarter = args.quarter
    n_results = args.n_results

    # Initialize embedder and vector DB
    print(f"Initializing embedder and vector DB client (device={args.device})...")
    embedder = TextEmbedder(model_name=args.embedding_model, device=args.device)
    vdb = VectorDB(vector_db_path=str(vectordb_path))

    print("Ready. Enter queries (empty line to exit).")
    while True:
        try:
            query = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not query:
            print("Exiting.")
            break

        # Create embedding for the query
        try:
            q_emb = embedder.embed_texts([query])[0]
        except Exception as e:
            print(f"Error creating embedding: {e}")
            continue

        # query vector db
        results = vdb.search(
            query_embedding=q_emb, 
            collection_name=collection_name,
            ticker=ticker,
            year=year,
            quarter=quarter,
            n_results=n_results
        )

        pprint.pprint('Results:')
        pprint.pprint(results)


if __name__ == "__main__":
    main()
