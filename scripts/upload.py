#!/usr/bin/env python3
"""
Upload a single TXT file to the vector DB.

This script:
- Reads a local .txt file
- Chunks the text (recursive or semantic)
- Generates embeddings for each chunk
- Uploads chunks + embeddings to a Chroma vector DB collection

Usage examples:
  python scripts/upload.py --file docs/transcript.txt --vectordb data/chroma_db --collection earnings --ticker AAPL --year 2023 --quarter 2

Note: run from repo root or ensure repo root is on PYTHONPATH. The script
adds the repo root to sys.path automatically so imports like `src.*` work.
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional


# Make repo root importable so `src` package imports work
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from src.chunking.chunker import RecursiveChunker, SemanticChunker
    from src.chunking.earnings_transcript_chunk import EarningsTranscriptChunk
    from src.text_embedding.text_embedder import TextEmbedder
    from src.vector_db.vector_db import VectorDB
except Exception as e:  # pragma: no cover - friendly import error
    raise ImportError(f"Unable to import project modules. Run from repo root and install requirements: {e}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Chunk, embed and upload a TXT file to the vector DB")
    p.add_argument("--file", required=True, help="Path to the input .txt file to upload")
    p.add_argument("--vectordb", required=True, help="Path to the vector DB directory")
    p.add_argument("--collection", required=True, help="Collection name in the vector DB to upload to")
    p.add_argument("--ticker", required=True, help="Ticker symbol for this transcript (e.g. AAPL)")
    p.add_argument("--year", type=int, required=True, help="Year for the transcript (e.g. 2023)")
    p.add_argument("--quarter", type=int, required=True, choices=[1,2,3,4], help="Quarter for the transcript (1-4)")
    p.add_argument("--chunker", choices=["recursive","semantic"], default="recursive", help="Chunking strategy (default: recursive)")
    p.add_argument("--chunk-size", type=int, default=1000, help="Maximum chunk size in characters")
    p.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap in characters")
    p.add_argument("--embedding-model", default="FinLang/finance-embeddings-investopedia", help="Embedding model name")
    p.add_argument("--similarity-threshold", type=float, default=0.4, help="Similarity threshold for semantic chunker")
    p.add_argument("--batch-size", type=int, default=50, help="Embedding batch size")
    return p


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    text = input_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Input file is empty")

    # Initialize components
    print("Initializing embedder and chunker...")
    embedder = TextEmbedder(model_name=args.embedding_model)
    if args.chunker == "recursive":
        chunker = RecursiveChunker()
    else:
        # SemanticChunker requires a TextEmbedder instance
        chunker = SemanticChunker(text_embedder=embedder)

    print("Creating chunks...")
    chunks: List[EarningsTranscriptChunk] = chunker.chunk(
        text=text,
        ticker=args.ticker,
        year=int(args.year),
        quarter=int(args.quarter),
        max_chunk_size=args.chunk_size,
        overlap=args.chunk_overlap,
        similarity_threshold=args.similarity_threshold if args.chunker == "semantic" else None,
    )

    if not chunks:
        raise RuntimeError("No chunks were produced from the input file")

    print(f"Produced {len(chunks)} chunks")

    # Generate embeddings in batches
    chunk_texts = [c.text for c in chunks]
    all_embeddings = []
    print("Generating embeddings...")
    for i in range(0, len(chunk_texts), args.batch_size):
        batch = chunk_texts[i:i+args.batch_size]
        print(f"  Embedding batch {i // args.batch_size + 1} ({len(batch)} items)")
        emb = embedder.embed_texts(batch)
        all_embeddings.extend(emb)

    if len(all_embeddings) != len(chunks):
        raise RuntimeError("Mismatch between number of chunks and embeddings")

    # Upload
    print(f"Uploading {len(chunks)} chunks to collection '{args.collection}' in {args.vectordb}...")
    vdb = VectorDB(vector_db_path=str(args.vectordb))
    vdb.upload(collection_name=args.collection, earnings_transcript_chunks=chunks, embeddings=all_embeddings)

    print("Upload complete.")


if __name__ == "__main__":
    main()
