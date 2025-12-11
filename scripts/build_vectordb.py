#!/usr/bin/env python3
"""
CLI to build a vector database from earnings transcripts.

This script:
1. Loads earnings transcripts from a CSV file
2. Uses chunking to split transcripts into chunks
3. Generates embeddings for each chunk using a pre-trained model
4. Creates a vector database and uploads chunks with their embeddings

Usage examples:
  python scripts/build_vectordb.py --input data/earnings_transcripts.csv --vectordb data/chroma_db --chunker recursive
  python scripts/build_vectordb.py --input data/earnings_transcripts.csv --vectordb data/chroma_db --chunker semantic --collection earnings_transcripts
  python scripts/build_vectordb.py --help
"""
import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Ensure `src` is on sys.path so we can import the package
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

try:
    from chunking.chunker import RecursiveChunker, SemanticChunker
    from chunking.earnings_transcript_chunk import EarningsTranscriptChunk
    from text_embedding.text_embedder import TextEmbedder
    from vector_db.vector_db import VectorDB
except Exception as e:  # pragma: no cover
    raise ImportError(f"Unable to import required modules: {e}")


def build_vectordb(
    input_csv: Path,
    vectordb_path: Path,
    chunker_type: str = "recursive",
    collection_name: str = "earnings_transcripts",
    max_chunk_size: int = 1000,
    chunk_overlap: int = 200,
    embedding_model: str = "FinLang/finance-embeddings-investopedia",
    device: Optional[str] = None,
    similarity_threshold: float = 0.4,
    batch_size: int = 50,
):
    """Build a vector database from earnings transcripts CSV.
    
    Args:
        input_csv (Path): Path to CSV file with columns: content, symbol, year, quarter
        vectordb_path (Path): Path to store the vector database
        chunker_type (str): 'recursive' or 'semantic'
        collection_name (str): Name of the collection in the vector database
        max_chunk_size (int): Maximum chunk size in characters
        chunk_overlap (int): Overlap between chunks in characters
        embedding_model (str): Pre-trained embedding model name
        similarity_threshold (float): Similarity threshold for semantic chunking
        batch_size (int): Number of chunks to embed in each batch
    """
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")
    
    print(f"Loading earnings transcripts from {input_csv}...")
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} records")
    
    # Initialize chunker
    print(f"\nInitializing {chunker_type} chunker...")
    if chunker_type == "recursive":
        chunker = RecursiveChunker()
    elif chunker_type == "semantic":
        text_embedder = TextEmbedder(model_name=embedding_model, device=device)
        chunker = SemanticChunker(text_embedder=text_embedder)
    else:
        raise ValueError(f"Unknown chunker type: {chunker_type}")
    
    # Initialize text embedder for chunk embeddings
    print(f"Initializing text embedder ({embedding_model}) on device={device}...")
    text_embedder = TextEmbedder(model_name=embedding_model, device=device)
    
    # Initialize vector database
    print(f"Initializing vector database at {vectordb_path}...")
    vector_db = VectorDB(vector_db_path=str(vectordb_path))
    
    # Delete collection if it exists.
    print(f"Deleteing collection '{collection_name}'...")
    vector_db.delete_collection(collection_name)
    
    # Process transcripts and build chunks
    print("\nProcessing transcripts and creating chunks...")
    all_chunks: List[EarningsTranscriptChunk] = []
    
    for idx, row in df.iterrows():
        if idx % 10 == 0:
            print(f"  Processing record {idx + 1}/{len(df)}...")
        
        content = row.get("content", "")
        ticker = row.get("symbol", "UNKNOWN")
        year = int(row.get("year", 0))
        quarter = int(row.get("quarter", 0))
        
        if not content or not isinstance(content, str):
            print(f"    Warning: Skipping record {idx} - empty or invalid content")
            continue
        
        try:
            chunks = chunker.chunk(
                text=content,
                ticker=ticker,
                year=year,
                quarter=quarter,
                max_chunk_size=max_chunk_size,
                overlap=chunk_overlap,
                similarity_threshold=similarity_threshold if chunker_type == "semantic" else None,
            )
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"    Warning: Error chunking record {idx} ({ticker} {year}Q{quarter}): {e}")
            continue
    
    if not all_chunks:
        raise ValueError("No chunks were created from the input data")
    
    print(f"Created {len(all_chunks)} chunks from {len(df)} records")
    
    # Generate embeddings in batches
    print(f"\nGenerating embeddings for {len(all_chunks)} chunks...")
    chunk_texts = [chunk.text for chunk in all_chunks]
    all_embeddings = []
    
    for i in range(0, len(chunk_texts), batch_size):
        batch_end = min(i + batch_size, len(chunk_texts))
        batch = chunk_texts[i:batch_end]
        print(f"  Embedding batch {i // batch_size + 1}/{(len(chunk_texts) + batch_size - 1) // batch_size}...")
        embeddings = text_embedder.embed_texts(batch)
        all_embeddings.extend(embeddings)
    
    if len(all_embeddings) != len(all_chunks):
        raise ValueError(
            f"Mismatch between chunks ({len(all_chunks)}) and embeddings ({len(all_embeddings)})"
        )
    
    # Upload to vector database
    print(f"\nUploading {len(all_chunks)} chunks and embeddings to vector database...")
    try:
        for i in range(0, len(all_chunks), batch_size):
            batch_end = min(i + batch_size, len(all_chunks))
            chunk__batch = all_chunks[i:batch_end]
            embedding_batch = all_embeddings[i:batch_end]
            vector_db.upload(
                collection_name=collection_name,
                earnings_transcript_chunks=chunk__batch,
                embeddings=embedding_batch,
            )
        print(f"Successfully uploaded all chunks to collection '{collection_name}'")
    except Exception as e:
        print(f"Error uploading to vector database: {e}")
        raise
    
    print(f"\n✓ Vector database built successfully!")
    print(f"  - Vector DB location: {vectordb_path}")
    print(f"  - Collection name: {collection_name}")
    print(f"  - Total chunks: {len(all_chunks)}")
    print(f"  - Embedding model: {embedding_model}")

def int_with_max(max_value):
    def _check(value):
        v = int(value)
        if v > max_value:
            raise argparse.ArgumentTypeError(f"Value must be ≤ {max_value}")
        return v
    return _check

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a vector database from earnings transcripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            python scripts/build_vectordb.py --input data/earnings_transcripts.csv --vectordb data/chroma_db
            python scripts/build_vectordb.py --input data/earnings_transcripts.csv --vectordb data/chroma_db --chunker semantic
            python scripts/build_vectordb.py --input data/earnings_transcripts.csv --vectordb data/chroma_db --collection earnings_2023
        """,
    )
    p.add_argument(
        "--input",
        default="data/earnings_transcripts.csv",
        help="Path to input CSV file (columns: content, symbol, year, quarter)",
    )
    p.add_argument(
        "--vectordb",
        default="../data/chroma_db",
        help="Path to vector database directory (default: data/chroma_db)",
    )
    p.add_argument(
        "--collection",
        default="earnings_transcripts",
        help="Name of the collection in the vector database (default: earnings_transcripts)",
    )
    p.add_argument(
        "--chunker",
        choices=["recursive", "semantic"],
        default="recursive",
        help="Chunking strategy to use (default: recursive)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum chunk size in characters (default: 1000)",
    )
    p.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200)",
    )
    p.add_argument(
        "--embedding-model",
        default="FinLang/finance-embeddings-investopedia",
        help="Pre-trained embedding model to use (default: FinLang/finance-embeddings-investopedia)",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device to load embeddings model on (e.g. 'cuda' or 'cpu'). If omitted, auto-detects GPU if available.",
    )
    p.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.4,
        help="Similarity threshold for semantic chunking (default: 0.4)",
    )
    p.add_argument(
        "--batch-size",
        type=int_with_max(5461), # max batch size for vectordb upload
        default=50,
        help="Number of chunks to embed amd index in each batch (default: 50, must be less than 5461)",
    )
    return p


def main(argv: Optional[List[str]] = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    
    try:
        build_vectordb(
            input_csv=Path(args.input),
            vectordb_path=Path(args.vectordb),
            chunker_type=args.chunker,
            collection_name=args.collection,
            max_chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embedding_model,
            device=args.device,
            similarity_threshold=args.similarity_threshold,
            batch_size=args.batch_size,
        )
    except Exception as exc:
        print(f"\n✗ Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
