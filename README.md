# Earnings Transcripts Vector Database & Evaluation Pipeline

A Python project for ingesting, chunking, embedding, and evaluating S&P 500 earnings call transcripts using semantic and recursive chunking strategies with LLM-based alignment evaluation.

## Table of Contents

- [Overview](#overview)
- [Why Semantic Chunking for Earnings Transcripts](#why-semantic-chunking-for-earnings-transcripts)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Scripts & Usage](#scripts--usage)
  - [Download Data](#1-download_dataapy)
  - [Build Vector DB](#2-build_vectordbpy)
  - [Upload Transcripts](#3-uploadpy)
  - [Search Vector DB](#4-searchpy)
  - [Evaluate Alignment](#5-evaluatepy)
  - [End-to-End Pipeline](#6-run_pipelinesh)
- [Examples](#examples)

---

## Overview

This project implements a complete pipeline for:

1. **Data Ingestion**: Download S&P 500 earnings transcripts and Q&A datasets
2. **Text Chunking**: Split transcripts using recursive or semantic strategies
3. **Embedding**: Generate vector embeddings for each chunk using an embedding model fine-tuned on financial data
4. **Storage**: Store chunks and embeddings in a Chroma vector database
5. **Search**: Query the vector DB with natural-language questions
6. **Evaluation**: Assess Q&A alignment by checking if correct answers are found in retrieved context using an LLM Evaluator

The pipeline supports **both semantic and recursive chunking**, enabling comparison of chunking strategies for unstructured earnings call transcripts.

---

## Why Semantic Chunking for Earnings Transcripts

Earnings call transcripts are highly **unstructured**, mixing:
- Multiple speakers (CEO, CFO, analysts) with interruptions
- Multiple topics within a single call (financial results, guidance, Q&A)
- Lack of natural paragraph breaks or transitions
- Dense financial and operational content

### Recursive Chunking Limitations

**Recursive chunking** (e.g., split by `\n\n`, then `\n`, then space) treats chunks as uniform blocks based on character count, ignoring semantic meaning. This leads to:

- **Topic splitting**: A semantically coherent discussion may be split across chunks because of formatting
- **Orphaned context**: Important financial statements may be separated from their explanations
- **Poor retrieval**: Queries may return chunks that lack the full context of an answer

### Semantic Chunking Advantages

**Semantic chunking** groups text based on **meaning similarity**, not just delimiters. For earnings transcripts:

1. **Topic coherence**: Chunks preserve complete discussions about a single topic (e.g., "Q3 revenue growth" stays together)
2. **Reduced redundancy**: Fewer overlapping chunks needed because chunks are naturally bounded by topic shifts
3. **LLM-friendly**: Chunks are more coherent for LLM evaluation—the model sees "complete thoughts" not fragmented text

**Trade-off**: Semantic chunking is slower (requires embeddings of sentences) but produces higher-quality chunks for evaluation metrics like alignment rate.

---

## Datasets

### Earnings Transcripts Dataset
- **Source**: HuggingFace (kurry/sp500_earnings_transcripts)
- **Format**: Parquet
- **Coverage**: S&P 500 companies, 2005-2025
- **Content**: Full earnings call transcripts (speaker names, participant sections, Q&A)
- **Columns**: `symbol` (ticker), `year`, `quarter`, `content` (full transcript text)

### Earnings Q&A Dataset
- **Source**: HuggingFace (lamini/earnings-calls-qa)
- **Format**: JSON Lines
- **Coverage**: 2019-2023
- **Content**: Curated question-answer pairs from earnings calls
- **Columns**: `ticker`, `q` (fiscal quarter as "YYYY-Qn"), `question`, `answer`, `context` (optional)

Both datasets are automatically downloaded via `scripts/download_data.py`.

---

## Project Structure

```
earnings_transcripts_project/
├── src/
│   ├── chunking/
│   │   ├── chunker.py                 # RecursiveChunker, SemanticChunker classes
│   │   └── earnings_transcript_chunk.py # EarningsTranscriptChunk dataclass
│   ├── text_embedding/
│   │   └── text_embedder.py           # TextEmbedder (SentenceTransformer wrapper)
│   ├── vector_db/
│   │   └── vector_db.py               # VectorDB (Chroma wrapper)
│   └── evaluation/
│       └── evaluator.py               # LLMEvaluator (Mistral-based)
├── scripts/
│   ├── download_data.py               # Download datasets from HuggingFace
│   ├── build_vectordb.py              # Build & populate vector DB
│   ├── upload.py                      # Upload single TXT file to vector DB
│   ├── search.py                      # Interactive semantic search CLI
│   ├── evaluate.py                    # Evaluate Q&A alignment rate
│   └── run_pipeline.sh                # End-to-end pipeline (recursive + semantic)
├── data/
│   ├── earnings_transcripts.csv       # Downloaded transcripts
│   ├── earnings_qa.csv                # Downloaded Q&A pairs
│   ├── chroma_db_recursive/           # Vector DB (recursive chunking)
│   └── chroma_db_semantic/            # Vector DB (semantic chunking)
├── requirements.txt                   # Python dependencies
├── .env                               # Environment variables (API keys, etc.)
└── README.md                          # This file
```

### Key Classes

#### `EarningsTranscriptChunk` (src/chunking/earnings_transcript_chunk.py)
```python
@dataclass
class EarningsTranscriptChunk:
    text: str                 # The chunk text
    ticker: str              # Stock ticker (e.g., "AAPL")
    year: int                # Year of earnings call
    quarter: int             # Quarter (1-4)
```

#### `RecursiveChunker` (src/chunking/chunker.py)
Splits text using `RecursiveCharacterTextSplitter` with configurable size and overlap.

#### `SemanticChunker` (src/chunking/chunker.py)
Splits text by sentence, embeds sentences, groups by cosine similarity threshold.

#### `TextEmbedder` (src/text_embedding/text_embedder.py)
Wraps SentenceTransformer for generating embeddings (default: FinLang finance embeddings).

#### `VectorDB` (src/vector_db/vector_db.py)
Wraps Chroma (persistent SQLite + HNSW index) for storing/querying chunks and embeddings.

#### `LLMEvaluator` (src/evaluation/evaluator.py)
Uses Mistral LLM to evaluate if Q&A pairs are answerable from retrieved context.

---

## Environment Setup

### Prerequisites
- Python 3.11+
- Git
- Bash (for `run_pipeline.sh`)

### 1. Clone the Repository
```bash
git clone https://github.com/czuber3/earnings_transcripts_project.git
cd earnings_transcripts_project
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (Windows Command Prompt)
.venv\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Mistral API Key
```bash
# Create or edit .env file
echo "MISTRAL_API_KEY=your_key_here" > .env

# Or export to environment
export MISTRAL_API_KEY="your_mistral_api_key"
```

### 5. Download spaCy Model (for semantic chunking)
```bash
python -m spacy download en_core_web_sm
```

---

## Scripts & Usage

### 1. `download_data.py`
Download earnings transcripts and Q&A datasets from HuggingFace.

**Usage**
```bash
# Download transcripts
python scripts/download_data.py --dataset earnings_transcripts --output data/earnings_transcripts.csv

# Download Q&A pairs
python scripts/download_data.py --dataset earnings_qa --output data/earnings_qa.csv

# Filter by years and tickers
python scripts/download_data.py --dataset earnings_transcripts --output data/aapl_msft_2023.csv --years 2023 --tickers AAPL,MSFT
```

**Arguments**
- `--dataset`: `earnings_transcripts` or `earnings_qa` (required)
- `--output`: Output CSV path (default: `data/<dataset>.csv`)
- `--years`: Comma-separated years (e.g., `2021,2022,2023`)
- `--tickers`: Comma-separated tickers (e.g., `AAPL,MSFT`)
- `--quarters`: Comma-separated quarters (e.g., `1,2,3,4`)

---

### 2. `build_vectordb.py`
Build a vector database by chunking, embedding, and uploading transcripts.

**Usage**
```bash
# Recursive chunking (faster, less context-aware)
python scripts/build_vectordb.py \
  --input data/earnings_transcripts.csv \
  --vectordb data/chroma_db_recursive \
  --chunker recursive \
  --collection earnings

# Semantic chunking (slower, more context-aware)
python scripts/build_vectordb.py \
  --input data/earnings_transcripts.csv \
  --vectordb data/chroma_db_semantic \
  --chunker semantic \
  --collection earnings \
  --similarity-threshold 0.4
```

**Arguments**
- `--input`: Path to CSV with transcripts (required)
- `--vectordb`: Vector DB directory (required)
- `--chunker`: `recursive` or `semantic` (default: `recursive`)
- `--collection`: Collection name (default: `earnings`)
- `--chunk-size`: Max chunk size in characters (default: `1000`)
- `--chunk-overlap`: Overlap in characters (default: `200`)
- `--embedding-model`: HuggingFace model ID (default: `FinLang/finance-embeddings-investopedia`)
- `--similarity-threshold`: For semantic chunker (default: `0.4`)
- `--batch-size`: Embeddings per batch (default: `50`)

---

### 3. `upload.py`
Upload a single TXT file to an existing vector DB.

**Usage**
```bash
python scripts/upload.py \
  --file docs/my_transcript.txt \
  --vectordb data/chroma_db_semantic \
  --collection earnings \
  --ticker AAPL \
  --year 2023 \
  --quarter 2 \
  --chunker semantic
```

**Arguments**
- `--file`: Path to input TXT file (required)
- `--vectordb`: Vector DB directory (required)
- `--collection`: Collection name (required)
- `--ticker`: Stock ticker (required)
- `--year`: Year (required)
- `--quarter`: Quarter 1-4 (required)
- `--chunker`: `recursive` or `semantic` (default: `recursive`)
- `--chunk-size`: Max chunk size (default: `1000`)
- `--chunk-overlap`: Overlap (default: `200`)
- Other options: `--embedding-model`, `--similarity-threshold`, `--batch-size`

---

### 4. `search.py`
Interactive CLI to search the vector DB with natural-language queries.

**Usage**
```bash
# Start interactive search
python scripts/search.py \
  --vectordb data/chroma_db_semantic \
  --collection earnings \
  --n-results 5 \
  --embedding-model FinLang/finance-embeddings-investopedia

# At the prompt:
# query> What was the revenue growth in Q3?
# [Results with document text and metadata]
```

**Arguments**
- `--vectordb`: Vector DB directory (required)
- `--collection`: Collection name (required)
- `--embedding-model`: HuggingFace model (default: `FinLang/finance-embeddings-investopedia`)
- `--n-results`: Number of results to return (default: `5`)
- `--ticker`: Optional ticker filter
- `--year`: Optional year filter
- `--quarter`: Optional quarter filter

---

### 5. `evaluate.py`
Evaluate Q&A alignment by checking if retrieved context contains correct answers.

**Usage**
```bash
# Basic evaluation (100 random Q&A pairs)
python scripts/evaluate.py \
  --api-key "$MISTRAL_API_KEY" \
  --vectordb data/chroma_db_semantic \
  --collection earnings

# Custom sample size and filters
python scripts/evaluate.py \
  --api-key "$MISTRAL_API_KEY" \
  --vectordb data/chroma_db_semantic \
  --collection earnings \
  --sample-size 50 \
  --ticker AAPL,MSFT \
  --year 2023 \
  --context-results 10 \
  --verbose
```

**Arguments**
- `--api-key`: Mistral API key (required)
- `--vectordb`: Vector DB directory (required)
- `--collection`: Collection name (required)
- `--sample-size`: Number of Q&A pairs to evaluate (default: `100`)
- `--ticker`: Comma-separated tickers to filter
- `--year`: Comma-separated years to filter
- `--quarter`: Comma-separated quarters to filter
- `--context-results`: Number of context chunks to retrieve (default: `5`)
- `--embedding-model`: Embedding model for queries (default: `FinLang/finance-embeddings-investopedia`)
- `--mistral-model`: Mistral model (default: `mistral-large-latest`)
- `--seed`: Random seed for reproducibility
- `--verbose`: Print per-query details

**Output**
```
ALIGNMENT RATE: 87.50%
  Aligned: 87/100
  Errors: 0
```

---

### 6. `run_pipeline.sh`
End-to-end bash script that runs the full pipeline for both chunking strategies.

**Usage**
```bash
export MISTRAL_API_KEY="your_mistral_key"
chmod +x scripts/run_pipeline.sh
./scripts/run_pipeline.sh
```

**What it does**
1. Downloads datasets (transcripts + Q&A)
2. Builds recursive chunking vector DB and evaluates
3. Builds semantic chunking vector DB and evaluates
4. Prints alignment rates for comparison

---

## Examples

### Example 1: Compare Chunking Strategies
```bash
# Build both vector DBs
python scripts/build_vectordb.py --input data/earnings_transcripts.csv --vectordb data/vdb_recursive --chunker recursive --collection earnings
python scripts/build_vectordb.py --input data/earnings_transcripts.csv --vectordb data/vdb_semantic --chunker semantic --collection earnings

# Evaluate each
python scripts/evaluate.py --api-key "$MISTRAL_API_KEY" --vectordb data/vdb_recursive --collection earnings --sample-size 100 | grep "ALIGNMENT RATE"
python scripts/evaluate.py --api-key "$MISTRAL_API_KEY" --vectordb data/vdb_semantic --collection earnings --sample-size 100 | grep "ALIGNMENT RATE"
```

### Example 2: Evaluate Specific Companies
```bash
# Evaluate only Apple and Microsoft transcripts
python scripts/evaluate.py \
  --api-key "$MISTRAL_API_KEY" \
  --vectordb data/chroma_db_semantic \
  --collection earnings \
  --ticker AAPL,MSFT \
  --sample-size 50 \
  --verbose
```

### Example 3: Upload and Search Custom Transcript
```bash
# Upload a new transcript
python scripts/upload.py \
  --file my_earnings_call.txt \
  --vectordb data/chroma_db_semantic \
  --collection earnings \
  --ticker CUSTOM \
  --year 2024 \
  --quarter 1 \
  --chunker semantic

# Search it
python scripts/search.py \
  --vectordb data/chroma_db_semantic \
  --collection earnings
# query> What were the key highlights?
```

### Example 4: Evaluate with Custom Embedding Model
```bash
python scripts/evaluate.py \
  --api-key "$MISTRAL_API_KEY" \
  --vectordb data/chroma_db_semantic \
  --collection earnings \
  --embedding-model "sentence-transformers/all-MiniLM-L6-v2" \
  --sample-size 50
```

---

## Dependencies

See `requirements.txt`:

- **Data & embeddings**: `pandas`, `sentence-transformers`, `datasets`
- **Vector DB**: `chromadb`
- **Text processing**: `langchain-text-splitters`, `spacy`
- **LLM evaluation**: `mistralai`
- **Utilities**: `numpy`, `tqdm`

---

## Troubleshooting

### spaCy Model Not Found
```bash
python -m spacy download en_core_web_sm
```

### Mistral API Key Error
Ensure `MISTRAL_API_KEY` is set:
```bash
export MISTRAL_API_KEY="sk-..."
echo $MISTRAL_API_KEY  # Verify
```

### Vector DB Path Issues
Use absolute paths or relative paths from the repo root:
```bash
# ✓ Good
python scripts/search.py --vectordb data/chroma_db_semantic --collection earnings

# ✓ Also good
python scripts/search.py --vectordb /absolute/path/to/chroma_db_semantic --collection earnings
```

### Out of Memory During Embedding
Reduce `--batch-size` when building the vector DB:
```bash
python scripts/build_vectordb.py \
  --input data/earnings_transcripts.csv \
  --vectordb data/chroma_db_semantic \
  --chunker semantic \
  --batch-size 10  # Smaller batches
```

---

## Next Steps & Future Work

- **Caching**: Cache embeddings to speed up re-runs
- **Evaluation improvements**: Extend LLMEvaluator to measure other metrics (relevance, completeness)
- **UI dashboard**: Add a web interface for search and evaluation visualization
- **Multi-model evaluation**: Compare Mistral with other LLMs (GPT-4, Claude, etc.)
- **GPU Inference**: Deploy embedding model on GPU for latency improvement.
- **Tune semantic chunking hyperparameters**: Fine-tune semantic similarity threshold to improve transcript structure.
- **Unit tests**
- **Logging**

---