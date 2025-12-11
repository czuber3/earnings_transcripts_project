#!/usr/bin/env bash
# Runs end-to-end ingestion and evaluation for both recursive and semantic chunking
# Usage: ./scripts/run_pipeline.sh
# Requires: Python venv in .venv, MISTRAL_API_KEY set in environment
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Activate virtualenv if present
if [ -f ".venv/bin/activate" ]; then
  # POSIX venv
  # shellcheck source=/dev/null
  source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
  # Windows Git-Bash style
  # shellcheck source=/dev/null
  source .venv/Scripts/activate
else
  echo "Warning: no .venv activation script found. Make sure your Python environment is active." >&2
fi

# Paths and names
TRANSCRIPTS_CSV="data/earnings_transcripts.csv"
QA_CSV="data/earnings_qa.csv"
VDB_RECURSIVE="data/chroma_db_recursive"
VDB_SEMANTIC="data/chroma_db_semantic"
COLLECTION_RECURSIVE="earnings_recursive"
COLLECTION_SEMANTIC="earnings_semantic"
EMBEDDING_MODEL="FinLang/finance-embeddings-investopedia"
SAMPLE_SIZE=100
CONTEXT_RESULTS=10
QA_TICKERS="LNC,MTG,FSLR,AVB,DAL,JWN,CBOE,SRE,DISCK,MTD"
DEVICE="${DEVICE:-}"
if [ -n "$DEVICE" ]; then
  OPT_DEVICE="--device $DEVICE"
else
  OPT_DEVICE=""
fi

# # 1) Download datasets (if not already present)
# echo "==> Downloading datasets (transcripts + QA)"
# python scripts/download_data.py --dataset earnings_transcripts --output "$TRANSCRIPTS_CSV" --quarters 4 --years 2020 || true
# python scripts/download_data.py --dataset earnings_qa --output "$QA_CSV" --tickers "$QA_TICKERS" --quarters 4 --years 2020 || true

# # 2) Build vector DB with recursive chunking
# echo "==> Building vector DB (recursive chunking) -> $VDB_RECURSIVE / collection: $COLLECTION_RECURSIVE"
# python -m scripts.build_vectordb --input "$TRANSCRIPTS_CSV" --vectordb "$VDB_RECURSIVE" --chunker recursive --collection "$COLLECTION_RECURSIVE" --embedding-model "$EMBEDDING_MODEL"

# 3) Evaluate using recursive vector DB
# echo "==> Evaluating (recursive chunking)"
# python -m scripts.evaluate --vectordb "$VDB_RECURSIVE" --collection "$COLLECTION_RECURSIVE" --sample-size $SAMPLE_SIZE --context-results $CONTEXT_RESULTS

# 4) Build vector DB with semantic chunking
echo "==> Building vector DB (semantic chunking) -> $VDB_SEMANTIC / collection: $COLLECTION_SEMANTIC"
python -m scripts.build_vectordb $OPT_DEVICE --input "$TRANSCRIPTS_CSV" --vectordb "$VDB_SEMANTIC" --chunker semantic --collection "$COLLECTION_SEMANTIC" --embedding-model "$EMBEDDING_MODEL"

# 5) Evaluate using semantic vector DB
echo "==> Evaluating (semantic chunking)"
python -m scripts.evaluate $OPT_DEVICE --vectordb "$VDB_SEMANTIC" --collection "$COLLECTION_SEMANTIC" --sample-size $SAMPLE_SIZE --context-results $CONTEXT_RESULTS

echo "\nAll steps complete."
