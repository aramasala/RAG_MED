#!/usr/bin/env bash
# Generate QA from a PDF using rag-med. Run from project root: ./scripts/generate_qa.sh <file.pdf> [options]
set -e
cd "$(dirname "$0")/.."
if [ -n "$VIRTUAL_ENV" ]; then
  rag-med generate "$@"
else
  . .venv/bin/activate && rag-med generate "$@"
fi
