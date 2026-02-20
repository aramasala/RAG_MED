#!/usr/bin/env bash
# Clean PDF(s) using rag-med. Run from project root: ./scripts/clean_pdf.sh <file.pdf|dir/>
set -e
cd "$(dirname "$0")/.."
if [ -n "$VIRTUAL_ENV" ]; then
  rag-med clean "$@"
else
  . .venv/bin/activate && rag-med clean "$@"
fi
