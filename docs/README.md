# RAG_MED — Documentation

Documentation for the medical RAG system (PDF cleaning and QA generation).

## Contents

- **Installation** — See main [README](../README.md) and `make setup`.
- **Usage** — CLI: `rag-med clean`, `rag-med generate`; Python API: `clean_pdf()`, `generate_qa_from_pdf()`.
- **Configuration** — Environment variables and `.env`; see [Configuration](../README.md#configuration) in the main README.
- **Project structure** — `rag_med/` (PDF cleaner, QA generator, evaluation, ValueAI client), `configs/` (settings and paths), `tests/` (unit and integration).

## Quick reference

| Task              | Command / API |
|-------------------|----------------|
| Clean PDF         | `rag-med clean document.pdf` |
| Generate QA       | `rag-med generate document.pdf` |
| Generate QA (n)    | `rag-med generate document.pdf -n 10` |
| ValueAI evaluation| `rag-med generate document.pdf --valueai-eval` |
| Python            | `from rag_med import clean_pdf, generate_qa_from_pdf` |
