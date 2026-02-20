# Notebooks

Jupyter notebooks for experimenting with RAG_MED (PDF cleaning, QA generation, evaluation).

## Setup

From project root:

```bash
source .venv/bin/activate
pip install jupyter  # or: poetry add --group dev jupyter
jupyter notebook notebooks/
```

## Suggested notebooks

- **01_pdf_clean_demo.ipynb** — Clean a sample PDF and inspect the result.
- **02_qa_generation.ipynb** — Generate QA pairs from a PDF and optionally run ValueAI evaluation.

Create these from the Jupyter UI or copy from a template; run cells from the project root so `configs` and `rag_med` are importable.
