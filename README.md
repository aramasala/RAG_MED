# RAG_MED

<div align="center">

[![PythonSupported](https://img.shields.io/badge/python-3.10+-brightgreen.svg)](https://python3statement.org/#sections50-why)
[![poetry](https://img.shields.io/endpoint?url=https://python-poetry.org/badge/v0.json)](https://python-poetry.org/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![mypy](https://img.shields.io/badge/type%20checked-mypy-039dfc)](https://mypy-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Medical RAG system for PDF processing and QA generation

</div>

##  Features

- **PDF Cleaning**: Remove unnecessary sections (bibliography, appendices) from medical PDF files
- **QA Generation**: Generate clinical questions and answers from PDF text using LLM
- **Professional Structure**: Modern Python package structure with Poetry, type hints, and testing
- **CLI Interface**: Easy-to-use command-line interface with rich output
- **Configurable**: Settings via environment variables or config files

##  Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/aramasala/RAG_MED.git
cd RAG_MED

# Install dependencies
make setup

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install poetry
poetry install
```

##  Usage

### Command Line Interface

#### Clean PDF files

```bash
# Clean a single PDF file
rag-med clean document.pdf

# Clean all PDFs in a directory
rag-med clean /path/to/pdfs/

# Specify output directory
rag-med clean /path/to/pdfs/ --output /path/to/output/
```

#### Generate QA from PDF

```bash
# Generate QA pairs from PDF
rag-med generate document.pdf

# Choose count explicitly (otherwise it will prompt after chunking)
rag-med generate document.pdf --num-questions 10

# Evaluate against ValueAI RAG (requires VALUEAI_USERNAME / VALUEAI_PASSWORD)
rag-med generate document.pdf --valueai-eval

# Specify output file
rag-med generate document.pdf --output results.json

# Verbose output
rag-med generate document.pdf --verbose
```

### Using Makefile

```bash
# Clean PDF
make clean-pdf ARGS="document.pdf"

# Generate QA
make generate-qa ARGS="document.pdf"

# Run tests
make test

# Format code
make format

# Run linter
make lint
```

### Python API

```python
from rag_med import clean_pdf, generate_qa_from_pdf
from pathlib import Path

# Clean PDF
input_pdf = Path("document.pdf")
output_pdf = Path("document_cleaned.pdf")
clean_pdf(input_pdf, output_pdf)

# Generate QA
results = generate_qa_from_pdf(input_pdf, output_file=Path("qa_result.json"))
for result in results:
    print(f"Question: {result.question}")
    print(f"Answer: {result.answer}")
```

##  Project Structure

```
RAG_MED/
├── rag_med/              # Main package
│   ├── __init__.py
│   ├── cli.py            # CLI interface
│   ├── config.py         # Configuration settings
│   ├── paths.py          # Path utilities
│   ├── pdf_cleaner/      # PDF cleaning module
│   │   ├── __init__.py
│   │   └── cleaner.py
│   └── qa_generator/     # QA generation module
│       ├── __init__.py
│       ├── generator.py
│       └── models.py
├── tests/                # Tests
│   ├── unit/
│   └── conftest.py
├── pyproject.toml        # Poetry configuration
├── Makefile              # Build commands
└── README.md
```

##  Configuration

Configuration can be set via environment variables or `.env` file:

```bash
# Ollama settings
OLLAMA_BASE_URL=http://localhost:11434/v1
MODEL_NAME=qwen3-vl:8b-instruct
TEMPERATURE=0.7
MAX_TOKENS=500

# Text splitting settings
CHUNK_SIZE=2000
CHUNK_OVERLAP=200
MIN_CHUNK_WORDS=20
NUM_CHUNKS_TO_SELECT=3

# PDF cleaning settings
START_SECTION_TEXT=Список литературы
END_SECTION_TEXT=Приложение А2. Методология разработки клинических рекомендаций

# ValueAI RAG evaluation (optional; used with `rag-med generate --valueai-eval`)
VALUEAI_BASE_URL=https://ml-request-develop2.wavea.cc/api/external/v1
VALUEAI_USERNAME=your.email@example.com
VALUEAI_PASSWORD=your_password
VALUEAI_RAG_ID=387
VALUEAI_MODEL_NAME=llm_qwen_3_32b_q8
VALUEAI_INSTRUCTIONS="you are helpful assistant"
VALUEAI_POLL_INTERVAL_SECONDS=2
VALUEAI_TIMEOUT_SECONDS=120
```

##  Development

```bash
# Setup development environment
make setup

# Run tests
make test

# Run linter
make lint

# Format code
make format

# Run all checks
make check-all
```



##  Author

Aram Aleksanyan <aram.aleksanyan@waveaccess.global>
