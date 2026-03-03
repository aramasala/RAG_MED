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
- **RAGAS Evaluation**: Evaluate answers with RAGAS Faithfulness (when using `--valueai-eval`)
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

### Run metrics test (`run_metrics_test.py`)

Скрипт сравнивает эталонные ответы с ответами ValueAI RAG и считает метрики (RAGAS faithfulness, cosine_similarity, factual_correctness, LLM alignment 1–10). Нужны учётные данные ValueAI в `.env`.

**Запуск:**

```bash
# Из корня проекта (по умолчанию читается test_qa_sample.json)
poetry run python run_metrics_test.py

# Свой JSON-файл с выборкой
RUN_METRICS_TEST_JSON=/path/to/my_samples.json poetry run python run_metrics_test.py
```

**Результаты:** в корне проекта создаются `test_metrics_result.json` (полный вывод по каждому вопросу) и `test_metrics_result.txt` (читаемый отчёт).

**Структура входного JSON** (как в `test_qa_sample.json`): массив объектов, в каждом объекте обязательны поля для сравнения эталона и ответа RAG. Остальные поля опциональны.

| Поле | Тип | Обязательное | Описание |
|------|-----|--------------|----------|
| `question` | string | да | Вопрос (клинический или от пациента). |
| `answer` | string | да | Эталонный ответ (правильный, для сравнения). |
| `valueai_answer` | string | да | Ответ RAG (ValueAI), который оценивается. |
| `chunk` | string | нет | Фрагмент текста (контекст); используется в LLM-судье. |
| `chunk_index` | number | нет | Номер фрагмента (для справки). |

Пример одного элемента массива:

```json
[
  {
    "chunk_index": 1,
    "chunk": "Фрагмент клинического текста...",
    "question": "Какие комбинации АГП считаются запрещенными?",
    "answer": "Эталонный ответ эксперта...",
    "valueai_answer": "Ответ, полученный от ValueAI RAG..."
  }
]
```

Файл-образец: [test_qa_sample.json](test_qa_sample.json).

##  Project Structure

```
RAG_MED/
├── configs/              # Configuration and paths (settings, PROJECT_DPATH, ValueAI LLM client)
│   ├── __init__.py
│   ├── settings.py
│   ├── paths.py
│   └── llm_api_client.py
├── docs/                 # Documentation
├── notebooks/            # Jupyter notebooks (demos, experiments)
├── reports/              # Generated reports and QA outputs
├── scripts/              # Helper scripts (clean_pdf.sh, generate_qa.sh)
├── rag_med/              # Main package
│   ├── __init__.py
│   ├── cli.py            # CLI interface
│   ├── pdf_cleaner/      # PDF cleaning module
│   │   ├── __init__.py
│   │   └── cleaner.py
│   ├── qa_generator/     # QA generation module
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   └── models.py
│   ├── evaluation/       # Answer evaluation metrics
│   │   ├── __init__.py
│   │   └── metrics.py
│   └── valueai/          # ValueAI RAG client
│       ├── __init__.py
│       └── client.py
├── tests/                # Tests
│   ├── unit/
│   ├── integration/
│   └── conftest.py
├── pyproject.toml        # Poetry configuration
├── Makefile              # Build commands
└── README.md
```

##  Configuration

Configuration can be set via environment variables or `.env` file:

```bash
# Sampling (QA generation)
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

# ValueAI: RAG and LLM for QA generation and evaluation (required for generate / --valueai-eval)
VALUEAI_BASE_URL=https://ml-request-develop2.wavea.cc/api/external/v1
VALUEAI_USERNAME=your.email@example.com
VALUEAI_PASSWORD=your_password
VALUEAI_RAG_ID=387
VALUEAI_MODEL_NAME=llm_qwen_3_32b_q8
VALUEAI_INSTRUCTIONS="you are helpful assistant"
VALUEAI_POLL_INTERVAL_SECONDS=2
VALUEAI_TIMEOUT_SECONDS=900

# Metrics / QA LLM (ValueAI v1/llm; used for RAGAS and QA generation)
METRICS_LLM_MODEL_NAME=llm_qwen_2_5_coder_32b_instruct_q8
METRICS_LLM_POLL_INTERVAL_SECONDS=2
METRICS_LLM_TIMEOUT_SECONDS=120
```

QA generation and evaluation use the **ValueAI** LLM (credentials and model from .env / `configs/settings`). With `--valueai-eval`, RAGAS (Faithfulness, FactualCorrectness) and an LLM alignment judge (1–10) are computed against ValueAI RAG answers.

### Метрики оценки (что показывает каждая метрика)

- **faithfulness** (RAGAS) — насколько ответ RAG опирается на извлечённые контексты; нет ли неподтверждённых утверждений (галлюцинаций). Оценка 0–1.
- **factual_correctness** (RAGAS) — фактическая правильность ответа по сравнению с эталонным ответом (reference). Оценка 0–1.
- **cosine_similarity** — лексическое (текстовое) сходство между эталонным ответом и ответом RAG (мешок слов, косинусная близость). Оценка 0–1.
- **alignment_score** (LLM-судья) — оценка по шкале 1–10: насколько ответ RAG по смыслу и качеству соответствует эталону (медицинская корректность, полнота, ясность).

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
