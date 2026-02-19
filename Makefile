.PHONY: help setup install test clean venv deps clean-pdf generate-qa info check-python

PROJECT_NAME = rag-med
VENV_DIR = .venv
PYTHON = $(VENV_DIR)/bin/python
VENV_ACTIVATE = . $(VENV_DIR)/bin/activate

GREEN = \033[0;32m
YELLOW = \033[1;33m
RED = \033[0;31m
BLUE = \033[0;34m
NC = \033[0m

.DEFAULT_GOAL := help

help:
	@echo "$(BLUE)RAG_MED - Makefile$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: check-python venv deps
	@echo "$(GREEN) Setup complete!$(NC)"
	@echo ""
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  make clean-pdf ARGS=\"file.pdf\""
	@echo "  make generate-qa ARGS=\"file.pdf\""
	@echo "  make test"

venv:
	@echo "$(BLUE)Creating virtual environment...$(NC)"
	@python3 -m venv $(VENV_DIR)
	@echo "$(GREEN) Virtual environment created$(NC)"

check-venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "$(RED) Virtual environment not found! Run 'make setup' first$(NC)"; \
		exit 1; \
	fi

deps: check-venv
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(VENV_ACTIVATE) && pip install --upgrade pip
	$(VENV_ACTIVATE) && pip install poetry
	$(VENV_ACTIVATE) && poetry install
	@echo "$(GREEN) Dependencies installed$(NC)"

install: deps
	@echo "$(GREEN) Installation complete$(NC)"

test: check-venv
	@echo "$(BLUE)Running tests...$(NC)"
	$(VENV_ACTIVATE) && python -m pytest -v

lint: check-venv
	@echo "$(BLUE)Running linter...$(NC)"
	$(VENV_ACTIVATE) && python -m ruff check .

format: check-venv
	@echo "$(BLUE)Formatting code...$(NC)"
	$(VENV_ACTIVATE) && python -m black .

check-all: lint test
	@echo "$(GREEN) All checks passed$(NC)"

clean-pdf: check-venv
	@if [ -z "$(ARGS)" ]; then \
		echo "$(RED) Usage: make clean-pdf ARGS=\"file.pdf\"$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Cleaning PDF(s)...$(NC)"
	cd $(VENV_DIR)/bin && ./rag-med clean $(ARGS)

generate-qa: check-venv
	@if [ -z "$(ARGS)" ]; then \
		echo "$(RED) Usage: make generate-qa ARGS=\"file.pdf\"$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Generating QA from PDF...$(NC)"
	cd $(VENV_DIR)/bin && ./rag-med generate "$(ARGS)"

clean:
	@echo "$(BLUE)Cleaning project...$(NC)"
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov *.egg-info dist build
	@echo "$(GREEN) Project cleaned$(NC)"

reset: clean setup
	@echo "$(GREEN) Project reset complete$(NC)"

info:
	@echo "$(BLUE)Project Information:$(NC)"
	@echo "  Name: $(PROJECT_NAME)"
	@echo "  Python: $$(python3 --version 2>/dev/null || echo 'Not found')"
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "  Virtual env: $(VENV_DIR)"; \
		echo "  Python (venv): $$($(PYTHON) --version 2>/dev/null)"; \
	else \
		echo "  Virtual env: Not created"; \
	fi

check-python:
	@echo "$(BLUE)Checking Python...$(NC)"
	@if ! command -v python3 >/dev/null 2>&1; then \
		echo "$(RED) Python3 not found!$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN) Python3 found: $$(python3 --version)$(NC)"
