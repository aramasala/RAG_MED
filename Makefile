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

# Auto-activation function
run-in-venv = $(VENV_ACTIVATE) && $(1)

# Main targets
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
	@$(call run-in-venv, pip install --upgrade pip)
	@$(call run-in-venv, pip install poetry)
	@$(call run-in-venv, poetry install)
	@echo "$(GREEN) Dependencies installed$(NC)"

install: deps
	@echo "$(GREEN) Installation complete$(NC)"

# Development commands
test: check-venv
	@echo "$(BLUE)Running tests...$(NC)"
	@$(call run-in-venv, python -m pytest -v)

lint: check-venv
	@echo "$(BLUE)Running linter...$(NC)"
	@$(call run-in-venv, python -m ruff check .)

format: check-venv
	@echo "$(BLUE)Formatting code...$(NC)"
	@$(call run-in-venv, python -m black .)

check-all: lint test
	@echo "$(GREEN) All checks passed$(NC)"

clean-pdf: check-venv
	@if [ -z "$(ARGS)" ]; then \
		echo "$(RED) Usage: make clean-pdf ARGS=\"file.pdf\"$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Cleaning PDF(s)...$(NC)"
	@$(call run-in-venv, rag-med clean $(ARGS))

generate-qa: check-venv
	@if [ -z "$(ARGS)" ]; then \
		echo "$(RED) Usage: make generate-qa ARGS=\"file.pdf\"$(NC)"; \
		exit 1; \
	fi
	@echo "$(BLUE)Generating QA from PDF...$(NC)"
	@$(call run-in-venv, rag-med generate $(ARGS))

shell: check-venv
	@echo "$(BLUE)Opening Python shell in virtual environment...$(NC)"
	@$(call run-in-venv, python)

activate:
	@echo "$(YELLOW)To activate virtual environment manually:$(NC)"
	@echo "  source $(VENV_DIR)/bin/activate"

clean:
	@echo "$(BLUE)Cleaning project...$(NC)"
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov *.egg-info dist build
	@echo "$(GREEN) Project cleaned$(NC)"

clean-all: clean
	@echo "$(BLUE)Cleaning all generated files...$(NC)"
	rm -f qa_result.json
	rm -f qa_result_valueai_eval.json
	rm -f *_cleaned.pdf
	@echo "$(GREEN) All cleaned$(NC)"

reset: clean setup
	@echo "$(GREEN) Project reset complete$(NC)"

info:
	@echo "$(BLUE)Project Information:$(NC)"
	@echo "  Name: $(PROJECT_NAME)"
	@echo "  Python: $$(python3 --version 2>/dev/null || echo 'Not found')"
	@if [ -d "$(VENV_DIR)" ]; then \
		echo "  Virtual env: $(VENV_DIR)"; \
		echo "  Python (venv): $$($(PYTHON) --version 2>/dev/null)"; \
		echo "  rag-med path: $$($(call run-in-venv, which rag-med) 2>/dev/null)"; \
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