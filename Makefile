.PHONY: install hooks up down ingest ask eval train serve test lint fmt typecheck

-include .env
export

PYTHON ?= python3
QUESTION ?= "What is the Spark API?"

install:            ## Install editable with dev tooling
	pip install -e ".[dev]"

hooks:              ## Install and run pre-commit hooks on all files
	pre-commit install
	pre-commit run --all-files

up:                 ## Start Qdrant + API
	docker compose up -d

down:               ## Stop Qdrant + API
	docker compose down

ingest:             ## Build the vector index from docs
	$(PYTHON) -m ingestion.pipeline --config configs/ingestion.yaml

ask:                ## Ask a question: make ask QUESTION="..."
	$(PYTHON) -m rag.cli ask --question "$(QUESTION)"

eval:               ## Run the eval harness, print scorecard
	$(PYTHON) -m eval.run_eval --config configs/eval.yaml

train:              ## QLoRA fine-tune (requires finetune extra + Unsloth)
	$(PYTHON) -m finetune.train --config configs/finetune.yaml

serve:              ## Run the API locally with reload
	uvicorn api.main:app --reload

test:               ## Run unit tests
	pytest

lint:               ## Ruff check
	ruff check .

fmt:                ## Ruff format
	ruff format .

typecheck:          ## Mypy strict over src/
	mypy

help:               ## List targets
	@grep -hE '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'
