"""Smoke test: every skeleton module imports cleanly."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "clients",
    "qdrant_store",
    "ingestion.chunk",
    "ingestion.clean",
    "ingestion.embed",
    "ingestion.pipeline",
    "rag.cli",
    "rag.generator",
    "rag.prompts",
    "rag.retriever",
    "finetune.dataset",
    "finetune.train",
    "eval.golden",
    "eval.judge",
    "eval.metrics",
    "eval.run_eval",
    "eval.scorecard",
    "api.main",
]


@pytest.mark.parametrize("module", MODULES)
def test_module_imports(module: str) -> None:
    importlib.import_module(module)
