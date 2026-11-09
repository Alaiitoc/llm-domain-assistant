"""Unit tests for rag.config — loading and validation, no network."""

from __future__ import annotations

import pathlib

import pytest
from pydantic import ValidationError

from rag.config import load_rag_config

VALID_YAML = """\
collection: terraform_docs
qdrant_url: http://localhost:6333
embedding_model: mistral-embed
top_k: 5
llm:
  base_url: http://localhost:11434/v1
  model: mistral:7b-instruct-v0.3-q4_K_M
"""


class TestLoadRagConfig:
    def test_full_config_loads_typed(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "rag.yaml"
        path.write_text(VALID_YAML, encoding="utf-8")
        cfg = load_rag_config(str(path))
        assert cfg.collection == "terraform_docs"
        assert cfg.top_k == 5
        assert cfg.llm.base_url == "http://localhost:11434/v1"
        assert cfg.llm.model == "mistral:7b-instruct-v0.3-q4_K_M"

    def test_missing_key_fails_fast_named(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "rag.yaml"
        path.write_text("collection: terraform_docs\ntop_k: 5\n", encoding="utf-8")
        with pytest.raises(ValidationError, match="qdrant_url"):
            load_rag_config(str(path))

    def test_misspelled_nested_key_fails(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / "rag.yaml"
        path.write_text(VALID_YAML.replace("base_url", "base_ur"), encoding="utf-8")
        with pytest.raises(ValidationError):
            load_rag_config(str(path))  # nested model catches it; a dict would not
