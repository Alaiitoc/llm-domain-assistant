"""Typed view of configs/rag.yaml, plus its loader.

Loaded at the entry points (``rag.cli``, ``api.main``, later ``eval.run_eval``)
so the functions they call stay pure: they receive a validated RagConfig,
never a file path. Nested sections get real models (``llm: LLMConfig``) rather
than loose dicts, so a misspelled key anywhere fails at load time with the exact
key named.
"""

from __future__ import annotations

import yaml
from pydantic import BaseModel


class LLMConfig(BaseModel):
    """Where answers are generated: Ollama locally, the adapter endpoint after fine-tuning.

    timeout and max_tokens are machine-tuned: this laptop runs the 7B on CPU at
    ~0.7 tokens/s, so a grounded answer (~1100-token prompt, ~400-token answer)
    needs minutes, not seconds. On a GPU box these can be drastically smaller.
    """

    base_url: str
    model: str
    timeout: float = 900.0
    max_tokens: int = 512


class RagConfig(BaseModel):
    """Serving-side values: what to search, with which embedding, how many hits."""

    collection: str
    qdrant_url: str
    # Must match configs/ingestion.yaml: query vectors are only comparable to
    # document vectors from the same embedding model.
    embedding_model: str
    top_k: int
    reranker: str | None = None
    llm: LLMConfig


# Shared by both entry points (cli + api): a bound on the question protects the
# embedding-API bill and the context window from oversized input. A real
# Terraform question is a few hundred characters at most.
QUESTION_MAX_CHARS = 2_000


def load_rag_config(path: str) -> RagConfig:
    """Load and validate the RAG yaml; wrong keys fail fast, loudly, named."""
    with open(path, encoding="utf-8") as f:
        return RagConfig.model_validate(yaml.safe_load(f))
