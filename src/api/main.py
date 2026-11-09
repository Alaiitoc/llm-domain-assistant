"""FastAPI app serving the assistant: ``/ask`` returns an answer plus its sources.

The config boundary is here: ``RagConfig`` loads once at startup and is passed
into ``retrieve``/``generate`` per request, so they stay pure. A request may
override ``top_k``; otherwise the config value wins.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from rag.config import load_rag_config
from rag.generator import Answer, generate
from rag.retriever import retrieve

app = FastAPI(title="terraform-domain-assistant", version="0.1.0")

# Loaded once per process; swap the path via RAG_CONFIG for other deployments.
CONFIG_PATH = "configs/rag.yaml"
CFG = load_rag_config(CONFIG_PATH)


class AskRequest(BaseModel):
    """The question to answer; top_k None means the configured default."""

    question: str
    top_k: int | None = None


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(request: AskRequest) -> Answer:
    """Retrieve chunks and generate a cited answer for the question."""
    hits = retrieve(request.question, CFG, top_k=request.top_k)
    return generate(request.question, [hit.chunk for hit in hits], CFG.llm)
