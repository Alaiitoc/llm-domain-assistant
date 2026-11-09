"""FastAPI app serving the assistant: ``/ask`` returns an answer plus its sources.

Config loads in the app ``lifespan`` (not at import): importing this module
never touches the filesystem, and the path is repo-relative so any cwd works.
``RAG_CONFIG`` points at another yaml for deployments. Request ``top_k``
overrides the config value; same error boundary as the CLI: operational
failures become 4xx/5xx JSON, bugs keep their traceback.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from httpx import HTTPError
from mistralai.client.errors.sdkerror import SDKError
from pydantic import BaseModel, Field

from rag.config import QUESTION_MAX_CHARS, RagConfig, load_rag_config
from rag.generator import Answer, generate
from rag.retriever import retrieve

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = os.environ.get("RAG_CONFIG", str(REPO_ROOT / "configs" / "rag.yaml"))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load and validate config at startup: fail fast complaining, not on first request."""
    app.state.cfg = load_rag_config(CONFIG_PATH)
    yield


app = FastAPI(title="terraform-domain-assistant", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    """The question to answer; top_k None means the configured default."""

    question: str = Field(max_length=QUESTION_MAX_CHARS)
    top_k: int | None = None


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(request: AskRequest, http_request: Request) -> Answer:
    """Retrieve chunks and generate a cited answer for the question."""
    cfg = cast(RagConfig, http_request.app.state.cfg)
    try:
        hits = retrieve(request.question, cfg, top_k=request.top_k)
        return generate(request.question, [hit.chunk for hit in hits], cfg.llm)
    except HTTPError as exc:
        raise HTTPException(
            status_code=503, detail=f"generation failed: {exc} — is Ollama up?"
        ) from exc
    except SDKError as exc:
        raise HTTPException(status_code=502, detail=f"embedding API error: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
