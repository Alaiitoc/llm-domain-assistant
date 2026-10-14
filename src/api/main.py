"""FastAPI app serving the assistant: ``/ask`` returns an answer plus its sources."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.generator import Answer, generate
from rag.retriever import retrieve

app = FastAPI(title="terraform-domain-assistant", version="0.1.0")


class AskRequest(BaseModel):
    """The question to answer."""

    question: str
    top_k: int = 5


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/ask", response_model=Answer)
def ask(request: AskRequest) -> Answer:
    """Retrieve chunks and generate a cited answer for the question."""
    try:
        chunks = retrieve(request.question, top_k=request.top_k)
        return generate(request.question, chunks)
    except NotImplementedError as exc:
        detail = str(exc) or "RAG pipeline not implemented yet"
        raise HTTPException(status_code=501, detail=detail) from exc
