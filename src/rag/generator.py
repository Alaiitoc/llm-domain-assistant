"""Generate a grounded answer from retrieved chunks via the configured LLM.

Speaks to Ollama's OpenAI-compatible endpoint with plain httpx (``base_url``
already ends in ``/v1``); no extra SDK dependency for one POST call.
``temperature=0`` keeps answers reproducible — they get evaled later.
"""

from __future__ import annotations

import re

import httpx
from pydantic import BaseModel

from ingestion.chunk import Chunk
from rag.config import LLMConfig
from rag.prompts import ANSWER_SYSTEM_PROMPT, ANSWER_USER_TEMPLATE, PROMPT_VERSION

_CITED_NUMBER = re.compile(r"\[(\d+)\]")


class Answer(BaseModel):
    """A grounded answer with the chunk ids it cites and the prompt that made it."""

    text: str
    sources: list[str] = []
    prompt_version: str = PROMPT_VERSION


def build_context(chunks: list[Chunk]) -> str:
    """Number the chunks [1]..[n] in retrieval order — the citation contract."""
    return "\n\n".join(f"[{i}] {chunk.text}" for i, chunk in enumerate(chunks, start=1))


def build_messages(question: str, chunks: list[Chunk]) -> list[dict[str, str]]:
    """Assemble the chat messages from the versioned templates."""
    return [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": ANSWER_USER_TEMPLATE.format(
                context=build_context(chunks), question=question
            ),
        },
    ]


def cited_sources(answer_text: str, chunks: list[Chunk]) -> list[str]:
    """Distinct source_urls of the chunks the answer actually cites.

    Citations out of range are ignored; if the model cites nothing, fall back
    to every retrieved source — an answer with no sources is a UX bug.
    """
    numbers = {int(n) for n in _CITED_NUMBER.findall(answer_text)}
    indices = sorted(n for n in numbers if 1 <= n <= len(chunks))
    sources: list[str] = []
    for i in indices:
        url = chunks[i - 1].source_url
        if url not in sources:
            sources.append(url)
    if not sources:
        sources = [chunk.source_url for chunk in chunks]
    return sources


def generate(question: str, chunks: list[Chunk], llm: LLMConfig) -> Answer:
    """Fill the answer prompt with ``chunks`` and call the LLM endpoint."""
    if not chunks:
        return Answer(text="No relevant documentation was retrieved for this question.")

    response = httpx.post(
        f"{llm.base_url}/chat/completions",
        json={
            "model": llm.model,
            "messages": build_messages(question, chunks),
            "temperature": 0,
            "max_tokens": llm.max_tokens,
        },
        timeout=llm.timeout,
    )
    response.raise_for_status()
    answer_text = response.json()["choices"][0]["message"]["content"]
    return Answer(text=answer_text, sources=cited_sources(answer_text, chunks))
