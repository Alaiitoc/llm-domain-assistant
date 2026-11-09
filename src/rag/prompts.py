"""Versioned prompt templates: answers grounded in retrieved chunks, with citations.

``PROMPT_VERSION`` must change whenever a template changes: eval and judge
scores are only comparable across runs that used the same prompt version.

The citation contract: chunks are numbered in the context block, in the same
order they were retrieved ([1] = top-ranked chunk); the model cites claims
with those numbers.
"""

from __future__ import annotations

PROMPT_VERSION = "v2"  # v1 wrongly claimed to be a Mistral products assistant

ANSWER_SYSTEM_PROMPT = """\
You are a documentation assistant for Terraform (by HashiCorp).
Answer ONLY from the provided context chunks.
Cite every claim with [n], where n is the number of the chunk it came from,
as given by the [n] header above each chunk in the context.
If the context does not contain the answer, say so explicitly instead of guessing.
"""

ANSWER_USER_TEMPLATE = """\
Context chunks:
{context}

Question: {question}
"""
