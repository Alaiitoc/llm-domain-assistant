"""Versioned prompt templates: answers grounded in retrieved chunks, with citations."""

ANSWER_SYSTEM_PROMPT = """\
You are a documentation assistant for Mistral AI products.
Answer ONLY from the provided context chunks.
Cite every claim with [n], where n is the number of the chunk it came from.
If the context does not contain the answer, say so explicitly instead of guessing.
"""

ANSWER_USER_TEMPLATE = """\
Context chunks:
{context}

Question: {question}
"""
