"""Connection factories: build clients, isolate env/auth concerns.

Only construction lives here — no embedding, search, or schema logic — so
deployment concerns (keys, URLs) stay reviewable apart from the code that
uses the connections. Config values are passed in by callers; this module
only reads secrets from the environment.
"""

from __future__ import annotations

import os

from mistralai.client import Mistral
from qdrant_client import QdrantClient


def mistral_client() -> Mistral:
    """Build the Mistral client; the key lives in the env, never in config."""
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is not set. Put it in the environment"
            " (e.g. source .env); config files never hold secrets."
        )
    return Mistral(api_key=api_key)


def qdrant_client(url: str = "http://localhost:6333") -> QdrantClient:
    """Build the Qdrant client; the URL comes from configs, not from here."""
    return QdrantClient(url=url)
