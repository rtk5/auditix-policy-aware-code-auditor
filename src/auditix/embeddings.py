"""Gemini embedding wrapper.

Policies are embedded with ``RETRIEVAL_DOCUMENT`` and code summaries with
``RETRIEVAL_QUERY``. Gemini uses these task types to place documents and
queries in a space where a query matches the most relevant document.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from auditix.config import GEMINI_EMBED_MODEL, get_gemini_api_key


class GeminiEmbedder:
    """Thin wrapper around ``google-genai``. The client is created lazily."""

    def __init__(self, api_key: str | None = None, model: str = GEMINI_EMBED_MODEL):
        self._api_key = api_key or get_gemini_api_key()
        self._model = model
        self._client = None  # created on first use, so construction is cheap and offline

    def _get_client(self):
        """Create the Gemini client once, on first use."""
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("GEMINI_API_KEY is not set (needed for policy embeddings)")
            from google import genai  # imported lazily so tests do not need the SDK

            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _embed(self, text: str, task_type: str) -> list[float]:
        """Embed one string with the given Gemini task type and return its vector."""
        from google.genai import types

        response = self._get_client().models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return response.embeddings[0].values

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        """Embed policies (or other documents). Shape: (n, dim), float32."""
        # One API call per text. Simple and matches the notebook; batching would be faster.
        vectors = [self._embed(t, "RETRIEVAL_DOCUMENT") for t in texts]
        return np.asarray(vectors, dtype="float32")

    def embed_query(self, text: str) -> np.ndarray:
        """Embed one query. Shape: (1, dim), float32."""
        vector = self._embed(text, "RETRIEVAL_QUERY")
        return np.asarray(vector, dtype="float32").reshape(1, -1)  # FAISS expects a 2-D batch
