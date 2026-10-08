"""FAISS index over policy embeddings — the retrieval half of the RAG pipeline.

Index layout on disk (``models/policy_index/`` by default)::

    index.faiss      the binary FAISS index (IndexFlatL2)
    policies.json    the policy texts in the same order as the index rows

The index is rebuilt automatically when the policy text changes, so the
``policies.json`` file always matches the vectors in ``index.faiss``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

logger = logging.getLogger(__name__)

INDEX_FILE = "index.faiss"
POLICIES_FILE = "policies.json"


class Embedder(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class PolicyIndex:
    def __init__(self, policies: Sequence[str], index, embedder: Embedder | None = None):
        self.policies: list[str] = list(policies)
        self._index = index
        self._embedder = embedder

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    @classmethod
    def build(cls, policies: Sequence[str], embedder: Embedder) -> "PolicyIndex":
        import faiss

        vectors = embedder.embed_documents(list(policies))
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)
        logger.info("Built FAISS index: %d policies, dimension %d", index.ntotal, vectors.shape[1])
        return cls(policies, index, embedder)

    def save(self, directory: str | Path) -> None:
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / INDEX_FILE))
        (directory / POLICIES_FILE).write_text(json.dumps(self.policies, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder | None = None) -> "PolicyIndex":
        import faiss

        directory = Path(directory)
        index = faiss.read_index(str(directory / INDEX_FILE))
        policies = json.loads((directory / POLICIES_FILE).read_text(encoding="utf-8"))
        return cls(policies, index, embedder)

    @classmethod
    def load_or_build(
        cls, policies: Sequence[str], directory: str | Path, embedder: Embedder
    ) -> "PolicyIndex":
        """Reuse the saved index if it was built from exactly these policies."""
        directory = Path(directory)
        if (directory / INDEX_FILE).exists() and (directory / POLICIES_FILE).exists():
            saved = cls.load(directory, embedder)
            if saved.policies == list(policies):
                logger.info("Loaded cached policy index from %s", directory)
                return saved
            logger.info("Policy list changed — rebuilding index")
        built = cls.build(policies, embedder)
        built.save(directory)
        return built

    def search(self, query_text: str, top_k: int = 3) -> list[str]:
        """Return the ``top_k`` policies closest to ``query_text`` (best first)."""
        if self._embedder is None:
            raise RuntimeError("PolicyIndex has no embedder attached; cannot embed queries")
        k = min(top_k, self.size)
        query = self._embedder.embed_query(query_text)
        _, indices = self._index.search(query, k)
        return [self.policies[i] for i in indices[0] if 0 <= i < len(self.policies)]
