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
    """Anything that can embed documents and queries (Gemini in production, a fake in tests)."""

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class PolicyIndex:
    """Policy texts plus a FAISS index whose row ``i`` holds the vector of ``policies[i]``."""

    def __init__(self, policies: Sequence[str], index, embedder: Embedder | None = None):
        self.policies: list[str] = list(policies)
        self._index = index          # the FAISS index object
        self._embedder = embedder    # needed only for search(); save/load work without it

    @property
    def size(self) -> int:
        """Number of policies in the index."""
        return int(self._index.ntotal)

    @classmethod
    def build(cls, policies: Sequence[str], embedder: Embedder) -> "PolicyIndex":
        """Embed every policy and load the vectors into a new exact (flat L2) index."""
        import faiss

        vectors = embedder.embed_documents(list(policies))
        index = faiss.IndexFlatL2(vectors.shape[1])  # exact search, fine for small policy sets
        index.add(vectors)
        logger.info("Built FAISS index: %d policies, dimension %d", index.ntotal, vectors.shape[1])
        return cls(policies, index, embedder)

    def save(self, directory: str | Path) -> None:
        """Write the FAISS index and the policy list to ``directory``."""
        import faiss

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / INDEX_FILE))
        (directory / POLICIES_FILE).write_text(json.dumps(self.policies, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, embedder: Embedder | None = None) -> "PolicyIndex":
        """Read an index saved by :meth:`save`. Pass an embedder if you intend to search."""
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
            # Compare the saved text with the current list: any edit forces a rebuild.
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
        k = min(top_k, self.size)  # cannot ask FAISS for more neighbours than it holds
        query = self._embedder.embed_query(query_text)
        _, indices = self._index.search(query, k)  # distances are ignored; only order matters
        return [self.policies[i] for i in indices[0] if 0 <= i < len(self.policies)]
