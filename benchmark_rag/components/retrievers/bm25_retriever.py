"""
BM25 sparse retriever built on rank-bm25 (BM25Okapi).

Install the optional dependency before use:
    pip install rank-bm25

Because BM25 scores documents against raw query text — not an embedding vector —
this retriever exposes retrieve_text(query_text, k) as its primary entry point.
The BaseRetriever.retrieve(query_embedding, k) method is implemented but raises
a RuntimeError directing callers to retrieve_text().
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

from benchmark_rag.components.base import BaseRetriever, EmbeddedChunk, RetrievedChunk


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever(BaseRetriever):
    """
    Sparse BM25 retriever.

    Parameters
    ----------
    k1:
        BM25 term-frequency saturation parameter (default 1.5).
    b:
        BM25 length-normalisation parameter (default 0.75).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._bm25 = None
        self._chunks: list[EmbeddedChunk] = []

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[EmbeddedChunk]) -> None:
        from rank_bm25 import BM25Okapi

        self._chunks = chunks
        tokenized_corpus = [_tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def save_index(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix(".bm25.pkl"), "wb") as f:
            pickle.dump({"bm25": self._bm25, "chunks": self._chunks}, f)

    def load_index(self, path: str | Path) -> None:
        path = Path(path)
        with open(path.with_suffix(".bm25.pkl"), "rb") as f:
            data = pickle.load(f)
        self._bm25 = data["bm25"]
        self._chunks = data["chunks"]

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_text(self, query_text: str, k: int = 5) -> list[RetrievedChunk]:
        """Primary entry point for BM25 — takes raw query text."""
        if self._bm25 is None:
            raise RuntimeError("Index not built. Call build_index() or load_index() first.")

        scores = self._bm25.get_scores(_tokenize(query_text))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        return [
            RetrievedChunk(
                text=self._chunks[i].text,
                doc_id=self._chunks[i].doc_id,
                chunk_idx=self._chunks[i].chunk_idx,
                metadata=self._chunks[i].metadata,
                embedding=self._chunks[i].embedding,
                score=float(scores[i]),
            )
            for i in top_indices
        ]

    def retrieve(self, query_embedding: list[float], k: int = 5) -> list[RetrievedChunk]:
        raise RuntimeError(
            "BM25Retriever requires query text, not an embedding. "
            "Use retrieve_text(query_text, k) instead."
        )
