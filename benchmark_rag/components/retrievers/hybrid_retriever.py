"""
Hybrid dense+sparse retriever combining FAISS (dense) and BM25 (sparse).

Two fusion modes are supported:

  1. Reranker fusion (preferred): retrieve_candidates() returns the deduplicated
     union of both retriever outputs as a candidate pool.  The caller (e.g.
     HybridRAGPipeline) passes this pool to a cross-encoder reranker (e.g.
     Kanon2Reranker) which scores every candidate against the raw query text.
     This is superior to positional fusion because the reranker reads the actual
     query+document pair rather than relying on rank position.

  2. RRF fusion (fallback, no reranker): retrieve_hybrid() combines the two
     ranked lists with Reciprocal Rank Fusion — score = sum of 1/(rank + rrf_k).
     RRF avoids score-scale mismatches between dense and sparse systems without
     requiring normalisation, but is weaker than a cross-encoder.

Use HybridRAGPipeline (pipeline/hybrid_pipeline.py) to wire this retriever into
a full evaluation run.  The pipeline automatically selects the right fusion path
depending on whether a reranker is configured.
"""
from __future__ import annotations

from pathlib import Path

import logging

from benchmark_rag.components.base import BaseRetriever, EmbeddedChunk, RetrievedChunk
from benchmark_rag.components.retrievers.bm25_retriever import BM25Retriever
from benchmark_rag.components.retrievers.faiss_retriever import FaissRetriever

log = logging.getLogger(__name__)


def _rrf_fuse(
    ranked_lists: list[list[RetrievedChunk]],
    rrf_k: int = 60,
) -> list[RetrievedChunk]:
    """
    Reciprocal Rank Fusion across multiple ranked retrieval lists.

    score(chunk) = sum over lists of  1 / (rank + rrf_k)

    Chunks appearing in multiple lists accumulate a score from each.
    rrf_k (default 60) dampens the top-rank bonus; higher values produce
    smoother, less position-sensitive distributions.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, RetrievedChunk] = {}

    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked):
            key = f"{chunk.doc_id}::{chunk.chunk_idx}"
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + rrf_k)
            chunk_map[key] = chunk

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [
        RetrievedChunk(
            text=chunk_map[k].text,
            doc_id=chunk_map[k].doc_id,
            chunk_idx=chunk_map[k].chunk_idx,
            metadata=chunk_map[k].metadata,
            embedding=chunk_map[k].embedding,
            score=scores[k],
        )
        for k in sorted_keys
    ]


def _merge_candidates(
    faiss_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """
    Deduplicated union of two candidate lists.

    FAISS results come first; BM25 results are appended if not already present.
    Deduplication key is (doc_id, chunk_idx).  The preserved score is whichever
    retriever first contributed the chunk (scores will be overwritten by the
    reranker anyway).
    """
    seen: set[str] = set()
    merged: list[RetrievedChunk] = []
    for chunk in [*faiss_results, *bm25_results]:
        key = f"{chunk.doc_id}::{chunk.chunk_idx}"
        if key not in seen:
            seen.add(key)
            merged.append(chunk)
    return merged


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining FAISS (dense) and BM25 (sparse) sub-indexes.

    Both sub-indexes are built from the same EmbeddedChunk list:
      - FAISS uses the embedding vectors for ANN search.
      - BM25 uses the raw text for term-frequency scoring.

    Two retrieval paths are exposed:

      retrieve_candidates(query_text, query_embedding, n_faiss, n_bm25)
          Returns the deduplicated union of both retrievers' top-N results.
          Feed this pool to a cross-encoder reranker (e.g. Kanon2Reranker)
          for query-aware scoring.  This is the preferred path.

      retrieve_hybrid(query_text, query_embedding, k)
          Returns the top-k results after RRF fusion.  Use this when no
          reranker is available.

    Parameters
    ----------
    metric:
        FAISS distance metric: "cosine" (default) or "l2".
    bm25_k1:
        BM25 term-frequency saturation parameter (default 1.5).
    bm25_b:
        BM25 length-normalisation parameter (default 0.75).
    rrf_k:
        RRF constant used in retrieve_hybrid fallback (default 60).
    faiss_candidates:
        Candidates fetched from FAISS in retrieve_candidates / retrieve_hybrid.
        Defaults to the k or n argument passed at call time * 2 when unset.
    bm25_candidates:
        Candidates fetched from BM25 (same default logic).
    """

    def __init__(
        self,
        metric: str = "cosine",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
        faiss_candidates: int | None = None,
        bm25_candidates: int | None = None,
    ):
        self.metric = metric
        self.rrf_k = rrf_k
        self._faiss_candidates = faiss_candidates
        self._bm25_candidates = bm25_candidates
        self._faiss = FaissRetriever(metric=metric)
        self._bm25 = BM25Retriever(k1=bm25_k1, b=bm25_b)

    # ------------------------------------------------------------------
    # Expose FAISS internals so IndexingPipeline can inspect them
    # without needing to know about the hybrid structure.
    # ------------------------------------------------------------------

    @property
    def _chunks(self) -> list[EmbeddedChunk]:
        return self._faiss._chunks

    @property
    def _index(self):
        return self._faiss._index

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(self, chunks: list[EmbeddedChunk]) -> None:
        """Build both FAISS (dense) and BM25 (sparse) indexes."""
        self._faiss.build_index(chunks)
        self._bm25.build_index(chunks)

    def add_chunks(self, chunks: list[EmbeddedChunk]) -> None:
        """
        Extend the FAISS index incrementally and rebuild BM25 from all chunks
        (rank-bm25 has no incremental add path).
        """
        self._faiss.add_chunks(chunks)
        self._bm25.build_index(self._faiss._chunks)

    def save_index(self, path: str | Path) -> None:
        """
        Save both sub-indexes to disk under the same path prefix.

        Written files:
            <path>.faiss          — FAISS flat index vectors
            <path>.chunks.pkl     — serialised EmbeddedChunk list (shared)
            <path>.bm25.pkl       — BM25 model + chunks
        """
        path = Path(path)
        self._faiss.save_index(path)  # writes .faiss + .chunks.pkl
        self._bm25.save_index(path)   # writes .bm25.pkl

    def load_index(self, path: str | Path) -> None:
        """
        Load both sub-indexes from disk.

        If the BM25 index file does not exist (e.g. the index was originally
        built by a non-hybrid run using the same dataset+chunker+embedder), it
        is rebuilt from the FAISS chunks already in memory and saved immediately
        so subsequent loads are fast.
        """
        path = Path(path)
        self._faiss.load_index(path)
        bm25_path = path.with_suffix(".bm25.pkl")
        if bm25_path.exists():
            self._bm25.load_index(path)
        else:
            log.info(
                "BM25 index not found at %s — rebuilding from %d FAISS chunks.",
                bm25_path, len(self._faiss._chunks),
            )
            self._bm25.build_index(self._faiss._chunks)
            self._bm25.save_index(path)
            log.info("BM25 index saved to %s", bm25_path)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve_candidates(
        self,
        query_text: str,
        query_embedding: list[float],
        n_faiss: int | None = None,
        n_bm25: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Return the deduplicated union of candidates from both sub-retrievers.

        This is the entry point for reranker-based fusion: pass the returned
        pool directly to a cross-encoder reranker (e.g. Kanon2Reranker) which
        will score every candidate against the raw query text and return a
        properly ranked list.

        Parameters
        ----------
        n_faiss:
            Candidates to fetch from FAISS (default: faiss_candidates or 50).
        n_bm25:
            Candidates to fetch from BM25 (default: bm25_candidates or 50).
        """
        n_faiss = n_faiss or self._faiss_candidates or 50
        n_bm25 = n_bm25 or self._bm25_candidates or 50

        faiss_results = self._faiss.retrieve(query_embedding, k=n_faiss)
        bm25_results = self._bm25.retrieve_text(query_text, k=n_bm25)
        return _merge_candidates(faiss_results, bm25_results)

    def retrieve_hybrid(
        self,
        query_text: str,
        query_embedding: list[float],
        k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        RRF fallback fusion — use when no cross-encoder reranker is available.

        Fetches candidates from both sub-retrievers and fuses their ranked
        lists with Reciprocal Rank Fusion, returning the top-k results.
        """
        n_faiss = self._faiss_candidates or k * 2
        n_bm25 = self._bm25_candidates or k * 2

        faiss_results = self._faiss.retrieve(query_embedding, k=n_faiss)
        bm25_results = self._bm25.retrieve_text(query_text, k=n_bm25)

        fused = _rrf_fuse([faiss_results, bm25_results], rrf_k=self.rrf_k)
        return fused[:k]

    def retrieve(self, query_embedding: list[float], k: int = 5) -> list[RetrievedChunk]:
        raise RuntimeError(
            "HybridRetriever requires both query text and an embedding vector. "
            "Use retrieve_candidates() for reranker-based fusion or "
            "retrieve_hybrid() for RRF fallback. "
            "HybridRAGPipeline selects the right path automatically."
        )
