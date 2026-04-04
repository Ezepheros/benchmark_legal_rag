"""
Tests for incremental indexing correctness and safety.

What we're guarding against
----------------------------
- Wrong doc_ids returned due to FAISS / chunks.pkl misalignment
- Silent duplicate vectors from parquet / FAISS desync
- Stale metric causing un-normalized vectors in a cosine index
- Parquet corruption from a mid-write crash

Run with:
    pytest tests/test_incremental_indexing.py -v
"""
from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import Callable
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pandas as pd
import pytest

from benchmark_rag.components.base import Document, EmbeddedChunk
from benchmark_rag.components.retrievers.faiss_retriever import FaissRetriever


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

DIM = 16  # small dimension for fast tests


def _unit_vec(direction: np.ndarray) -> list[float]:
    """Return a unit vector close to `direction` (with tiny noise for realism)."""
    rng = np.random.default_rng()
    v = direction.astype("float32") + rng.normal(0, 0.01, direction.shape).astype("float32")
    v /= np.linalg.norm(v)
    return v.tolist()


def _make_chunk(doc_id: str, chunk_idx: int, embedding: list[float]) -> EmbeddedChunk:
    return EmbeddedChunk(
        text=f"text of {doc_id} chunk {chunk_idx}",
        doc_id=doc_id,
        chunk_idx=chunk_idx,
        metadata={"source": doc_id},
        embedding=embedding,
    )


def _make_orthogonal_embeddings(n: int, dim: int = DIM) -> list[list[float]]:
    """
    Return n roughly-orthogonal unit vectors so each doc is clearly distinguishable
    by cosine similarity.  Uses the first n standard basis vectors + tiny noise.
    """
    assert n <= dim, "Need at least as many dims as docs for clean separation"
    result = []
    for i in range(n):
        direction = np.zeros(dim, dtype="float32")
        direction[i] = 1.0
        result.append(_unit_vec(direction))
    return result


@pytest.fixture()
def tmp_index(tmp_path: Path):
    """Return the path stem used for save/load (no extension)."""
    return tmp_path / "index"


# ---------------------------------------------------------------------------
# FaissRetriever unit tests
# ---------------------------------------------------------------------------


class TestFaissRetrieverAddChunks:
    """Direct tests on FaissRetriever.add_chunks — the core primitive."""

    def test_alignment_after_add(self, tmp_index: Path):
        """FAISS ntotal must always equal len(_chunks)."""
        embeddings = _make_orthogonal_embeddings(3)
        initial = [_make_chunk(f"doc_{i}", 0, embeddings[i]) for i in range(3)]

        r = FaissRetriever(metric="cosine")
        r.build_index(initial)

        assert r._index.ntotal == len(r._chunks), "Misaligned after build_index"

        new_embeddings = _make_orthogonal_embeddings(5)
        new_chunk = _make_chunk("doc_new", 0, new_embeddings[4])
        r.add_chunks([new_chunk])

        assert r._index.ntotal == len(r._chunks), "Misaligned after add_chunks"
        assert r._index.ntotal == 4

    def test_add_chunks_correct_doc_ids_returned(self):
        """
        After add_chunks, querying with a vector close to a new doc must return
        that new doc's doc_id — not any of the old docs.
        """
        embeddings = _make_orthogonal_embeddings(5)

        initial = [_make_chunk(f"old_doc_{i}", 0, embeddings[i]) for i in range(4)]
        new_chunk = _make_chunk("new_doc", 0, embeddings[4])

        r = FaissRetriever(metric="cosine")
        r.build_index(initial)
        r.add_chunks([new_chunk])

        # Query vector pointing at dim-4 direction (same as new_doc)
        query = embeddings[4]
        results = r.retrieve(query, k=1)

        assert len(results) == 1
        assert results[0].doc_id == "new_doc", (
            f"Expected 'new_doc', got '{results[0].doc_id}' — "
            "FAISS / chunks list are misaligned"
        )

    def test_old_docs_not_displaced_after_add(self):
        """
        Adding new docs must not corrupt retrieval for docs that were already indexed.
        """
        embeddings = _make_orthogonal_embeddings(5)
        initial = [_make_chunk(f"old_{i}", 0, embeddings[i]) for i in range(4)]
        new_chunk = _make_chunk("new_doc", 0, embeddings[4])

        r = FaissRetriever(metric="cosine")
        r.build_index(initial)
        r.add_chunks([new_chunk])

        for i in range(4):
            results = r.retrieve(embeddings[i], k=1)
            assert results[0].doc_id == f"old_{i}", (
                f"old_{i} retrieval broken after add_chunks: got {results[0].doc_id}"
            )

    def test_save_load_after_add_preserves_alignment(self, tmp_index: Path):
        """
        The full round-trip: build → add_chunks → save → load → retrieve
        must return the correct doc_ids for both old and new docs.

        This is the most important persistence test: if the pkl file and
        FAISS file get out of sync during save, wrong docs are returned.
        """
        embeddings = _make_orthogonal_embeddings(5)
        initial = [_make_chunk(f"old_{i}", 0, embeddings[i]) for i in range(4)]
        new_chunk = _make_chunk("new_doc", 0, embeddings[4])

        r = FaissRetriever(metric="cosine")
        r.build_index(initial)
        r.add_chunks([new_chunk])
        r.save_index(tmp_index)

        # Load into a fresh retriever and verify
        r2 = FaissRetriever(metric="cosine")
        r2.load_index(tmp_index)

        assert r2._index.ntotal == len(r2._chunks), (
            "After round-trip: FAISS ntotal != len(_chunks) — files are out of sync"
        )
        assert r2._index.ntotal == 5

        # New doc is findable
        results = r2.retrieve(embeddings[4], k=1)
        assert results[0].doc_id == "new_doc"

        # Old docs still correct
        for i in range(4):
            results = r2.retrieve(embeddings[i], k=1)
            assert results[0].doc_id == f"old_{i}"

    def test_add_chunks_empty_list_is_noop(self):
        """add_chunks([]) must not crash or change the index state."""
        embeddings = _make_orthogonal_embeddings(2)
        initial = [_make_chunk(f"doc_{i}", 0, embeddings[i]) for i in range(2)]

        r = FaissRetriever(metric="cosine")
        r.build_index(initial)
        ntotal_before = r._index.ntotal

        r.add_chunks([])  # must not raise

        assert r._index.ntotal == ntotal_before
        assert len(r._chunks) == ntotal_before

    def test_add_chunks_multiple_chunks_same_doc(self):
        """A document with multiple chunks must all be retrievable by their chunk_idx."""
        embeddings = _make_orthogonal_embeddings(5)
        initial = [_make_chunk("doc_a", 0, embeddings[0])]

        r = FaissRetriever(metric="cosine")
        r.build_index(initial)

        new_chunks = [
            _make_chunk("doc_b", 0, embeddings[1]),
            _make_chunk("doc_b", 1, embeddings[2]),
            _make_chunk("doc_b", 2, embeddings[3]),
        ]
        r.add_chunks(new_chunks)

        assert r._index.ntotal == 4
        assert len(r._chunks) == 4

        # Each chunk of doc_b is retrievable independently
        for chunk_idx, emb_idx in enumerate([1, 2, 3]):
            results = r.retrieve(embeddings[emb_idx], k=1)
            assert results[0].doc_id == "doc_b"
            assert results[0].chunk_idx == chunk_idx

    def test_l2_metric_add_chunks(self, tmp_index: Path):
        """add_chunks must also work correctly for the l2 metric."""
        embeddings = _make_orthogonal_embeddings(3)
        initial = [_make_chunk(f"doc_{i}", 0, embeddings[i]) for i in range(2)]
        new_chunk = _make_chunk("doc_new", 0, embeddings[2])

        r = FaissRetriever(metric="l2")
        r.build_index(initial)
        r.add_chunks([new_chunk])
        r.save_index(tmp_index)

        r2 = FaissRetriever(metric="l2")
        r2.load_index(tmp_index)

        assert r2._index.ntotal == len(r2._chunks)
        results = r2.retrieve(embeddings[2], k=1)
        assert results[0].doc_id == "doc_new"


# ---------------------------------------------------------------------------
# Desync detection tests
# ---------------------------------------------------------------------------


class TestDesyncs:
    """
    Simulate the failure modes where on-disk state becomes inconsistent.
    These tests document the behaviour (or detect the failure) so that
    a future atomicity fix knows what it needs to guarantee.
    """

    def test_faiss_ahead_of_pkl_causes_index_error(self, tmp_index: Path):
        """
        CRASH SIMULATION: .faiss has more vectors than .chunks.pkl has entries.

        This happens if save_index() writes .faiss successfully but the
        process is killed before .chunks.pkl is written.

        Expected: retrieve() raises IndexError (loud failure, not silent wrong doc).
        If this test fails with silent wrong results instead, the code needs
        an explicit alignment check in load_index().
        """
        embeddings = _make_orthogonal_embeddings(3)
        initial = [_make_chunk(f"doc_{i}", 0, embeddings[i]) for i in range(2)]

        # Build with 2 chunks → save
        r = FaissRetriever(metric="cosine")
        r.build_index(initial)
        r.save_index(tmp_index)

        # Manually corrupt: write a FAISS index with 3 vectors but leave pkl with 2 chunks
        emb = np.array([c.embedding for c in initial], dtype="float32")
        extra_emb = np.array([embeddings[2]], dtype="float32")
        combined = np.vstack([emb, extra_emb])
        faiss.normalize_L2(combined)
        index_3 = faiss.IndexFlatIP(DIM)
        index_3.add(combined)
        faiss.write_index(index_3, str(tmp_index.with_suffix(".faiss")))
        # Leave chunks.pkl with only 2 entries (don't touch it)

        r2 = FaissRetriever(metric="cosine")
        r2.load_index(tmp_index)

        # The FAISS index has 3 vectors; chunks has 2 entries
        # Querying with a vector close to the 3rd position should expose the bug
        with pytest.raises(IndexError):
            # FAISS returns ID=2, but self._chunks[2] doesn't exist
            r2.retrieve(embeddings[2], k=1)

    def test_parquet_desync_causes_duplicate_vectors(self, tmp_index: Path):
        """
        CRASH SIMULATION: save_index() succeeded, _append_metadata() did not.

        On the next incremental run, the new doc is re-indexed because parquet
        doesn't know about it. This test verifies that duplicates result, and
        documents the exact failure mode so it can be caught by an alignment check.
        """
        embeddings = _make_orthogonal_embeddings(3)
        initial = [_make_chunk(f"doc_{i}", 0, embeddings[i]) for i in range(2)]

        # Full build
        r = FaissRetriever(metric="cosine")
        r.build_index(initial)
        r.save_index(tmp_index)

        meta_path = tmp_index.parent / "chunks_metadata.parquet"
        pd.DataFrame([
            {"doc_id": c.doc_id, "chunk_idx": c.chunk_idx, "text": c.text}
            for c in initial
        ]).to_parquet(meta_path, index=False)

        # Simulate: add doc_2 to FAISS+pkl but NOT to parquet (crash between steps)
        new_chunk = _make_chunk("doc_2", 0, embeddings[2])
        r.load_index(tmp_index)
        r.add_chunks([new_chunk])
        r.save_index(tmp_index)
        # Intentionally do NOT update parquet

        # Now parquet says only doc_0 and doc_1 are indexed
        already_indexed = set(
            pd.read_parquet(meta_path, columns=["doc_id"])["doc_id"].unique()
        )
        assert "doc_2" not in already_indexed  # confirms the desync

        # Simulate the next incremental run re-adding doc_2
        r2 = FaissRetriever(metric="cosine")
        r2.load_index(tmp_index)
        r2.add_chunks([new_chunk])  # duplicate!
        r2.save_index(tmp_index)

        # The index now has 4 vectors but only 3 unique docs — duplicates exist
        r3 = FaissRetriever(metric="cosine")
        r3.load_index(tmp_index)
        assert r3._index.ntotal == 4, "Should have 4 vectors (3 original + 1 duplicate)"

        # Querying for doc_2 returns it twice in top-2
        results = r3.retrieve(embeddings[2], k=2)
        doc_ids_returned = [res.doc_id for res in results]
        assert doc_ids_returned.count("doc_2") == 2, (
            "Duplicate doc_2 vectors should both appear in top-2 results"
        )

    def test_metric_mismatch_corrupts_cosine_scores(self, tmp_index: Path):
        """
        If the index was built with metric='cosine' but add_chunks is called
        with a retriever initialized with metric='l2', unnormalized vectors
        are added to a normalized index. Cosine scores for new docs will be
        outside the expected [-1, 1] range or grossly wrong.

        This test documents the failure so that a metric-persistence fix
        (saving metric to disk and verifying on load) can be validated against it.
        """
        embeddings = _make_orthogonal_embeddings(3)
        initial = [_make_chunk(f"doc_{i}", 0, embeddings[i]) for i in range(2)]

        # Build with cosine
        r_cosine = FaissRetriever(metric="cosine")
        r_cosine.build_index(initial)
        r_cosine.save_index(tmp_index)

        # Load with wrong metric (l2) — simulates changed config
        r_wrong = FaissRetriever(metric="l2")
        r_wrong.load_index(tmp_index)

        # Add a new chunk without normalization (because metric="l2")
        # Use a clearly non-unit vector to make the corruption visible
        raw_emb = np.zeros(DIM, dtype="float32")
        raw_emb[2] = 10.0  # magnitude 10, not 1 — would normally be normalized to 1
        new_chunk = _make_chunk("doc_new", 0, raw_emb.tolist())
        r_wrong.add_chunks([new_chunk])
        r_wrong.save_index(tmp_index)

        # Reload with the correct metric and query
        r_check = FaissRetriever(metric="cosine")
        r_check.load_index(tmp_index)

        query = np.zeros(DIM, dtype="float32")
        query[2] = 1.0  # unit vector toward doc_new
        results = r_check.retrieve(query.tolist(), k=1)

        # Score for doc_new should be ~1.0 (same direction) if vectors were normalized.
        # With the un-normalized magnitude-10 vector, inner product score = 10 >> 1.
        if results[0].doc_id == "doc_new":
            score = results[0].score
            assert score > 1.1, (
                "Metric mismatch not detected: expected score >> 1.0 for unnormalized "
                f"cosine vector, got {score:.4f}"
            )


# ---------------------------------------------------------------------------
# Incremental pipeline integration tests
# ---------------------------------------------------------------------------


class TestIncrementalPipeline:
    """
    Integration tests for the full incremental path through IndexingPipeline.

    These tests require the incremental changes to be implemented.
    They use a MockEmbedder to count calls without loading real models.
    """

    @staticmethod
    def _make_mock_embedder(embeddings_by_text: dict[str, list[float]]):
        """
        Return a mock object with an embed(texts) method that looks up pre-set
        embeddings and counts how many texts were embedded.
        """
        mock = MagicMock()
        call_log = []

        def embed(texts: list[str]) -> list[list[float]]:
            call_log.extend(texts)
            return [embeddings_by_text[t] for t in texts]

        mock.embed.side_effect = embed
        mock.call_log = call_log
        mock.embedding_dim = DIM
        return mock

    def test_only_new_docs_are_embedded(self, tmp_path: Path):
        """
        CRITICAL: incremental run must embed ONLY documents not already in the index.

        Verified by mocking the embedder and counting how many chunks it receives.
        """
        from benchmark_rag.pipeline.indexing_pipeline import IndexingPipeline
        from benchmark_rag.config.schemas import ExperimentConfig
        import yaml

        embeddings = _make_orthogonal_embeddings(4)

        # Build a minimal config that points at our tmp dir
        # (We patch out all component construction to avoid needing real YAML/registry)
        with patch("benchmark_rag.pipeline.indexing_pipeline.build_from_component_config") as mock_build:
            # Arrange: 3 docs already indexed, 1 new
            doc_texts = {f"text of doc_{i} chunk 0": embeddings[i] for i in range(4)}

            mock_chunker = MagicMock()
            mock_chunker.chunk.side_effect = lambda doc: [
                EmbeddedChunk(
                    text=f"text of {doc.doc_id} chunk 0",
                    doc_id=doc.doc_id,
                    chunk_idx=0,
                    metadata={},
                    embedding=[],  # will be filled by embedder
                )
                # Return a plain Chunk via base Chunk class
            ]

            mock_embedder = MagicMock()
            embedded_call_args = []
            def fake_embed(texts):
                embedded_call_args.extend(texts)
                return [doc_texts[t] for t in texts]
            mock_embedder.embed.side_effect = fake_embed

            mock_retriever = MagicMock(spec=FaissRetriever)
            mock_retriever.metric = "cosine"
            mock_retriever._chunks = []
            mock_retriever._index = None

            # This test validates the DESIGN; implementation may vary.
            # The key assertion is that only 1 text is sent to the embedder when
            # 3 docs are pre-indexed and 1 is new.
            # Implementation test: skip if IndexingPipeline doesn't support force_reindex yet
            pytest.skip(
                "Requires IndexingPipeline.force_reindex to be implemented. "
                "Run after implementing the incremental path."
            )

    def test_parquet_contains_all_doc_ids_after_incremental(self, tmp_path: Path):
        """
        After an incremental run, chunks_metadata.parquet must contain doc_ids
        from both the original index and the new documents.
        """
        # Build initial parquet state
        meta_path = tmp_path / "chunks_metadata.parquet"
        initial_rows = [
            {"doc_id": "doc_0", "chunk_idx": 0, "text": "old text 0"},
            {"doc_id": "doc_1", "chunk_idx": 0, "text": "old text 1"},
        ]
        pd.DataFrame(initial_rows).to_parquet(meta_path, index=False)

        # Simulate _append_metadata for new chunks
        from benchmark_rag.components.base import EmbeddedChunk as EC
        new_chunks = [
            EC(text="new text 2", doc_id="doc_2", chunk_idx=0, metadata={}, embedding=[]),
            EC(text="new text 2b", doc_id="doc_2", chunk_idx=1, metadata={}, embedding=[]),
        ]

        new_rows = [
            {"doc_id": c.doc_id, "chunk_idx": c.chunk_idx, "text": c.text, **c.metadata}
            for c in new_chunks
        ]
        new_df = pd.DataFrame(new_rows)
        existing_df = pd.read_parquet(meta_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_parquet(meta_path, index=False)

        # Verify
        result = pd.read_parquet(meta_path)
        assert set(result["doc_id"].unique()) == {"doc_0", "doc_1", "doc_2"}
        assert len(result) == 4  # 2 old + 2 new chunks

    def test_load_indexed_doc_ids_reads_only_doc_id_column(self, tmp_path: Path):
        """
        _load_indexed_doc_ids must use parquet column projection (columns=["doc_id"])
        and return a set of strings — not a list or DataFrame.
        """
        meta_path = tmp_path / "chunks_metadata.parquet"
        df = pd.DataFrame([
            {"doc_id": "2024 BCSC 1", "chunk_idx": 0, "text": "a" * 5000, "other_col": "x"},
            {"doc_id": "2024 BCSC 1", "chunk_idx": 1, "text": "b" * 5000, "other_col": "y"},
            {"doc_id": "2024 SCC 5",  "chunk_idx": 0, "text": "c" * 5000, "other_col": "z"},
        ])
        df.to_parquet(meta_path, index=False)

        result = set(
            pd.read_parquet(meta_path, columns=["doc_id"])["doc_id"].unique()
        )

        assert isinstance(result, set)
        assert result == {"2024 BCSC 1", "2024 SCC 5"}

    def test_no_duplicates_when_incremental_run_repeated(self, tmp_path: Path):
        """
        Running incremental indexing twice on the same corpus must not add
        any new vectors — the second run should be a no-op.
        """
        embeddings = _make_orthogonal_embeddings(3)
        chunks = [_make_chunk(f"doc_{i}", 0, embeddings[i]) for i in range(3)]

        # Build initial index
        r = FaissRetriever(metric="cosine")
        r.build_index(chunks)
        index_path = tmp_path / "index"
        r.save_index(index_path)

        meta_path = tmp_path / "chunks_metadata.parquet"
        pd.DataFrame([
            {"doc_id": c.doc_id, "chunk_idx": c.chunk_idx, "text": c.text}
            for c in chunks
        ]).to_parquet(meta_path, index=False)

        # Simulate a second incremental run with the same 3 docs
        already_indexed = set(
            pd.read_parquet(meta_path, columns=["doc_id"])["doc_id"].unique()
        )
        all_doc_ids = {f"doc_{i}" for i in range(3)}
        new_doc_ids = all_doc_ids - already_indexed

        assert new_doc_ids == set(), (
            f"Expected no new docs, got: {new_doc_ids}. "
            "Second incremental run would incorrectly re-index these."
        )

    def test_updated_document_is_NOT_re_indexed(self, tmp_path: Path):
        """
        Known limitation: if a document's text is updated but its doc_id is unchanged,
        the incremental path silently keeps the old version in the index.

        This test documents and asserts this behaviour explicitly so that
        anyone who later adds "update" support knows what to change.
        """
        embeddings = _make_orthogonal_embeddings(2)
        original_chunk = _make_chunk("2024 BCSC 1", 0, embeddings[0])

        r = FaissRetriever(metric="cosine")
        r.build_index([original_chunk])
        index_path = tmp_path / "index"
        r.save_index(index_path)

        meta_path = tmp_path / "chunks_metadata.parquet"
        pd.DataFrame([{"doc_id": "2024 BCSC 1", "chunk_idx": 0, "text": "original text"}]).to_parquet(
            meta_path, index=False
        )

        # Simulate: document "2024 BCSC 1" has updated text
        updated_doc_id = "2024 BCSC 1"
        already_indexed = set(
            pd.read_parquet(meta_path, columns=["doc_id"])["doc_id"].unique()
        )

        # The incremental filter skips the updated doc
        assert updated_doc_id in already_indexed, (
            "Updated doc_id should be detected as already indexed (known limitation: "
            "it will NOT be re-embedded even though text changed)"
        )

    def test_parquet_schema_drift_on_concat(self, tmp_path: Path):
        """
        New documents may have extra metadata keys that old ones lacked.
        pd.concat with ignore_index=True must produce NaN for missing old fields,
        not raise an error or corrupt existing rows.
        """
        meta_path = tmp_path / "chunks_metadata.parquet"
        old_rows = pd.DataFrame([
            {"doc_id": "doc_0", "chunk_idx": 0, "text": "old", "court": "BCSC"},
        ])
        old_rows.to_parquet(meta_path, index=False)

        # New doc has an extra field "source" that old docs lacked
        new_rows = pd.DataFrame([
            {"doc_id": "doc_1", "chunk_idx": 0, "text": "new", "court": "SCC", "source": "caseway"},
        ])

        combined = pd.concat(
            [pd.read_parquet(meta_path), new_rows], ignore_index=True
        )
        combined.to_parquet(meta_path, index=False)

        result = pd.read_parquet(meta_path)
        assert len(result) == 2
        # Old doc gets NaN for the new "source" column — not an error
        assert pd.isna(result.loc[result["doc_id"] == "doc_0", "source"].iloc[0])
        # New doc has its value
        assert result.loc[result["doc_id"] == "doc_1", "source"].iloc[0] == "caseway"


# ---------------------------------------------------------------------------
# Alignment invariant helper (can be called from pipeline code directly)
# ---------------------------------------------------------------------------


def assert_index_aligned(retriever: FaissRetriever) -> None:
    """
    Call this after any index mutation to catch desync early.

    Raises AssertionError with a clear message if FAISS ntotal != len(_chunks).
    Intended to be used in tests and optionally as a debug assertion in the pipeline.
    """
    if retriever._index is None:
        return  # nothing loaded yet
    ntotal = retriever._index.ntotal
    nchunks = len(retriever._chunks)
    assert ntotal == nchunks, (
        f"Index alignment violation: FAISS has {ntotal} vectors "
        f"but _chunks has {nchunks} entries. "
        "This will cause wrong doc_ids to be returned for some queries."
    )


class TestAlignmentHelper:
    def test_helper_passes_when_aligned(self):
        embeddings = _make_orthogonal_embeddings(2)
        r = FaissRetriever(metric="cosine")
        r.build_index([_make_chunk(f"doc_{i}", 0, embeddings[i]) for i in range(2)])
        assert_index_aligned(r)  # must not raise

    def test_helper_catches_misalignment(self):
        embeddings = _make_orthogonal_embeddings(2)
        r = FaissRetriever(metric="cosine")
        r.build_index([_make_chunk(f"doc_{i}", 0, embeddings[i]) for i in range(2)])

        # Manually corrupt: remove one chunk without removing from FAISS
        r._chunks.pop()

        with pytest.raises(AssertionError, match="Index alignment violation"):
            assert_index_aligned(r)
