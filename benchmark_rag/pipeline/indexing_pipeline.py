"""
Indexing pipeline: Document → Chunks → EmbeddedChunks → FAISS index.

Responsibilities:
  - Load documents from a dataset
  - Chunk with the configured chunker
  - Embed in batches with the configured embedder
  - Build and persist the FAISS index
  - Save chunk metadata to parquet for later inspection

Usage
-----
    from benchmark_rag.config.schemas import ExperimentConfig
    from benchmark_rag.pipeline.indexing_pipeline import IndexingPipeline

    cfg = ExperimentConfig.from_yaml("configs/experiments/qwen_1024.yaml")
    pipeline = IndexingPipeline(cfg)
    pipeline.run(documents)
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from benchmark_rag.components.base import BaseChunker, BaseEmbedder, BaseRetriever, Document, EmbeddedChunk
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.logging import get_logger, log_resource_snapshot
from benchmark_rag.registry import build_from_component_config


class IndexingPipeline:
    """
    Runs the full indexing flow for one experiment configuration.

    Parameters
    ----------
    cfg:
        Validated ExperimentConfig.  All component construction is deferred
        until run() is called so the object is cheap to create.
    """

    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.log = get_logger(__name__)
        self._output_dir = Path(cfg.indexing.output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, documents: list[Document]) -> Path:
        """
        Index a list of Documents.

        Returns the path to the saved FAISS index directory.
        """
        cfg = self.cfg
        self.log.info(
            f"Indexing {len(documents)} documents | experiment={cfg.experiment_id}"
        )
        t0 = time.perf_counter()

        chunker: BaseChunker = build_from_component_config(cfg.chunker.to_build_dict())
        embedder: BaseEmbedder = build_from_component_config(cfg.embedder.to_build_dict())
        retriever: BaseRetriever = build_from_component_config(cfg.retriever.to_build_dict())

        # --- Chunking ---
        self.log.info("Chunking documents...")
        all_chunks = []
        for doc in tqdm(documents, desc="Chunking"):
            all_chunks.extend(chunker.chunk(doc))
        self.log.info(f"Produced {len(all_chunks)} chunks from {len(documents)} documents")

        # --- Embedding (batched) ---
        self.log.info("Embedding chunks...")
        embedded_chunks = self._embed_chunks(all_chunks, embedder)
        self.log.info(f"Embedded {len(embedded_chunks)} chunks")

        # --- Build + save index ---
        self.log.info("Building FAISS index...")
        retriever.build_index(embedded_chunks)
        index_path = self._output_dir / "index"
        retriever.save_index(index_path)
        self.log.info(f"Index saved to {index_path}")

        # --- Save chunk metadata for inspection ---
        self._save_metadata(embedded_chunks)

        elapsed = time.perf_counter() - t0
        self.log.info(f"Indexing complete in {elapsed:.1f}s")
        log_resource_snapshot(self.log)

        return self._output_dir

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _embed_chunks(self, chunks, embedder: BaseEmbedder) -> list[EmbeddedChunk]:
        from benchmark_rag.components.base import EmbeddedChunk

        batch_size = self.cfg.indexing.embedding_batch_size
        embedded: list[EmbeddedChunk] = []
        save_every = self.cfg.indexing.save_intermediate_every

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = embedder.embed(texts)

            for chunk, emb in zip(batch, embeddings):
                embedded.append(
                    EmbeddedChunk(
                        text=chunk.text,
                        doc_id=chunk.doc_id,
                        chunk_idx=chunk.chunk_idx,
                        metadata=chunk.metadata,
                        embedding=emb,
                    )
                )

            if len(embedded) % save_every < batch_size:
                self.log.info(f"Embedded {len(embedded)}/{len(chunks)} chunks")
                log_resource_snapshot(self.log)

        return embedded

    def _save_metadata(self, chunks: list[EmbeddedChunk]) -> None:
        """Save chunk text + metadata (without embeddings) to parquet."""
        rows = [
            {
                "doc_id": c.doc_id,
                "chunk_idx": c.chunk_idx,
                "text": c.text,
                **c.metadata,
            }
            for c in chunks
        ]
        df = pd.DataFrame(rows)
        out = self._output_dir / "chunks_metadata.parquet"
        df.to_parquet(out, index=False)
        self.log.info(f"Chunk metadata saved to {out}")
