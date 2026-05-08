"""
Pre-compute retrieval for the multi-generator experiment.

For each of the 50 stratified queries, retrieves top-5 chunks per GT document
using the Qwen 2048-chunk FAISS index (IDSelectorBatch). Saves the full
document texts and retrieved chunks — context building (truncation) is
deferred to the generator scripts so each can apply its own token budget.

Output: runs/multi_generator_50/results/shared_retrieval.jsonl

Usage:
    python scripts/multi_gen_retrieve.py

Requires: GPU (Qwen embedder for query embedding)
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from benchmark_rag.components.base import EmbeddedChunk, RetrievedChunk
from benchmark_rag.components.embedders.qwen import QwenEmbedder
from benchmark_rag.config.schemas import ExperimentConfig

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "runs" / "multi_generator_50" / "results"
INDEX_CONFIG = "configs/experiments/qwen_recursive_2048.yaml"
QUERIES_PATH = PROJECT_ROOT / "data" / "test_dataset" / "stratified_50_queries.json"
DATASET_PATH = PROJECT_ROOT / "data" / "test_dataset" / "test_dataset.parquet"

RETRIEVE_K = 5


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    if not QUERIES_PATH.exists():
        sys.exit(f"Run select_stratified_queries.py first. Missing: {QUERIES_PATH}")
    queries = json.loads(QUERIES_PATH.read_text())
    log.info(f"Loaded {len(queries)} stratified queries")

    cfg = ExperimentConfig.from_yaml(PROJECT_ROOT / INDEX_CONFIG)
    index_dir = PROJECT_ROOT / cfg.indexing.output_dir

    faiss_index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "index.chunks.pkl", "rb") as f:
        chunks: list[EmbeddedChunk] = pickle.load(f)
    log.info(f"Loaded FAISS index: {faiss_index.ntotal} vectors, {len(chunks)} chunks")

    doc_to_indices: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        doc_to_indices.setdefault(c.doc_id, []).append(i)

    embedder = QwenEmbedder(model_name="Qwen/Qwen3-Embedding-8B", device="cuda:0")

    doc_df = pd.read_parquet(PROJECT_ROOT / DATASET_PATH, columns=["citation", "text", "url", "name"])
    doc_texts = dict(zip(doc_df["citation"], doc_df["text"]))
    doc_urls = dict(zip(doc_df["citation"], doc_df["url"].fillna("")))
    doc_names = dict(zip(doc_df["citation"], doc_df["name"].fillna("")))
    log.info(f"Loaded {len(doc_texts)} documents")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for q in tqdm(queries, desc="Retrieving"):
        query_text = str(q.get("query_text", ""))
        province = q.get("province", "")
        if province:
            query_text = f"I am in {province}. {query_text}"
        gold_citations = list(q.get("ground_truth_citations", []))

        # FAISS search: top RETRIEVE_K chunks per gold document
        retrieved: list[RetrievedChunk] = []
        query_emb = None

        for cit in gold_citations:
            doc_indices = doc_to_indices.get(cit, [])
            if not doc_indices:
                continue

            if query_emb is None:
                query_emb = np.array(embedder.embed([query_text]), dtype=np.float32)
                faiss.normalize_L2(query_emb)

            id_array = np.array(doc_indices, dtype=np.int64)
            selector = faiss.IDSelectorBatch(id_array)
            params = faiss.SearchParameters(sel=selector)
            k = min(RETRIEVE_K, len(doc_indices))
            scores, indices = faiss_index.search(query_emb, k, params=params)

            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                c = chunks[idx]
                retrieved.append(RetrievedChunk(
                    text=c.text, doc_id=c.doc_id, chunk_idx=c.chunk_idx,
                    metadata=c.metadata, embedding=None, score=float(score),
                ))

        retrieved.sort(key=lambda c: c.score, reverse=True)

        # Chunk details
        chunk_details = []
        for c in retrieved:
            chunk_details.append({
                "doc_id": c.doc_id,
                "chunk_idx": c.chunk_idx,
                "score": round(c.score, 6),
                "text": c.text,
                "url": doc_urls.get(c.doc_id, ""),
                "name": doc_names.get(c.doc_id, ""),
            })

        # Full document texts for all GT docs (not just those with retrieved chunks)
        gt_doc_texts = {}
        for cit in gold_citations:
            if cit in doc_texts:
                gt_doc_texts[cit] = doc_texts[cit]

        # Document metadata
        doc_info = []
        unique_docs = list(dict.fromkeys(
            [c.doc_id for c in retrieved] + gold_citations
        ))
        for did in unique_docs:
            if did in doc_texts:
                doc_info.append({
                    "citation": did,
                    "name": doc_names.get(did, ""),
                    "url": doc_urls.get(did, ""),
                    "char_count": len(doc_texts.get(did, "")),
                    "has_chunks": any(c.doc_id == did for c in retrieved),
                })

        results.append({
            "query_id": q["query_id"],
            "query_text": query_text,
            "gold_citations": gold_citations,
            "retrieved_chunk_details": chunk_details,
            "retrieved_documents": doc_info,
            "gt_doc_texts": gt_doc_texts,
        })

    output_path = RESULTS_DIR / "shared_retrieval.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_with_chunks = sum(1 for r in results if r["retrieved_chunk_details"])
    total_doc_chars = sum(
        sum(len(t) for t in r["gt_doc_texts"].values())
        for r in results
    )
    log.info(f"Saved {len(results)} queries ({n_with_chunks} with chunks) to {output_path}")
    log.info(f"Total GT doc text: {total_doc_chars / 1e6:.1f}M chars")
    log.info(f"File size: {output_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
