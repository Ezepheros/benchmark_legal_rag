"""
BM25 benchmark with query rewriting.

Identical to run_bm25_benchmark.py except each query is rewritten into legal
language by the configured query_rewriter before BM25 retrieval.
The original query text is preserved in query_results.jsonl as original_query_text.

Usage
-----
    python scripts/run_bm25_benchmark_rewrite_query.py \
        --config configs/experiments/bm25_recursive_1024_rewrite_query.yaml
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import pandas as pd
from tqdm import tqdm

from benchmark_rag.components.retrievers.bm25_retriever import BM25Retriever
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.evaluation.metrics import evaluate_retrieval
from benchmark_rag.logging import setup_experiment_logging, get_logger
from benchmark_rag.registry import build_from_component_config


def load_queries(queries_path: str) -> list[dict]:
    p = Path(queries_path)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    elif p.suffix == ".pkl":
        with open(p, "rb") as f:
            data = pickle.load(f)
        if hasattr(data, "to_dict"):
            return data.to_dict(orient="records")
        return data
    elif p.suffix == ".parquet":
        return pd.read_parquet(p).to_dict(orient="records")
    raise ValueError(f"Unsupported query file format: {p.suffix}")


def main():
    parser = argparse.ArgumentParser(
        description="BM25 benchmark with pre-retrieval query rewriting."
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)

    if cfg.query_rewriter is None:
        sys.exit(
            "ERROR: No query_rewriter configured. "
            "Add a query_rewriter section to your experiment YAML."
        )

    import os
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        sys.exit("ERROR: GOOGLE_API_KEY or GEMINI_API_KEY is not set. Required for query rewriting.")

    setup_experiment_logging(
        experiment_id=cfg.experiment_id,
        log_dir=cfg.logging.log_dir,
        level=cfg.logging.level,
        resource_monitor_interval=0,
    )
    log = get_logger(__name__)

    k1 = float(cfg.retriever.model_extra.get("k1", 1.5))
    b  = float(cfg.retriever.model_extra.get("b",  0.75))

    log.info("=" * 60)
    log.info(f"EXPERIMENT    : {cfg.experiment_id}")
    log.info(f"DESCRIPTION   : {cfg.description}")
    log.info(f"QUERY REWRITER: {cfg.query_rewriter.type}")
    log.info(f"BM25 params   : k1={k1}  b={b}")
    log.info(f"INDEX ID      : {cfg.index_id}")
    log.info("=" * 60)

    # --- Build query rewriter ---
    rewriter = build_from_component_config(cfg.query_rewriter.to_build_dict())
    log.info(f"Query rewriter ready: {cfg.query_rewriter.type}")

    # --- Load chunks from existing index ---
    chunks_path = Path(cfg.indexing.output_dir) / "index.chunks.pkl"
    if not chunks_path.exists():
        log.error(f"Chunks file not found at {chunks_path}. Run run_indexing.py first.")
        sys.exit(1)

    log.info(f"Loading chunks from {chunks_path}")
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    log.info(f"  {len(chunks)} chunks loaded.")

    # Build doc → chunk count for recall denominator
    chunks_meta_path = Path(cfg.indexing.output_dir) / "chunks_metadata.parquet"
    doc_chunk_count: dict[str, int] = {}
    if chunks_meta_path.exists():
        _meta_df = pd.read_parquet(chunks_meta_path, columns=["doc_id"])
        doc_chunk_count = _meta_df["doc_id"].value_counts().to_dict()

    # --- Build BM25 index ---
    log.info(f"Building BM25 index (k1={k1}, b={b}) ...")
    retriever = BM25Retriever(k1=k1, b=b)
    retriever.build_index(chunks)
    log.info("  Done.")

    # --- Load queries ---
    queries = load_queries(cfg.evaluation.queries_path)
    log.info(f"Loaded {len(queries)} queries from {cfg.evaluation.queries_path}")

    # --- Run queries ---
    k_values = cfg.evaluation.k_values
    max_k = max(k_values)

    all_retrieved: list[list[str]] = []
    all_relevant: list[set[str]] = []
    rows = []
    t0 = time.perf_counter()

    for q in tqdm(queries, desc="Querying BM25 (rewrite → retrieve)"):
        original_query = str(q.get("query_text", ""))
        gold_citations: set[str] = set(q.get("ground_truth_citations", []))
        if not original_query.strip() or not gold_citations:
            continue

        rewritten_query = rewriter.rewrite(original_query)

        results = retriever.retrieve_text(rewritten_query, k=max_k)
        retrieved_ids = [c.doc_id for c in results]

        all_retrieved.append(retrieved_ids)
        all_relevant.append(gold_citations)
        rows.append({
            "query_id": q.get("query_id"),
            "original_query_text": original_query,
            "query_text": rewritten_query,
            "gold_citations": list(gold_citations),
            "retrieved_ids": retrieved_ids,
            "answer": None,
            "total_relevant_chunks": sum(doc_chunk_count.get(c, 0) for c in gold_citations),
        })

    elapsed = time.perf_counter() - t0
    log.info(f"Queried {len(rows)} examples in {elapsed:.1f}s")
    rewriter.log_usage_summary()

    # --- Compute metrics ---
    eval_result = evaluate_retrieval(
        experiment_id=cfg.experiment_id,
        retrieved_lists=all_retrieved,
        relevant_sets=all_relevant,
        k_values=k_values,
        metric_names=cfg.evaluation.metrics,
    )

    # --- Save results ---
    results_dir = Path(f"runs/{cfg.experiment_id}/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    (results_dir / "query_results.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )

    (results_dir / "metrics.json").write_text(
        json.dumps(
            {
                "experiment_id": eval_result.experiment_id,
                "num_queries": eval_result.num_queries,
                "scores": {m: dict(by_k) for m, by_k in eval_result.scores.items()},
                "judge_scores": eval_result.judge_scores,
            },
            indent=2,
        )
    )

    log.info(f"Results saved to {results_dir}")
    print(eval_result.summary())


if __name__ == "__main__":
    main()
