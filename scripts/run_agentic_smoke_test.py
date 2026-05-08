"""
Smoke test for AgenticRAGPipeline.

Loads the pre-built BM25 index referenced by the agentic config and runs the
agentic pipeline on a handful of queries with shortened iteration and search
limits, to verify the search/review/answer loop end-to-end.

Usage
-----
    python scripts/run_agentic_smoke_test.py
    python scripts/run_agentic_smoke_test.py --num-queries 3 --max-iterations 2

Requires: GOOGLE_API_KEY or GEMINI_API_KEY (Gemini agent + answer generation).
Requires: a pre-built BM25 index at runs/indexes/<index_id>/index.bm25.pkl
          (built by scripts/run_indexing.py with a Hybrid/BM25 retriever config
          whose dataset + chunker + embedder match the agentic config).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from benchmark_rag.components.retrievers.bm25_retriever import BM25Retriever
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.cost_logging import (
    DEFAULT_BENCHMARK_COST_CSV,
    append_cost_entry,
    sum_component_costs,
)
from benchmark_rag.pipeline.agentic_pipeline import AgenticRAGPipeline


DEFAULT_CONFIG = "configs/experiments/agentic_gemini_bm25_recursive_1024.yaml"


def main():
    parser = argparse.ArgumentParser(description="Small-scale smoke test for AgenticRAGPipeline.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Agentic experiment YAML (default: %(default)s).")
    parser.add_argument("--num-queries", type=int, default=2,
                        help="Number of queries to run from queries.json (default: 2).")
    parser.add_argument("--max-iterations", type=int, default=2,
                        help="Cap on agent search-review rounds (default: 2).")
    parser.add_argument("--max-k-per-search", type=int, default=5,
                        help="Max chunks per keyword_search call (default: 5).")
    parser.add_argument("--max-doc-chunks", type=int, default=5,
                        help="Max chunks loaded per saved citation (default: 5).")
    parser.add_argument("--model", default=None,
                        help="Override Gemini model from config (e.g. gemini-2.5-flash).")
    parser.add_argument("--query-text", default=None,
                        help="Run a single ad-hoc query instead of sampling from queries.json.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )

    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        sys.exit("ERROR: set GOOGLE_API_KEY or GEMINI_API_KEY (see .env.example).")

    cfg = ExperimentConfig.from_yaml(args.config)

    index_path = Path(cfg.indexing.output_dir) / "index"
    bm25_file = index_path.with_suffix(".bm25.pkl")
    if not bm25_file.exists():
        sys.exit(
            f"ERROR: BM25 index not found at {bm25_file}.\n"
            f"Build it first with:\n"
            f"  python scripts/run_indexing.py --config {args.config}"
        )

    print(f"Loading BM25 index from {bm25_file} ...")
    retriever = BM25Retriever()
    retriever.load_index(index_path)
    print(f"  loaded {len(retriever._chunks)} chunks "
          f"across {len({c.doc_id for c in retriever._chunks})} documents")

    model_name = args.model or cfg.agentic.model_name
    print(f"\nAgenticRAGPipeline config:")
    print(f"  model           = {model_name}")
    print(f"  max_iterations  = {args.max_iterations}")
    print(f"  max_k_per_search= {args.max_k_per_search}")
    print(f"  max_doc_chunks  = {args.max_doc_chunks}")

    pipeline = AgenticRAGPipeline(
        retriever=retriever,
        model_name=model_name,
        max_iterations=args.max_iterations,
        max_k_per_search=args.max_k_per_search,
        max_doc_chunks=args.max_doc_chunks,
    )

    if args.query_text:
        selected = [{"query_id": "ad-hoc", "query_text": args.query_text,
                     "ground_truth_citations": []}]
    else:
        queries_path = Path(cfg.evaluation.queries_path)
        if not queries_path.exists():
            sys.exit(f"ERROR: queries file not found at {queries_path}.")
        queries = json.loads(queries_path.read_text(encoding="utf-8"))
        selected = queries[: args.num_queries]

    print(f"\nRunning {len(selected)} query/queries ...\n")

    for i, q in enumerate(selected, 1):
        qtext = str(q.get("query_text", ""))
        province = q.get("province", "")
        if province:
            qtext = f"I am in {province}. {qtext}"
        gold = list(q.get("ground_truth_citations", []))
        print("=" * 72)
        print(f"[{i}/{len(selected)}] query_id={q.get('query_id')}")
        print(f"QUESTION: {qtext[:240]}{'...' if len(qtext) > 240 else ''}")
        print(f"GOLD    : {gold}")
        print()

        result = pipeline.query(qtext)

        meta = result.metadata or {}
        retrieved_ids = [c.doc_id for c in result.retrieved_chunks]
        hits = [c for c in gold if c in retrieved_ids]

        print(f"iterations     : {meta.get('iterations')}")
        print(f"searches_run   : {meta.get('searches_run')}")
        print(f"saved_docs     : {meta.get('saved_docs')}")
        print(f"retrieved chks : {len(result.retrieved_chunks)}")
        print(f"gold hits      : {hits} ({len(hits)}/{len(gold)})")
        if result.answer:
            print("ANSWER:")
            preview = result.answer.strip()
            print(preview[:800] + ("\n... [truncated]" if len(preview) > 800 else ""))
        print()

    print("=" * 72)
    pipeline.log_usage_summary()

    run_cost = sum_component_costs(pipeline)
    total, csv_path = append_cost_entry(
        DEFAULT_BENCHMARK_COST_CSV,
        experiment_id=f"{cfg.experiment_id}__smoke",
        cost_of_run_usd=run_cost,
    )
    print(f"\nCost logged to {csv_path}: "
          f"run=${run_cost:.6f} | total_so_far=${total:.6f}")
    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
