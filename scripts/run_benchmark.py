"""
Run retrieval (and optionally generation + judging) evaluation for one experiment.

Usage
-----
    # Retrieval-only
    python scripts/run_benchmark.py --config configs/experiments/qwen_recursive_1024.yaml

    # With generation + LLM judge
    python scripts/run_benchmark.py --config configs/experiments/qwen_recursive_1024.yaml \
        --generate --judge

Expects an already-built index at runs/<experiment_id>/index/ (run run_indexing.py first).
Writes results to runs/<experiment_id>/results/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env before anything else so API keys are available to all components.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass  # python-dotenv not installed; fall back to shell environment

import pandas as pd
from tqdm import tqdm

from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.evaluation.metrics import evaluate_retrieval, EvaluationResult
from benchmark_rag.logging import setup_experiment_logging, get_logger
from benchmark_rag.pipeline.rag_pipeline import RAGPipeline


def _validate_api_keys(cfg: ExperimentConfig, args) -> None:
    """Fail fast if required API keys are missing for the configured components."""
    import os
    embedder_type = cfg.embedder.type.lower()
    generator_type = (cfg.generator.type.lower() if cfg.generator else "")

    if "kanon2" in embedder_type and not os.environ.get("ISAACUS_API_KEY"):
        sys.exit("ERROR: ISAACUS_API_KEY is not set. Required for Kanon2Embedder.")

    needs_google = "gemini" in embedder_type or "gemini" in generator_type or args.iterretgen
    if needs_google and not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        sys.exit("ERROR: GOOGLE_API_KEY or GEMINI_API_KEY is not set. Required for Gemini components.")


def load_queries(queries_path: str) -> list[dict]:
    """
    Load queries produced by build_test_dataset.py.

    Supports queries.json (primary format) and legacy .pkl / .parquet files.

    queries.json schema (one item per query):
        query_id, query_text, user_answer, custom_instruction,
        batch_id, ground_truth_citations (list[str])
    """
    p = Path(queries_path)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    elif p.suffix == ".pkl":
        import pickle
        with open(p, "rb") as f:
            data = pickle.load(f)
        # Normalise legacy pkl DataFrames to list-of-dicts
        if hasattr(data, "to_dict"):
            return data.to_dict(orient="records")
        return data
    elif p.suffix == ".parquet":
        return pd.read_parquet(p).to_dict(orient="records")
    raise ValueError(f"Unsupported query file format: {p.suffix}")


def _log_run_context(log, cfg: ExperimentConfig, config_source: str, args) -> None:
    """Log the full config and all relevant file paths for this benchmark run."""
    run_dir = Path(f"runs/{cfg.experiment_id}")
    index_dir = Path(cfg.indexing.output_dir)
    results_dir = run_dir / "results"

    log.info("=" * 60)
    log.info(f"EXPERIMENT : {cfg.experiment_id}")
    log.info(f"DESCRIPTION: {cfg.description}")
    log.info(f"SEED       : {cfg.seed}")
    log.info(f"CONFIG SRC : {config_source}")
    log.info(f"FLAGS      : generate={args.generate}  judge={args.judge}  iterretgen={args.iterretgen}")
    log.info(f"INDEX ID   : {cfg.index_id}")

    # --- Input paths ---
    log.info("--- inputs ---")
    log.info(f"  dataset.path    : {cfg.dataset.path}")
    log.info(f"  dataset.max_docs: {cfg.dataset.max_docs}")
    log.info(f"  queries_path    : {cfg.evaluation.queries_path}")
    log.info(f"  index dir       : {index_dir}")
    log.info(f"    index.faiss   : {index_dir}/index.faiss")
    log.info(f"    index.chunks  : {index_dir}/index.chunks.pkl")

    # --- Component config ---
    log.info("--- components ---")
    log.info(f"  embedder : {cfg.embedder.type}")
    log.info(f"    model_name  : {cfg.embedder.model_name}")
    log.info(f"    device      : {cfg.embedder.model_extra.get('device', 'N/A')}")
    log.info(f"  chunker  : {cfg.chunker.type}")
    log.info(f"    max_chunk_chars : {cfg.chunker.max_chunk_chars}")
    log.info(f"    overlap_chars   : {cfg.chunker.overlap_chars}")
    log.info(f"  retriever: {cfg.retriever.type}")
    log.info(f"    metric      : {cfg.retriever.model_extra.get('metric', 'cosine')}")
    if cfg.generator:
        log.info(f"  generator: {cfg.generator.type}")
        log.info(f"    model_name  : {cfg.generator.model_name}")

    # --- Evaluation config ---
    log.info("--- evaluation ---")
    log.info(f"  k_values : {cfg.evaluation.k_values}")
    log.info(f"  metrics  : {cfg.evaluation.metrics}")

    # --- Output paths ---
    log.info("--- outputs ---")
    log.info(f"  results dir      : {results_dir}")
    log.info(f"    metrics        : {results_dir}/metrics.json")
    log.info(f"    query results  : {results_dir}/query_results.jsonl")
    log.info(f"  log dir          : {cfg.logging.log_dir}")
    log.info(f"    human log      : {cfg.logging.log_dir}/{cfg.experiment_id}.log")
    log.info(f"    json log       : {cfg.logging.log_dir}/{cfg.experiment_id}.jsonl")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a RAG experiment.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--generate", action="store_true", help="Run answer generation")
    parser.add_argument("--judge", action="store_true", help="Run LLM judge on generated answers")
    parser.add_argument("--iterretgen", action="store_true", help="Use IterRetGen pipeline (iterative retrieval augmented by intermediate generation)")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config)
    _validate_api_keys(cfg, args)

    if cfg.evaluation is None:
        print("No evaluation config found — nothing to do.")
        sys.exit(1)

    setup_experiment_logging(
        experiment_id=cfg.experiment_id,
        log_dir=cfg.logging.log_dir,
        level=cfg.logging.level,
        resource_monitor_interval=0,  # no background monitor during eval
    )
    log = get_logger(__name__)
    _log_run_context(log, cfg, config_source=args.config, args=args)

    results_dir = Path(f"runs/{cfg.experiment_id}/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Load queries ---
    # queries.json schema: query_id, query_text, user_answer, ground_truth_citations (list)
    queries = load_queries(cfg.evaluation.queries_path)
    log.info(f"Loaded {len(queries)} queries from {cfg.evaluation.queries_path}")

    # --- Build pipeline ---
    eval_cfg = cfg.evaluation
    if args.iterretgen:
        from benchmark_rag.pipeline.iterretgen_pipeline import IterRetGenPipeline
        pipeline = IterRetGenPipeline.from_config(cfg)
    elif args.generate and cfg.generator is not None:
        pipeline = RAGPipeline.from_config(cfg)
    else:
        # Retrieval-only: build without generator
        from benchmark_rag.components.retrievers.faiss_retriever import FaissRetriever
        from benchmark_rag.registry import build_from_component_config

        embedder = build_from_component_config(cfg.embedder.to_build_dict())
        retriever = FaissRetriever(metric=cfg.retriever.model_extra.get("metric", "cosine"))
        retriever.load_index(Path(cfg.indexing.output_dir) / "index")
        pipeline = RAGPipeline(
            embedder=embedder,
            retriever=retriever,
            k=max(eval_cfg.k_values),
        )

    # --- Run queries ---
    # ground_truth_citations is a list[str] — a query may have multiple relevant docs
    all_retrieved: list[list[str]] = []
    all_relevant: list[set[str]] = []
    rows = []
    t0 = time.perf_counter()

    for q in tqdm(queries, desc="Querying"):
        query_text = str(q.get("query_text", ""))
        gold_citations: set[str] = set(q.get("ground_truth_citations", []))
        if not query_text.strip() or not gold_citations:
            continue

        result = pipeline.query(query_text, k=max(eval_cfg.k_values))
        retrieved_ids = [c.doc_id for c in result.retrieved_chunks]

        all_retrieved.append(retrieved_ids)
        all_relevant.append(gold_citations)

        record = {
            "query_id": q.get("query_id"),
            "query_text": query_text,
            "gold_citations": list(gold_citations),
            "retrieved_ids": retrieved_ids,
            "answer": result.answer,
        }
        rows.append(record)

    elapsed = time.perf_counter() - t0
    log.info(f"Queried {len(rows)} examples in {elapsed:.1f}s")
    if hasattr(pipeline, "log_usage_summary"):
        pipeline.log_usage_summary()
    elif pipeline.generator is not None and hasattr(pipeline.generator, "log_usage_summary"):
        pipeline.generator.log_usage_summary()

    # --- Compute metrics ---
    eval_result: EvaluationResult = evaluate_retrieval(
        experiment_id=cfg.experiment_id,
        retrieved_lists=all_retrieved,
        relevant_sets=all_relevant,
        k_values=eval_cfg.k_values,
        metric_names=eval_cfg.metrics,
    )

    # --- Optional: LLM judge ---
    if args.judge and cfg.generator is not None:
        from benchmark_rag.components.generators.gemini import GeminiJudge

        # Build a lookup: query_id → user_answer (the annotator's reference answer)
        ref_by_id = {str(q.get("query_id")): str(q.get("user_answer", "")) for q in queries}

        judge = GeminiJudge()
        judge_scores: dict[str, list[float]] = {}
        for rec in tqdm(rows, desc="Judging"):
            if not rec.get("answer"):
                continue
            ref = ref_by_id.get(str(rec.get("query_id")), "")
            scores = judge.judge(rec["query_text"], rec["answer"], ref)
            for k, v in scores.items():
                if k != "rationale":
                    judge_scores.setdefault(k, []).append(float(v))
        eval_result.judge_scores = {k: sum(v) / len(v) for k, v in judge_scores.items() if v}
        judge.log_usage_summary()

    # --- Save results ---
    results_file = results_dir / "query_results.jsonl"
    with open(results_file, "w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")

    metrics_file = results_dir / "metrics.json"
    metrics_file.write_text(
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
