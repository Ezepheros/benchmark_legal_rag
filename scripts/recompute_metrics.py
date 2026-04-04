"""
Recompute retrieval metrics from a saved query_results.jsonl file.

No re-embedding or re-retrieval is done — the retrieved_ids and gold_citations
already stored in the file are used directly.  This lets you add new metrics,
change k_values, or fix bugs without re-running the full pipeline.

Usage
-----
    # Use defaults from the original config
    python scripts/recompute_metrics.py \
        --results runs/gemini_recursive_1024_1k-docs/results/query_results.jsonl

    # Override k values and metrics
    python scripts/recompute_metrics.py \
        --results runs/gemini_recursive_1024_1k-docs/results/query_results.jsonl \
        --k 3 5 10 20 50 \
        --metrics recall_at_k doc_recall_at_k hit_at_k mrr ndcg_at_k

    # Write to a custom output file instead of overwriting metrics.json
    python scripts/recompute_metrics.py \
        --results runs/gemini_recursive_1024_1k-docs/results/query_results.jsonl \
        --output runs/gemini_recursive_1024_1k-docs/results/metrics_recomputed.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_rag.evaluation.metrics import evaluate_retrieval

ALL_METRICS = ["recall_at_k", "doc_recall_at_k", "precision_at_k", "hit_at_k", "mrr", "ndcg_at_k"]


def main():
    parser = argparse.ArgumentParser(
        description="Recompute retrieval metrics from a saved query_results.jsonl."
    )
    parser.add_argument(
        "--results", required=True,
        help="Path to query_results.jsonl produced by run_benchmark.py",
    )
    parser.add_argument(
        "--k", nargs="+", type=int, default=None,
        metavar="K",
        help="k cutoff values (default: infer from max retrieved_ids length)",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=ALL_METRICS,
        choices=ALL_METRICS,
        metavar="METRIC",
        help=f"Metrics to compute. Choices: {ALL_METRICS}. Default: all.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path. Default: metrics.json next to the results file.",
    )
    parser.add_argument(
        "--experiment-id", default=None,
        help="Experiment ID label for the output. Default: inferred from results path.",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        sys.exit(f"ERROR: results file not found: {results_path}")

    # --- Load query results ---
    rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    if not rows:
        sys.exit("ERROR: results file is empty.")

    retrieved_lists = [r["retrieved_ids"] for r in rows]
    relevant_sets   = [set(r["gold_citations"]) for r in rows]

    # --- Infer k values if not specified ---
    if args.k:
        k_values = sorted(args.k)
    else:
        max_retrieved = max(len(r) for r in retrieved_lists)
        k_values = sorted(k for k in [3, 5, 10, 20, 50, 100] if k <= max_retrieved)
        if not k_values:
            k_values = [max_retrieved]
        print(f"Inferred k_values from data: {k_values}")

    # --- Experiment ID ---
    experiment_id = args.experiment_id or results_path.parts[-3]  # runs/<exp_id>/results/...

    # --- Compute ---
    result = evaluate_retrieval(
        experiment_id=experiment_id,
        retrieved_lists=retrieved_lists,
        relevant_sets=relevant_sets,
        k_values=k_values,
        metric_names=args.metrics,
    )

    print(result.summary())

    # --- Save ---
    output_path = Path(args.output) if args.output else results_path.parent / "metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        {
            "experiment_id": result.experiment_id,
            "num_queries": result.num_queries,
            "scores": {m: dict(by_k) for m, by_k in result.scores.items()},
            "judge_scores": result.judge_scores,
        },
        indent=2,
    ))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
