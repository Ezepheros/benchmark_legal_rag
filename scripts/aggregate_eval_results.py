"""
Aggregate final evaluation results across conditions and generators.

Detects missing runs, computes aggregate metrics, and exports tables for the paper.

Usage
-----
    # Check what's complete / missing:
    python scripts/aggregate_eval_results.py --check-missing

    # Produce aggregates:
    python scripts/aggregate_eval_results.py

    # Exclude specific queries and recompute:
    python scripts/aggregate_eval_results.py --exclude-query-ids 933,1068,1069
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RESULTS_BASE = Path(__file__).parent.parent / "runs" / "final_eval"
CONDITIONS = ["oracle", "pipeline"]
GENERATORS = ["gemini", "qwen", "gemma"]


def check_missing() -> None:
    print("=" * 70)
    print("FINAL EVAL STATUS")
    print("=" * 70)

    for condition in CONDITIONS:
        cond_dir = RESULTS_BASE / condition
        print(f"\n  {condition.upper()}:")

        for gen in GENERATORS:
            answers_path = cond_dir / f"{gen}_answers.jsonl"
            if not answers_path.exists():
                print(f"    {gen:10s} answers: MISSING")
                continue
            with open(answers_path) as f:
                rows = [json.loads(l) for l in f if l.strip()]
            n_answered = sum(1 for r in rows if r.get("generated_answer"))
            n_total = len(rows)
            print(f"    {gen:10s} answers: {n_answered}/{n_total}")

        atomic_path = cond_dir / "atomic_facts.jsonl"
        if atomic_path.exists():
            with open(atomic_path) as f:
                n = sum(1 for l in f if l.strip())
            print(f"    {'':10s} atomic_facts: {n} decompositions")
        else:
            print(f"    {'':10s} atomic_facts: MISSING")

        judge_path = cond_dir / "judge_results.jsonl"
        if judge_path.exists():
            with open(judge_path) as f:
                rows = [json.loads(l) for l in f if l.strip()]
            by_gen = {}
            for r in rows:
                by_gen.setdefault(r["generator"], []).append(r)
            for gen in GENERATORS:
                n = len(by_gen.get(gen, []))
                print(f"    {gen:10s} judged: {n}")
        else:
            print(f"    {'':10s} judge_results: MISSING")

        batch_dir = cond_dir / "batch"
        jobs_path = batch_dir / "jobs.json"
        if jobs_path.exists():
            jobs = json.loads(jobs_path.read_text())
            for j in jobs:
                print(f"    {'':10s} batch job: {j['file']} (submitted {j['submitted'][:10]})")
        else:
            print(f"    {'':10s} batch jobs: NOT SUBMITTED")

    print(f"\n{'='*70}")


def aggregate(exclude_ids: set[int] | None = None) -> None:
    all_rows: list[dict] = []

    for condition in CONDITIONS:
        judge_path = RESULTS_BASE / condition / "judge_results.jsonl"
        if not judge_path.exists():
            continue
        with open(judge_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if exclude_ids and rec["query_id"] in exclude_ids:
                    continue
                all_rows.append(rec)

    if not all_rows:
        print("No judge results found. Run the evaluation pipeline first.")
        return

    table: list[dict] = []
    for condition in CONDITIONS:
        for gen in GENERATORS:
            subset = [r for r in all_rows if r["condition"] == condition and r["generator"] == gen]
            if not subset:
                continue
            p_vals = [r["precision"] for r in subset]
            r_vals = [r["recall"] for r in subset]
            f1_vals = [r["f1"] for r in subset]
            avg_p = sum(p_vals) / len(p_vals)
            avg_r = sum(r_vals) / len(r_vals)
            avg_f1 = sum(f1_vals) / len(f1_vals)
            table.append({
                "condition": condition,
                "generator": gen,
                "num_queries": len(subset),
                "precision": round(avg_p, 4),
                "recall": round(avg_r, 4),
                "f1": round(avg_f1, 4),
            })

    # Print table
    print(f"\n{'Condition':<12} {'Generator':<10} {'N':>5} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 62)
    for row in table:
        print(f"{row['condition']:<12} {row['generator']:<10} {row['num_queries']:>5} "
              f"{row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1']:>10.4f}")
    if exclude_ids:
        print(f"\n(Excluded query IDs: {sorted(exclude_ids)})")

    # Save JSON
    summary = {
        "table": table,
        "excluded_query_ids": sorted(exclude_ids) if exclude_ids else [],
        "total_judged": len(all_rows),
    }
    summary_path = RESULTS_BASE / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {summary_path}")

    # Save CSV
    csv_path = RESULTS_BASE / "summary_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "generator", "num_queries", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(table)
    print(f"Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate final evaluation results.")
    parser.add_argument("--check-missing", action="store_true", help="Show status of all runs")
    parser.add_argument("--exclude-query-ids", type=str, default="",
                        help="Comma-separated query IDs to exclude from aggregation")
    args = parser.parse_args()

    if args.check_missing:
        check_missing()
        return

    exclude_ids = set()
    if args.exclude_query_ids:
        exclude_ids = {int(x.strip()) for x in args.exclude_query_ids.split(",") if x.strip()}

    aggregate(exclude_ids)


if __name__ == "__main__":
    main()
