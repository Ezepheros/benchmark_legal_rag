"""
Compare retrieval performance for layperson vs. legal-expert query styles.

Query style is determined by query_id:
    (query_id // 2) % 2 == 0  →  layperson  (ids 0-1, 4-5, 8-9, …)
    (query_id // 2) % 2 == 1  →  expert     (ids 2-3, 6-7, 10-11, …)

Usage
-----
    # Single results file
    python scripts/analyze_query_styles.py \
        --results runs/gemini_recursive_1024_1k-docs/results/query_results.jsonl

    # All runs at once (auto-discovers every query_results.jsonl under runs/)
    python scripts/analyze_query_styles.py --all

    # Custom k values
    python scripts/analyze_query_styles.py \
        --results runs/gemini_recursive_1024_1k-docs/results/query_results.jsonl \
        --k 3 5 10 20 50
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmark_rag.evaluation.metrics import evaluate_retrieval, ALL_METRICS, EXCLUDED_QUERY_IDS

RUNS_DIR = Path(__file__).parent.parent / "runs"

STYLE_HATCH = {"layperson": "", "expert": "///"}


def query_style(query_id: int) -> str:
    return "layperson" if (query_id // 2) % 2 == 0 else "expert"


def load_results(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def analyze(rows: list[dict], k_values: list[int], metrics: list[str]) -> dict[str, dict]:
    """Split rows by query style and compute metrics for each group."""
    groups: dict[str, tuple[list, list]] = {"layperson": ([], []), "expert": ([], [])}

    for row in rows:
        qid = row.get("query_id")
        if qid is None:
            continue
        if qid in EXCLUDED_QUERY_IDS:
            continue
        style = query_style(int(qid))
        retrieved = row.get("retrieved_ids", [])
        relevant = set(row.get("gold_citations", []))
        if not retrieved or not relevant:
            continue
        groups[style][0].append(retrieved)
        groups[style][1].append(relevant)

    results = {}
    for style, (retrieved_lists, relevant_sets) in groups.items():
        if not retrieved_lists:
            continue
        result = evaluate_retrieval(
            experiment_id=style,
            retrieved_lists=retrieved_lists,
            relevant_sets=relevant_sets,
            k_values=k_values,
            metric_names=metrics,
        )
        results[style] = result
    return results


def print_comparison(experiment_id: str, results: dict, k_values: list[int]) -> None:
    styles = ["layperson", "expert"]
    present = [s for s in styles if s in results]
    if not present:
        print(f"  No results.")
        return

    print(f"\n{'='*70}")
    print(f"  {experiment_id}")
    print(f"{'='*70}")

    # Header
    n_queries = {s: results[s].num_queries for s in present}
    print(f"  {'Metric':<30}  " + "  ".join(f"{s} (n={n_queries[s]})" for s in present))
    print(f"  {'-'*30}  " + "  ".join("-" * 18 for _ in present))

    # Rows
    all_cols: list[tuple[str, int | None]] = []
    for metric, by_k in results[present[0]].scores.items():
        for k in sorted(by_k.keys()):
            all_cols.append((metric, k))

    for metric, k in all_cols:
        label = "mrr" if metric == "mrr" else f"{metric}@{k}"
        row_vals = []
        for s in present:
            val = results[s].scores.get(metric, {}).get(k, float("nan"))
            row_vals.append(f"{val:.4f}")
        # Highlight which style is better
        if len(row_vals) == 2:
            try:
                delta = float(row_vals[0]) - float(row_vals[1])
                arrow = " ▲" if delta > 0.001 else (" ▼" if delta < -0.001 else "  ")
                row_vals[0] += arrow
            except ValueError:
                pass
        print(f"  {label:<30}  " + "  ".join(f"{v:<18}" for v in row_vals))


def plot_all(
    all_results: dict[str, dict],  # {experiment_id: {style: EvaluationResult}}
    plot_dir: Path,
) -> None:
    """One chart per metric comparing all experiments.

    Color  = experiment
    Hatch  = query style  (solid = layperson, /// = expert)
    """
    plot_dir.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab10.colors
    styles = ["layperson", "expert"]
    experiment_ids = list(all_results.keys())

    # Collect all metrics present across any experiment
    all_metrics: list[str] = []
    for exp_results in all_results.values():
        for style_results in exp_results.values():
            for m in style_results.scores:
                if m not in all_metrics:
                    all_metrics.append(m)

    for metric in all_metrics:
        # Union of k values across all experiments for this metric
        k_set: set[int] = set()
        for exp_results in all_results.values():
            for style_results in exp_results.values():
                k_set.update(style_results.scores.get(metric, {}).keys())
        k_values = sorted(k_set)
        x_labels = ["mrr" if metric == "mrr" else str(k) for k in k_values]
        n_groups = len(k_values)

        # bars per group: one per (experiment × style)
        n_series = len(experiment_ids) * len(styles)
        bar_width = 0.8 / n_series

        fig, ax = plt.subplots(figsize=(7, 5))

        for exp_i, exp_id in enumerate(experiment_ids):
            for style_i, style in enumerate(styles):
                if style not in all_results[exp_id]:
                    continue
                result = all_results[exp_id][style]
                vals = [result.scores.get(metric, {}).get(k, 0.0) for k in k_values]

                bar_i = exp_i * len(styles) + style_i
                offsets = [xi + bar_i * bar_width for xi in range(n_groups)]
                color = colors[exp_i % len(colors)]
                hatch = STYLE_HATCH[style]

                ax.bar(
                    offsets, vals,
                    width=bar_width,
                    color=color,
                    hatch=hatch,
                    edgecolor="white" if not hatch else color,
                    alpha=0.85,
                    linewidth=0.5,
                )

        ax.set_title(f"{metric} — Layperson vs Expert", fontweight="bold")
        center_offset = bar_width * (n_series - 1) / 2
        ax.set_xticks([xi + center_offset for xi in range(n_groups)])
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("k" if metric != "mrr" else "")
        ax.set_ylabel("Score")

        # y-axis ceiling
        max_val = max(
            all_results[exp_id][style].scores.get(metric, {}).get(k, 0.0)
            for exp_id in experiment_ids
            for style in styles
            if style in all_results[exp_id]
            for k in k_values
        )
        ax.set_ylim(0, max(max_val * 1.15, 0.1))
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        # Two-part legend: color patches for experiments, hatch patches for style
        color_patches = [
            mpatches.Patch(color=colors[i % len(colors)], label=exp_id)
            for i, exp_id in enumerate(experiment_ids)
        ]
        hatch_patches = [
            mpatches.Patch(facecolor="white", hatch=STYLE_HATCH[s],
                           edgecolor="black", linewidth=1.0, label=s)
            for s in styles
        ]
        # Two separate legends outside the axes on the right
        exp_legend = ax.legend(
            handles=color_patches,
            fontsize=7,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            borderaxespad=0,
            title="experiment",
            title_fontsize=7,
            handlelength=1.2,
            handleheight=0.8,
        )
        ax.add_artist(exp_legend)
        style_legend = ax.legend(
            handles=hatch_patches,
            fontsize=7,
            loc="lower left",
            bbox_to_anchor=(1.02, 0),
            borderaxespad=0,
            title="query style",
            title_fontsize=7,
            handlelength=1.2,
            handleheight=0.8,
        )

        out_path = plot_dir / f"{metric}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight",
                    bbox_extra_artists=(exp_legend, style_legend))
        plt.close(fig)
        print(f"  Chart saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare retrieval metrics for layperson vs. expert queries."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--results", nargs="+",
        help="Path(s) to query_results.jsonl file(s)",
    )
    group.add_argument(
        "--all", action="store_true",
        help="Auto-discover all query_results.jsonl under runs/ (excludes old_runs/)",
    )
    parser.add_argument(
        "--k", nargs="+", type=int, default=None, metavar="K",
        help="k cutoff values (default: infer from data)",
    )
    parser.add_argument(
        "--metrics", nargs="+", default=None, choices=ALL_METRICS, metavar="METRIC",
        help=f"Metrics to compute (default: all). Choices: {ALL_METRICS}",
    )
    parser.add_argument(
        "--output", default=None,
        help="Save results to a JSON file",
    )
    parser.add_argument(
        "--plot-dir", default=None, metavar="DIR",
        help="Directory to save chart images (default: runs/charts/query_styles/)",
    )
    args = parser.parse_args()

    # Collect files
    if args.all:
        files = [
            p for p in sorted(RUNS_DIR.glob("*/results/query_results.jsonl"))
            if "old_runs" not in p.parts
        ]
        if not files:
            sys.exit(f"No query_results.jsonl files found under {RUNS_DIR}")
    else:
        files = [Path(p) for p in args.results]

    all_output = {}
    all_style_results: dict[str, dict] = {}

    for path in files:
        if not path.exists():
            print(f"WARNING: {path} not found, skipping.")
            continue

        rows = load_results(path)
        experiment_id = path.parts[-3]  # runs/<exp_id>/results/query_results.jsonl

        # Infer k values
        if args.k:
            k_values = sorted(args.k)
        else:
            max_retrieved = max((len(r.get("retrieved_ids", [])) for r in rows), default=50)
            k_values = sorted(k for k in [3, 5, 10, 20, 50, 100] if k <= max_retrieved)

        metrics = args.metrics or ALL_METRICS

        results = analyze(rows, k_values, metrics)
        print_comparison(experiment_id, results, k_values)

        all_style_results[experiment_id] = results
        all_output[experiment_id] = {
            style: {
                "num_queries": r.num_queries,
                "scores": {m: dict(by_k) for m, by_k in r.scores.items()},
            }
            for style, r in results.items()
        }

    # Plot all experiments together, one chart per metric
    if all_style_results:
        plot_dir = Path(args.plot_dir) if args.plot_dir else RUNS_DIR / "charts" / "query_styles"
        plot_all(all_style_results, plot_dir)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_output, indent=2))
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
