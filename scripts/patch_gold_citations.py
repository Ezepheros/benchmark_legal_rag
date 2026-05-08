"""
Patch gold_citations in all query_results.jsonl files to match dedup changes.

Hardcoded remaps/drops from build_test_dataset.py text dedup:
  - "SY 2008, c 1"      -> "SY 2013, c 16"            (text duplicate)
  - "1994 CANLII 117"    -> "RJR-MacDonald Inc v Canada (Attorney General), [1994] SCJ No 17"  (text duplicate, different citation style)
  - "2020 ONSC 8035"     -> REMOVE                     (empty text, no survivor)
  - "Employment Insurance Act, SC 1996, c 23, <https://canliica/t/56bzf> retrieved on 2026-04-10"
                          -> REMOVE                     (empty text, no survivor)

Usage
-----
    python scripts/patch_gold_citations.py --dry-run          # preview
    python scripts/patch_gold_citations.py --filter bm25      # patch BM25 only
    python scripts/patch_gold_citations.py --recompute        # patch all + recompute metrics
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RUNS_DIR = Path(__file__).parent.parent / "runs"

REMAP = {
    "SY 2008, c 1": "SY 2013, c 16",
    "1994 CANLII 117": "RJR-MacDonald Inc v Canada (Attorney General), [1994] SCJ No 17",
}

DROP = {
    "2020 ONSC 8035",
    "Employment Insurance Act, SC 1996, c 23, <https://canliica/t/56bzf> retrieved on 2026-04-10",
}


def patch_citations(old: list[str]) -> list[str]:
    seen = set()
    new = []
    for c in old:
        if c in DROP:
            continue
        c = REMAP.get(c, c)
        if c not in seen:
            seen.add(c)
            new.append(c)
    return new


def patch_file(path: Path, dry_run: bool) -> tuple[int, int]:
    lines = path.read_text().splitlines()
    patched = []
    n_queries = 0
    n_cits = 0
    for line in lines:
        if not line.strip():
            patched.append(line)
            continue
        row = json.loads(line)
        old = row.get("gold_citations", [])
        new = patch_citations(old)
        if old != new:
            n_queries += 1
            n_cits += len(set(old) - set(new)) + len(set(new) - set(old))
            row["gold_citations"] = new
        patched.append(json.dumps(row))
    if not dry_run and n_queries > 0:
        path.write_text("\n".join(patched) + "\n")
    return n_queries, n_cits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--filter", default=None, help="Only patch experiments matching this substring")
    parser.add_argument("--recompute", action="store_true", help="Recompute metrics.json after patching")
    args = parser.parse_args()

    files = sorted(RUNS_DIR.glob("*/results/query_results.jsonl"))
    if args.filter:
        files = [f for f in files if args.filter in str(f)]
    if not files:
        sys.exit("No query_results.jsonl files found.")

    print(f"Remap: {REMAP}")
    print(f"Drop:  {DROP}")
    print(f"{'DRY RUN — ' if args.dry_run else ''}Scanning {len(files)} experiment(s)\n")

    changed = 0
    for f in files:
        exp = f.parts[-3]
        nq, nc = patch_file(f, args.dry_run)
        if nq:
            changed += 1
            print(f"  {exp}: {nq} queries patched ({nc} citation changes)")
        else:
            print(f"  {exp}: ok")

    print(f"\n{'Would patch' if args.dry_run else 'Patched'} {changed}/{len(files)} files.")

    if args.recompute and not args.dry_run and changed > 0:
        print("\nRecomputing metrics...")
        from benchmark_rag.evaluation.metrics import evaluate_retrieval, EXCLUDED_QUERY_IDS
        ALL_METRICS = ["recall_at_k", "doc_recall_at_k", "precision_at_k", "hit_at_k", "mrr", "ndcg_at_k"]

        for f in files:
            exp = f.parts[-3]
            rows = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
            rows = [r for r in rows
                    if r.get("query_id") not in EXCLUDED_QUERY_IDS
                    and r.get("gold_citations")]
            if not rows:
                continue
            retrieved = [r["retrieved_ids"] for r in rows]
            relevant = [set(r["gold_citations"]) for r in rows]
            max_k = max(len(r) for r in retrieved)
            k_values = sorted(k for k in [3, 5, 10, 20, 50] if k <= max_k) or [max_k]
            result = evaluate_retrieval(exp, retrieved, relevant, k_values, ALL_METRICS)
            out = f.parent / "metrics.json"
            out.write_text(json.dumps({
                "experiment_id": result.experiment_id,
                "num_queries": result.num_queries,
                "scores": {m: dict(by_k) for m, by_k in result.scores.items()},
                "judge_scores": result.judge_scores,
            }, indent=2))
            print(f"  {exp}: metrics recomputed")


if __name__ == "__main__":
    main()
