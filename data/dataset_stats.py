"""
Comprehensive statistics for the test_dataset (parquet + queries.json).
Run from anywhere — uses absolute paths relative to this file.
"""

import json
import pathlib

import pandas as pd

DATA_DIR = pathlib.Path(__file__).parent / "test_dataset"


def sep(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def subsep(title: str) -> None:
    print(f"\n--- {title} ---")


# ─── Load ────────────────────────────────────────────────────────────────────

df = pd.read_parquet(DATA_DIR / "test_dataset.parquet")
queries: list[dict] = json.loads((DATA_DIR / "queries.json").read_text())

# Parse JSON-encoded list columns
for col in ("ground_truth_query_ids", "ground_truth_query_texts", "snippets"):
    df[col] = df[col].apply(lambda v: json.loads(v) if isinstance(v, str) else (v or []))

df["text_len"] = df["text"].str.len()
df["document_date"] = pd.to_datetime(df["document_date"], utc=True, errors="coerce")

# ─── DOCUMENTS ───────────────────────────────────────────────────────────────

sep("DOCUMENT CORPUS")

print(f"Total documents : {len(df)}")
print(f"Columns         : {', '.join(df.columns.tolist())}")

subsep("By source")
print(df["source"].value_counts().to_string())

subsep("By court")
print(df["court"].value_counts().to_string())

subsep("Ground truth vs background")
gt = df["is_ground_truth"]
print(f"  Ground-truth documents : {gt.sum()}")
print(f"  Background documents   : {(~gt).sum()}")

subsep("Document text length (characters)")
stats = df["text_len"].describe(percentiles=[0.25, 0.5, 0.75, 0.90, 0.95])
for k, v in stats.items():
    print(f"  {k:<10}: {v:,.0f}")

subsep("Document text length by source")
print(df.groupby("source")["text_len"].agg(["min", "mean", "median", "max"])
      .rename(columns={"min": "min", "mean": "mean", "median": "median", "max": "max"})
      .map(lambda x: f"{x:,.0f}")
      .to_string())

subsep("Document dates")
valid_dates = df["document_date"].dropna()
if len(valid_dates):
    print(f"  Earliest : {valid_dates.min().date()}")
    print(f"  Latest   : {valid_dates.max().date()}")
    print(f"  Missing  : {df['document_date'].isna().sum()}")
    by_year = valid_dates.dt.year.value_counts().sort_index()
    print(f"\n  Documents per year:\n{by_year.to_string()}")

subsep("Citation coverage")
print(f"  Has citation2    : {df['citation2'].notna().sum()} / {len(df)}")
print(f"  Has upstream_license : {df['upstream_license'].notna().sum()} / {len(df)}")

# ─── GROUND TRUTH DOCUMENTS ──────────────────────────────────────────────────

sep("GROUND TRUTH DOCUMENTS")

gt_df = df[df["is_ground_truth"]].copy()
print(f"Total ground-truth docs : {len(gt_df)}")

subsep("Queries covered per ground-truth doc")
n_queries_per_doc = gt_df["ground_truth_query_ids"].apply(len)
stats = n_queries_per_doc.describe(percentiles=[0.5])
for k, v in stats.items():
    print(f"  {k:<10}: {v:.2f}")

subsep("Snippets per ground-truth doc")
n_snippets = gt_df["snippets"].apply(len)
stats = n_snippets.describe(percentiles=[0.5])
for k, v in stats.items():
    print(f"  {k:<10}: {v:.2f}")

subsep("Ground-truth docs by court")
print(gt_df["court"].value_counts().to_string())

# ─── QUERIES ─────────────────────────────────────────────────────────────────

sep("QUERIES")

print(f"Total queries : {len(queries)}")

query_ids       = [q["query_id"] for q in queries]
query_texts     = [q["query_text"] for q in queries]
gt_citations    = [q["ground_truth_citations"] for q in queries]
has_answer      = sum(1 for q in queries if q.get("user_answer"))
has_instruction = sum(1 for q in queries if q.get("custom_instruction"))
batch_ids       = [q.get("batch_id") for q in queries]

print(f"  Has user_answer        : {has_answer}")
print(f"  Has custom_instruction : {has_instruction}")

subsep("By batch_id")
batch_series = pd.Series(batch_ids)
print(batch_series.value_counts().sort_index().to_string())

subsep("Query text length (characters)")
q_lens = pd.Series([len(t) for t in query_texts])
stats = q_lens.describe(percentiles=[0.25, 0.5, 0.75])
for k, v in stats.items():
    print(f"  {k:<10}: {v:,.0f}")

subsep("Ground-truth citations per query")
n_gt = pd.Series([len(c) for c in gt_citations])
stats = n_gt.describe(percentiles=[0.5])
for k, v in stats.items():
    print(f"  {k:<10}: {v:.2f}")
dist = n_gt.value_counts().sort_index()
print(f"\n  Distribution (# citations → # queries):")
for k, v in dist.items():
    print(f"    {k} citations : {v} queries")

# ─── CROSS-REFERENCE ─────────────────────────────────────────────────────────

sep("CORPUS ↔ QUERY CROSS-REFERENCE")

all_gt_citations = {c for q in queries for c in q["ground_truth_citations"]}
corpus_citations = set(df["citation"].dropna())

found     = all_gt_citations & corpus_citations
not_found = all_gt_citations - corpus_citations

print(f"  Unique ground-truth citations across all queries : {len(all_gt_citations)}")
print(f"  Found in corpus                                  : {len(found)}")
print(f"  Missing from corpus                              : {len(not_found)}")
if not_found:
    for c in sorted(not_found):
        print(f"    - {c}")
