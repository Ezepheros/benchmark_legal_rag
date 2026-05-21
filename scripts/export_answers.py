"""
Export all generated answers as a fully self-contained JSONL file.

Embeds chunk text, full document text, and ground truth answers so the
export can be analyzed without access to the index or parquet files.

Output: 6 files (condition x generator) in runs/final_eval/export/

Usage
-----
    python scripts/export_answers.py
    python scripts/export_answers.py --no-doc-text   # skip full document text (much smaller)
    python scripts/export_answers.py --output /tmp/export/
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
EVAL_BASE = PROJECT_ROOT / "runs" / "final_eval"
DATASET_PATH = PROJECT_ROOT / "data" / "test_dataset" / "test_dataset.parquet"
QUERIES_PATH = PROJECT_ROOT / "data" / "test_dataset" / "queries.json"

INDEX_DIRS = [
    "qwen3_embedding_8b__recursive8192__ad014d42de",
    "gemini_embedding_2__recursive4096__b280f9ae16",
    "embeddinggemma_300m__recursive8192__0ef38ebc1e",
]

CONDITIONS = ["oracle", "pipeline"]
GENERATORS = ["gemini", "gemma", "qwen"]


def main():
    parser = argparse.ArgumentParser(description="Export generated answers as self-contained JSONL.")
    parser.add_argument("--output", default=str(EVAL_BASE / "export"),
                        help="Output directory (default: runs/final_eval/export/)")
    parser.add_argument("--no-doc-text", action="store_true",
                        help="Exclude full document text (much smaller file)")
    args = parser.parse_args()

    # Load documents
    doc_columns = ["citation", "name", "court", "url"]
    if not args.no_doc_text:
        doc_columns.append("text")
    doc_df = pd.read_parquet(DATASET_PATH, columns=doc_columns)
    doc_meta = {row["citation"]: row.to_dict() for _, row in doc_df.iterrows()}
    print(f"Loaded {len(doc_meta)} documents")

    # Load chunk texts from indexes
    chunk_text_lookup: dict[tuple, str] = {}
    for idx_dir in INDEX_DIRS:
        pkl = PROJECT_ROOT / "runs" / "indexes" / idx_dir / "index.chunks.pkl"
        if pkl.exists():
            with open(pkl, "rb") as f:
                chunks = pickle.load(f)
            for c in chunks:
                chunk_text_lookup[(c.doc_id, c.chunk_idx)] = c.text
            print(f"  {idx_dir}: {len(chunks)} chunks")
    print(f"Total chunk texts: {len(chunk_text_lookup)}")

    # Load ground truth
    queries = json.loads(QUERIES_PATH.read_text())
    gt_lookup = {q["query_id"]: q for q in queries}

    # Build and write per condition x generator
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'Condition':<10} {'Generator':<10} {'Rows':>5} {'Answered':>8} {'Size':>10}")
    print("-" * 50)

    for cond in CONDITIONS:
        for gen in GENERATORS:
            path = EVAL_BASE / cond / f"{gen}_answers.jsonl"
            if not path.exists():
                print(f"{cond:<10} {gen:<10}  SKIP (not found)")
                continue

            rows = []
            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)

                    enriched_chunks = []
                    for cd in r.get("retrieved_chunk_details", []):
                        chunk = dict(cd)
                        chunk["text"] = chunk_text_lookup.get(
                            (cd["doc_id"], cd.get("chunk_idx", 0)), ""
                        )
                        enriched_chunks.append(chunk)

                    unique_docs = list(dict.fromkeys(r.get("retrieved_ids", [])))
                    doc_details = []
                    for did in unique_docs:
                        meta = dict(doc_meta.get(did, {}))
                        meta.setdefault("citation", did)
                        doc_details.append(meta)

                    gt = gt_lookup.get(r["query_id"], {})

                    rows.append({
                        "query_id": r["query_id"],
                        "query_text": r.get("query_text", ""),
                        "condition": cond,
                        "generator": gen,
                        "gold_citations": r.get("gold_citations", []),
                        "ground_truth_answer": gt.get("user_answer", ""),
                        "generated_answer": r.get("generated_answer"),
                        "retrieval_method": r.get("retrieval_method", ""),
                        "retrieved_chunks": enriched_chunks,
                        "retrieved_documents": doc_details,
                        "context_meta": r.get("context_meta", {}),
                        "usage": r.get("usage", {}),
                    })

            out_path = out_dir / f"{cond}_{gen}_answers.jsonl"
            with open(out_path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            answered = sum(1 for r in rows if r.get("generated_answer"))
            size_mb = os.path.getsize(out_path) / 1e6
            print(f"{cond:<10} {gen:<10} {len(rows):>5} {answered:>8} {size_mb:>8.1f} MB")


if __name__ == "__main__":
    main()
