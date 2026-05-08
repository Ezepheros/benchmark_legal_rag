"""
Combine retrieval and answer results into a single JSON file for human evaluation UI.

For each of the 50 stratified queries, produces one record containing:
  - Query text, province, law area
  - Gold truth: citations, URLs, snippets, GT answer (user_answer)
  - Retrieval per method: retrieved doc IDs + chunk text spans (char offsets)
  - Combined unique documents across all methods
  - Generated answers from 3 generators (Gemini, Qwen, Gemma)

Output: analysis/eval_ui/eval_ui_data.json

Usage:
    # Build from actual results:
    python scripts/build_eval_ui_data.py

    # Build with dummy values for UI development:
    python scripts/build_eval_ui_data.py --dummy

Requires: runs with retrieval + answer results (or --dummy for placeholder data)
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "test_dataset"
RUNS_DIR = PROJECT_ROOT / "runs"
OUTPUT_DIR = PROJECT_ROOT / "analysis" / "eval_ui"

STRATIFIED_PATH = DATA_DIR / "stratified_50_queries.json"
QUERIES_PATH = DATA_DIR / "queries.json"
DATASET_PATH = DATA_DIR / "test_dataset.parquet"
BATCHES_PATH = DATA_DIR / "batches_law_areas.csv"

RETRIEVAL_METHODS = {
    "qwen_8192_irg_rerank": "qwen_recursive_8192_iterretgen_rerank_strat50",
    "gemini2_4096_irg": "gemini2_recursive_4096_iterretgen_1k-docs",
    "kanon2_8192_rerank": "kanon2_recursive_8192_rerank_1k-docs",
}

GENERATOR_NAMES = ["gemini", "qwen", "gemma"]

METHOD_INDEX_IDS = {
    "qwen_8192_irg_rerank": "qwen3_embedding_8b__recursive8192__ad014d42de",
    "gemini2_4096_irg": "gemini_embedding_2__recursive4096__b280f9ae16",
    "kanon2_8192_rerank": "kanon_2_embedder__recursive8192__005dcf2f82",
}


def load_chunk_index(index_id: str) -> dict[str, list[dict]]:
    """Load chunks and group by doc_id, sorted by chunk_idx."""
    path = RUNS_DIR / "indexes" / index_id / "index.chunks.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        chunks = pickle.load(f)
    by_doc: dict[str, list] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    for v in by_doc.values():
        v.sort(key=lambda c: c.chunk_idx)
    return by_doc


def compute_char_spans(doc_text: str, doc_chunks: list, retrieved_chunk_idxs: set[int]) -> list[dict]:
    """Map chunk_idx values to approximate character offsets in the full document text."""
    spans = []
    offset = 0
    for chunk in doc_chunks:
        # Find the chunk text in the document starting from current offset
        pos = doc_text.find(chunk.text[:200], max(0, offset - 300))
        if pos == -1:
            pos = doc_text.find(chunk.text[:100])
        if pos == -1:
            pos = offset

        if chunk.chunk_idx in retrieved_chunk_idxs:
            spans.append({
                "chunk_idx": chunk.chunk_idx,
                "char_start": pos,
                "char_end": pos + len(chunk.text),
                "text_preview": chunk.text[:200],
            })
        offset = pos + len(chunk.text) - 256  # account for overlap
    return spans


def build_retrieval_info(query_id: int, method_name: str, experiment_id: str,
                          strat_ids: set, chunk_index: dict, doc_texts: dict,
                          top_k_docs: int = 10) -> dict | None:
    """Extract retrieval results for one method + one query."""
    results_file = RUNS_DIR / experiment_id / "results" / "query_results.jsonl"
    if not results_file.exists():
        return None

    with open(results_file) as f:
        for line in f:
            row = json.loads(line)
            if row.get("query_id") == query_id:
                retrieved_ids = row.get("retrieved_ids", [])
                unique_docs = list(dict.fromkeys(retrieved_ids))[:top_k_docs]

                # Build per-document chunk spans
                doc_spans = {}
                chunk_details = row.get("retrieved_chunk_details", [])
                for did in unique_docs:
                    chunk_idxs = {cd["chunk_idx"] for cd in chunk_details if cd.get("doc_id") == did}
                    if did in chunk_index and did in doc_texts:
                        doc_spans[did] = compute_char_spans(doc_texts[did], chunk_index[did], chunk_idxs)
                    elif chunk_idxs:
                        doc_spans[did] = [{"chunk_idx": ci, "char_start": -1, "char_end": -1, "text_preview": ""}
                                           for ci in sorted(chunk_idxs)]

                return {
                    "method": method_name,
                    "retrieved_doc_ids": unique_docs,
                    "num_chunks": len([rid for rid in retrieved_ids if rid in set(unique_docs)]),
                    "num_unique_docs": len(unique_docs),
                    "chunk_spans_by_doc": doc_spans,
                }
    return None


def build_dummy_record(query_id: int = 999) -> dict:
    """Create a dummy record for UI development."""
    return {
        "query_id": query_id,
        "query_text": "What constitutes wrongful dismissal in Ontario and what remedies are available?",
        "province": "Ontario",
        "law_area": "employment",
        "gold_truth": {
            "citations": ["2022 ONCA 100", "2019 SCC 65"],
            "urls": {
                "2022 ONCA 100": "https://canlii.ca/t/example1",
                "2019 SCC 65": "https://canlii.ca/t/example2",
            },
            "snippets": {
                "2022 ONCA 100": [
                    "The employer failed to provide adequate notice of termination.",
                    "An employee with 17 years of service is entitled to reasonable notice.",
                ],
                "2019 SCC 65": [
                    "The standard of review for administrative decisions requires reasonableness.",
                ],
            },
            "gt_answer": (
                "1. Opening Statements\n"
                "This area of law concerns wrongful dismissal and the remedies available.\n\n"
                "2. Supporting Arguments\n"
                "Courts have held that employers must provide reasonable notice (2022 ONCA 100).\n\n"
                "3. Final Conclusion\n"
                "An employee dismissed without adequate notice may claim damages."
            ),
        },
        "retrieval_methods": {
            "qwen_8192_irg_rerank": {
                "method": "qwen_8192_irg_rerank",
                "retrieved_doc_ids": ["2022 ONCA 100", "2019 SCC 65", "2021 BCSC 45"],
                "num_chunks": 15,
                "num_unique_docs": 3,
                "chunk_spans_by_doc": {
                    "2022 ONCA 100": [
                        {"chunk_idx": 0, "char_start": 0, "char_end": 8192, "text_preview": "The trial judge found..."},
                        {"chunk_idx": 3, "char_start": 22000, "char_end": 30192, "text_preview": "The employer failed..."},
                    ],
                    "2019 SCC 65": [
                        {"chunk_idx": 5, "char_start": 40000, "char_end": 48192, "text_preview": "The standard of review..."},
                    ],
                },
            },
            "gemini1_8192_irg": {
                "method": "gemini1_8192_irg",
                "retrieved_doc_ids": ["2022 ONCA 100", "2019 SCC 65"],
                "num_chunks": 10,
                "num_unique_docs": 2,
                "chunk_spans_by_doc": {
                    "2022 ONCA 100": [
                        {"chunk_idx": 0, "char_start": 0, "char_end": 8192, "text_preview": "The trial judge found..."},
                    ],
                },
            },
            "kanon2_8192_rerank": {
                "method": "kanon2_8192_rerank",
                "retrieved_doc_ids": ["2022 ONCA 100", "2021 BCSC 45", "2020 SCC 16"],
                "num_chunks": 12,
                "num_unique_docs": 3,
                "chunk_spans_by_doc": {},
            },
        },
        "combined_unique_docs": [
            {
                "citation": "2022 ONCA 100",
                "url": "https://canlii.ca/t/example1",
                "name": "Smith v. Employer Corp.",
                "char_count": 65000,
                "is_gold": True,
                "retrieved_by": ["qwen_8192_irg_rerank", "gemini1_8192_irg", "kanon2_8192_rerank"],
            },
            {
                "citation": "2019 SCC 65",
                "url": "https://canlii.ca/t/example2",
                "name": "Canada v. Vavilov",
                "char_count": 410000,
                "is_gold": True,
                "retrieved_by": ["qwen_8192_irg_rerank", "gemini1_8192_irg"],
            },
            {
                "citation": "2021 BCSC 45",
                "url": "",
                "name": "Doe v. Company",
                "char_count": 30000,
                "is_gold": False,
                "retrieved_by": ["qwen_8192_irg_rerank", "kanon2_8192_rerank"],
            },
            {
                "citation": "2020 SCC 16",
                "url": "",
                "name": "Uber v. Heller",
                "char_count": 345000,
                "is_gold": False,
                "retrieved_by": ["kanon2_8192_rerank"],
            },
        ],
        "generated_answers": {
            "gemini": "1. Opening Statements\nThis response addresses wrongful dismissal...\n\n2. Supporting Arguments\nIn 2022 ONCA 100, the court found...\n\n3. Final Conclusion\nBased on the evidence, an employee dismissed without cause...",
            "qwen": "[Placeholder — Qwen answer not yet generated]",
            "gemma": "[Placeholder — Gemma answer not yet generated]",
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Build evaluation UI data from retrieval + answer results.")
    parser.add_argument("--dummy", action="store_true", help="Generate dummy data for UI development")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.dummy:
        dummy = [build_dummy_record(i) for i in [999, 998, 997]]
        out_path = OUTPUT_DIR / "eval_ui_data_dummy.json"
        out_path.write_text(json.dumps(dummy, indent=2, ensure_ascii=False))
        print(f"Saved dummy data ({len(dummy)} records) to {out_path}")
        return

    # Load all source data
    import pandas as pd

    strat_queries = json.loads(STRATIFIED_PATH.read_text())
    strat_ids = {q["query_id"] for q in strat_queries}

    all_queries = json.loads(QUERIES_PATH.read_text())
    query_lookup = {q["query_id"]: q for q in all_queries}

    batches = pd.read_csv(BATCHES_PATH)
    batch_to_area = dict(zip(batches["batch number"], batches["cleaned_law_area"]))

    doc_df = pd.read_parquet(DATASET_PATH, columns=["citation", "text", "url", "name"])
    doc_texts = dict(zip(doc_df["citation"], doc_df["text"]))
    doc_urls = dict(zip(doc_df["citation"], doc_df["url"].fillna("")))
    doc_names = dict(zip(doc_df["citation"], doc_df["name"].fillna("")))

    chunk_indexes = {}
    for method_name, index_id in METHOD_INDEX_IDS.items():
        chunk_indexes[method_name] = load_chunk_index(index_id)
        print(f"  chunk index {method_name}: {sum(len(v) for v in chunk_indexes[method_name].values())} chunks")

    # Load generator answers
    gen_answers: dict[str, dict[int, str]] = {}
    for gen_name in GENERATOR_NAMES:
        gen_file = RUNS_DIR / "multi_generator_50" / "results" / f"{gen_name}_answers.jsonl"
        answers = {}
        if gen_file.exists():
            with open(gen_file) as f:
                for line in f:
                    row = json.loads(line)
                    if row.get("answer"):
                        answers[row["query_id"]] = row["answer"]
        gen_answers[gen_name] = answers
        print(f"  {gen_name}: {len(answers)} answers loaded")

    # Build records
    records = []
    for sq in strat_queries:
        qid = sq["query_id"]
        q = query_lookup.get(qid, sq)
        query_text = q.get("query_text", "")
        province = q.get("province", "")
        if province:
            query_text_full = f"I am in {province}. {query_text}"
        else:
            query_text_full = query_text

        gold_citations = list(q.get("ground_truth_citations", []))
        law_area = batch_to_area.get(q.get("batch_id"), "unknown")

        # Gold truth info
        gold_urls = {cit: doc_urls.get(cit, "") for cit in gold_citations}
        gold_snippets = q.get("ground_truth_snippets", {})
        gt_answer = q.get("user_answer", "")

        # Retrieval per method
        retrieval_methods = {}
        for method_name, experiment_id in RETRIEVAL_METHODS.items():
            info = build_retrieval_info(qid, method_name, experiment_id, strat_ids, chunk_indexes.get(method_name, {}), doc_texts)
            if info:
                retrieval_methods[method_name] = info
            else:
                retrieval_methods[method_name] = {
                    "method": method_name, "retrieved_doc_ids": [],
                    "num_chunks": 0, "num_unique_docs": 0, "chunk_spans_by_doc": {},
                    "note": "results not available yet",
                }

        # Combined unique documents
        all_retrieved_docs = set()
        doc_retrieved_by: dict[str, list[str]] = {}
        for method_name, info in retrieval_methods.items():
            for did in info.get("retrieved_doc_ids", []):
                all_retrieved_docs.add(did)
                doc_retrieved_by.setdefault(did, []).append(method_name)

        # Include gold docs even if not retrieved
        for cit in gold_citations:
            all_retrieved_docs.add(cit)

        combined_docs = []
        for did in sorted(all_retrieved_docs):
            combined_docs.append({
                "citation": did,
                "url": doc_urls.get(did, ""),
                "name": doc_names.get(did, ""),
                "char_count": len(doc_texts.get(did, "")),
                "text": doc_texts.get(did, ""),
                "is_gold": did in gold_citations,
                "retrieved_by": doc_retrieved_by.get(did, []),
            })

        # Generated answers
        answers = {}
        for gen_name in GENERATOR_NAMES:
            answers[gen_name] = gen_answers[gen_name].get(qid, f"[Not yet generated — {gen_name}]")

        records.append({
            "query_id": qid,
            "query_text": query_text_full,
            "province": province,
            "law_area": law_area,
            "gold_truth": {
                "citations": gold_citations,
                "urls": gold_urls,
                "snippets": gold_snippets,
                "gt_answer": gt_answer,
            },
            "retrieval_methods": retrieval_methods,
            "combined_unique_docs": combined_docs,
            "generated_answers": answers,
        })

    out_path = OUTPUT_DIR / "eval_ui_data.json"
    out_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))

    n_with_all_answers = sum(1 for r in records
                             if all(not v.startswith("[Not yet") for v in r["generated_answers"].values()))
    print(f"\nSaved {len(records)} records to {out_path}")
    print(f"  With all 3 answers: {n_with_all_answers}")
    print(f"  File size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
