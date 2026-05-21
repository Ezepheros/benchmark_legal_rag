"""
Save example prompts for each pipeline type to prompt_examples/ for inspection.

Usage
-----
    python scripts/save_prompt_examples.py

Requires: GPU (for Qwen embedder), GOOGLE_API_KEY (for Gemini intermediate gen)
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    print(f"Loading .env from: {env_path.resolve()}")
    print(f"  exists: {env_path.exists()}")
    load_dotenv(env_path, override=True)
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "prompt_examples"
OUT_DIR.mkdir(exist_ok=True)

QUERIES_PATH = PROJECT_ROOT / "data" / "test_dataset" / "queries.json"


def get_sample_query() -> dict:
    queries = json.loads(QUERIES_PATH.read_text())
    # Pick one with moderate GT docs
    for q in queries:
        if len(q.get("ground_truth_citations", [])) >= 3 and q.get("province") == "Ontario":
            return q
    return queries[0]


def save_iterretgen_example():
    """Save all prompts from an IterRetGen run."""
    from benchmark_rag.config.schemas import ExperimentConfig
    from benchmark_rag.pipeline.iterretgen_pipeline import IterRetGenPipeline
    from benchmark_rag.components.generators.gemini import _build_context
    from benchmark_rag.prompts.iterretgen import FULL_INTERMEDIATE_PROMPT
    from benchmark_rag.prompts.answer_generator import ANSWER_SYSTEM_PROMPT

    cfg = ExperimentConfig.from_yaml(PROJECT_ROOT / "configs/experiments/gemma_recursive_4096_iterretgen.yaml")
    pipeline = IterRetGenPipeline.from_config(cfg)

    q = get_sample_query()
    query_text = f"I am in {q['province']}. {q['query_text']}"

    # Run manually to capture each step
    current_query = query_text
    all_prompts = []

    for iteration in range(pipeline.max_iterations):
        query_emb = pipeline.embedder.embed([current_query])[0]
        chunks = pipeline.retriever.retrieve(query_emb, k=5)

        context = _build_context(chunks)
        prompt = f"Context:\n{context}\n\nQuestion: {query_text}"

        if iteration < pipeline.max_iterations - 1:
            # Intermediate generation
            all_prompts.append({
                "step": f"iteration_{iteration + 1}_intermediate",
                "system_prompt": FULL_INTERMEDIATE_PROMPT,
                "user_prompt": prompt,
                "augmented_query": current_query,
                "num_chunks": len(chunks),
            })

            intermediate_answer = pipeline.intermediate_generator.generate(query_text, chunks)
            current_query = f"{query_text}\n{intermediate_answer}"
        else:
            # Final generation
            all_prompts.append({
                "step": f"iteration_{iteration + 1}_final",
                "system_prompt": ANSWER_SYSTEM_PROMPT,
                "user_prompt": prompt,
                "augmented_query": current_query,
                "num_chunks": len(chunks),
            })

    out_path = OUT_DIR / "iterretgen_prompts.json"
    with open(out_path, "w") as f:
        json.dump({
            "query_id": q["query_id"],
            "query_text": query_text,
            "config": "gemma_recursive_4096_iterretgen.yaml",
            "max_iterations": pipeline.max_iterations,
            "prompts": all_prompts,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")


def save_oracle_generation_example():
    """Save the oracle generation prompt."""
    from benchmark_rag.prompts.answer_generator import ANSWER_SYSTEM_PROMPT

    # Load a pre-computed oracle result
    oracle_path = PROJECT_ROOT / "runs" / "oracle_qwen_8192_rerank_1k-docs" / "results" / "query_results.jsonl"
    with open(oracle_path) as f:
        row = json.loads(f.readline())

    # Load chunks from index
    index_path = PROJECT_ROOT / "runs" / "indexes" / "qwen3_embedding_8b__recursive8192__ad014d42de" / "index.chunks.pkl"
    with open(index_path, "rb") as f:
        all_chunks = pickle.load(f)
    chunk_lookup = {(c.doc_id, c.chunk_idx): c.text for c in all_chunks}

    # Build context like run_full_eval does
    import pandas as pd
    doc_df = pd.read_parquet(PROJECT_ROOT / "data" / "test_dataset" / "test_dataset.parquet", columns=["citation", "text"])
    doc_texts = dict(zip(doc_df["citation"], doc_df["text"]))

    from scripts.run_full_eval import build_context, TOKEN_BUDGET
    from benchmark_rag.components.base import RetrievedChunk

    retrieved = []
    for i, doc_id in enumerate(row.get("retrieved_ids", [])[:25]):
        detail = row.get("retrieved_chunk_details", [{}])[i] if i < len(row.get("retrieved_chunk_details", [])) else {}
        cidx = detail.get("chunk_idx", i)
        text = chunk_lookup.get((doc_id, cidx), "")
        retrieved.append(RetrievedChunk(text=text, doc_id=doc_id, chunk_idx=cidx, metadata={}, score=detail.get("score", 0.0)))

    context, ctx_meta = build_context(retrieved, doc_texts, TOKEN_BUDGET)
    prompt = f"Context:\n{context}\n\nQuestion: {row['query_text']}"

    out_path = OUT_DIR / "oracle_generation_prompt.json"
    with open(out_path, "w") as f:
        json.dump({
            "query_id": row["query_id"],
            "query_text": row["query_text"],
            "system_prompt": ANSWER_SYSTEM_PROMPT,
            "user_prompt": prompt,
            "context_meta": ctx_meta,
            "num_retrieved_chunks": len(retrieved),
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")


def save_groundedness_judge_example():
    """Save an example groundedness judge prompt from the batch JSONL."""
    for cond in ["oracle", "pipeline"]:
        for gen in ["gemini", "gemma", "qwen"]:
            path = PROJECT_ROOT / "runs" / "final_eval" / cond / "batch" / f"groundedness_requests_{gen}.jsonl"
            if path.exists():
                with open(path) as f:
                    entry = json.loads(f.readline())
                out_path = OUT_DIR / f"groundedness_judge_prompt_{cond}_{gen}.json"
                with open(out_path, "w") as f:
                    json.dump({
                        "key": entry["key"],
                        "system_instruction": entry["request"]["system_instruction"]["parts"][0]["text"],
                        "user_prompt": entry["request"]["contents"][0]["parts"][0]["text"],
                        "generation_config": entry["request"]["generation_config"],
                    }, f, indent=2, ensure_ascii=False)
                print(f"Saved: {out_path}")
                return  # Just one example


def save_atomic_fact_judge_example():
    """Save an example atomic fact judge prompt from the batch JSONL."""
    path = PROJECT_ROOT / "runs" / "final_eval" / "oracle" / "batch" / "judge_requests.jsonl"
    if path.exists():
        with open(path) as f:
            entry = json.loads(f.readline())
        out_path = OUT_DIR / "atomic_fact_judge_prompt.json"
        with open(out_path, "w") as f:
            json.dump({
                "key": entry["key"],
                "system_instruction": entry["request"]["system_instruction"]["parts"][0]["text"],
                "user_prompt": entry["request"]["contents"][0]["parts"][0]["text"],
                "generation_config": entry["request"]["generation_config"],
            }, f, indent=2, ensure_ascii=False)
        print(f"Saved: {out_path}")


def save_agentic_example():
    """Save example prompts for each phase of the agentic pipeline (no GPU needed)."""
    from benchmark_rag.prompts.agentic import (
        SEARCH_SYSTEM_PROMPT,
        REVIEW_SYSTEM_PROMPT,
        SUMMARIZE_SYSTEM_PROMPT,
        SUMMARIZE_INSTRUCTION,
        ANSWER_SYSTEM_PROMPT,
    )

    q = get_sample_query()
    query_text = f"I am in {q['province']}. {q['query_text']}"

    # Simulate the state at different points
    empty_state = (
        "=== Research State ===\n"
        "No searches run yet.\n\n"
        "No documents saved yet."
    )

    mid_state = (
        '=== Research State ===\n'
        'Searches run: "employment insurance just cause" | "voluntary leaving benefits"\n\n'
        'Saved documents (2):\n\n'
        '[2009 FCA 122] Canada (Attorney General) v. Smith  (found via: "employment insurance just cause")\n'
        'The Federal Court of Appeal held that voluntarily leaving employment without just cause '
        'under s. 29(c) of the Employment Insurance Act results in disqualification. The court '
        'emphasized that "just cause" requires showing no reasonable alternative to leaving.\n\n'
        '[2008 FCA 18] Jones v. Canada  (found via: "voluntary leaving benefits")\n'
        'This case established that the burden of proving just cause rests on the claimant. '
        'The court considered factors including financial circumstances and availability of alternatives.'
    )

    # Search phase prompt (iteration 1)
    search_prompt_1 = f"Question: {query_text}\n\n{empty_state}\n\nYou need to find 25 more documents."

    # Search phase prompt (iteration 3, with state)
    search_prompt_3 = f"Question: {query_text}\n\n{mid_state}\n\nYou need to find 23 more documents."

    # Review phase prompt (after a search)
    review_prompt = (
        f"Question: {query_text}\n\n"
        f'SEARCH RESULTS for "employment insurance just cause":\n'
        f"[1] (2009 FCA 122, score=12.45)\n"
        f"The Federal Court of Appeal held that voluntarily leaving employment...\n\n"
        f"[2] (2015 FC 1142, score=10.23)\n"
        f"The applicant sought judicial review of a decision denying benefits...\n\n"
        f"[3] (2003 FCA 377, score=9.87)\n"
        f"The court considered whether the claimant had just cause...\n\n"
        f"{empty_state}"
    )

    # Summarize prompt
    summarize_prompt = (
        f"QUESTION: {query_text}\n\n"
        f"DOCUMENT [2009 FCA 122] — Canada (Attorney General) v. Smith\n\n"
        f"[Full document text would appear here...]\n\n"
        f"{SUMMARIZE_INSTRUCTION}"
    )

    # Final answer prompt
    answer_prompt = (
        f"Question: {query_text}\n\n"
        f"SAVED DOCUMENTS:\n\n"
        f"=== [2009 FCA 122] Canada (Attorney General) v. Smith ===\n"
        f"[Full document text...]\n\n"
        f"=== [2008 FCA 18] Jones v. Canada ===\n"
        f"[Full document text...]\n"
    )

    out_path = OUT_DIR / "agentic_prompts.json"
    with open(out_path, "w") as f:
        json.dump({
            "query_id": q["query_id"],
            "query_text": query_text,
            "config": "agentic_gemini_bm25_recursive_4096_7iter.yaml",
            "phases": [
                {
                    "phase": "search_iteration_1",
                    "description": "First search — agent chooses keyword_search query",
                    "system_prompt": SEARCH_SYSTEM_PROMPT,
                    "user_prompt": search_prompt_1,
                    "available_tools": ["keyword_search(query, k)"],
                },
                {
                    "phase": "review_iteration_1",
                    "description": "Agent reviews search results and saves relevant citations",
                    "system_prompt": REVIEW_SYSTEM_PROMPT,
                    "user_prompt": review_prompt,
                    "available_tools": ["save_citations(citations)"],
                },
                {
                    "phase": "summarize",
                    "description": "After saving, each new document gets a summary for agent memory",
                    "system_prompt": SUMMARIZE_SYSTEM_PROMPT,
                    "user_prompt": summarize_prompt,
                },
                {
                    "phase": "search_iteration_3",
                    "description": "Later iteration — agent sees summaries of saved docs + past searches",
                    "system_prompt": SEARCH_SYSTEM_PROMPT,
                    "user_prompt": search_prompt_3,
                    "available_tools": ["keyword_search(query, k)"],
                },
                {
                    "phase": "final_answer",
                    "description": "After loop exits — full document context for answer synthesis",
                    "system_prompt": ANSWER_SYSTEM_PROMPT,
                    "user_prompt": answer_prompt,
                },
            ],
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    print("Saving prompt examples...\n")

    # Debug: show which API keys are set
    google_key = os.environ.get("GOOGLE_API_KEY", "")
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    print(f"GOOGLE_API_KEY: {'set (' + google_key[:8] + '...)' if google_key else 'NOT SET'}")
    print(f"GEMINI_API_KEY: {'set (' + gemini_key[:8] + '...)' if gemini_key else 'NOT SET'}")
    print()

    # These don't need GPU
    print("--- Oracle generation prompt ---")
    save_oracle_generation_example()

    print("\n--- Groundedness judge prompt ---")
    save_groundedness_judge_example()

    print("\n--- Atomic fact judge prompt ---")
    save_atomic_fact_judge_example()

    print("\n--- Agentic search prompts ---")
    save_agentic_example()

    # This needs GPU for Qwen/Gemma embedder
    try:
        print("\n--- IterRetGen prompts (needs GPU) ---")
        save_iterretgen_example()
    except Exception as e:
        print(f"  Skipped (needs GPU): {e}")

    print(f"\nAll examples saved to: {OUT_DIR}/")
