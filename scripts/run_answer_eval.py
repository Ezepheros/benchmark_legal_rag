"""
Evaluate answer generation quality using atomic-fact comparison.

Retrieves documents at each k, constructs context within a 128k-token budget,
generates answers with Gemini Flash, and judges them against pre-computed
ground-truth atomic facts using Gemini Pro.

Usage
-----
    # First, pre-compute ground-truth atomic facts (once):
    python scripts/decompose_atomic_facts.py

    # Then run evaluation:
    python scripts/run_answer_eval.py --config configs/experiments/qwen_recursive_1024.yaml
    python scripts/run_answer_eval.py --config configs/experiments/qwen_recursive_1024.yaml \
        --k-values 10 25 --token-budget 128000

Context construction logic
--------------------------
For each query at each k:
  1. Retrieve top-k chunks and identify the unique source documents.
  2. If the full text of all source documents fits within the token budget,
     pass them as-is to the generator.
  3. Otherwise, include all retrieved chunks, subtract their token cost from
     the budget, compute a truncation ratio, and truncate every full document
     by that ratio so the combined context fits.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import pandas as pd
from tqdm import tqdm

from benchmark_rag.components.base import RetrievedChunk
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.cost_logging import (
    DEFAULT_BENCHMARK_COST_CSV,
    append_cost_entry,
)
from benchmark_rag.logging import setup_experiment_logging, get_logger
from benchmark_rag.prompts.answer_generator import ANSWER_SYSTEM_PROMPT
from benchmark_rag.evaluation.metrics import EXCLUDED_QUERY_IDS, is_query_usable
from benchmark_rag.prompts.atomic_facts import (
    DECOMPOSE_SYSTEM_PROMPT,
    JUDGE_ATOMIC_SYSTEM_PROMPT,
)

CHARS_PER_TOKEN = 4

_RETRY_DELAYS = [5, 10, 30, 60, 120]
_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30 / 1_000_000, 0.30 / 1_000_000),
    "gemini-2.5-pro":   (1.25 / 1_000_000, 10.0 / 1_000_000),
    "gemini-2.0-flash": (0.10 / 1_000_000, 0.40 / 1_000_000),
}


def _estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    for prefix, (ip, op) in _PRICING.items():
        if model.startswith(prefix):
            return in_tok * ip + out_tok * op
    return 0.0


def _generate_with_retry(client, **kwargs):
    from google.genai.errors import ClientError, ServerError

    attempt = 0
    while True:
        try:
            return client.models.generate_content(**kwargs)
        except (ClientError, ServerError) as e:
            code = getattr(e, "code", None)
            if code not in (429, 503):
                raise
            delay = _RETRY_DELAYS[attempt] if attempt < len(_RETRY_DELAYS) else _RETRY_DELAYS[-1]
            attempt += 1
            logging.warning("Gemini %d — retrying in %ds (attempt %d)...", code, delay, attempt)
            time.sleep(delay)


# ── Context construction ─────────────────────────────────────────────────

def _token_estimate(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def build_context(
    chunks: list[RetrievedChunk],
    doc_texts: dict[str, str],
    token_budget: int,
) -> tuple[str, dict]:
    """Build generator context from retrieved chunks and full documents.

    Returns (context_string, metadata_dict).
    """
    doc_ids = list(dict.fromkeys(c.doc_id for c in chunks))
    full_docs = {did: doc_texts[did] for did in doc_ids if did in doc_texts}

    total_doc_tokens = sum(_token_estimate(t) for t in full_docs.values())

    if total_doc_tokens <= token_budget:
        parts = []
        for did in doc_ids:
            if did in full_docs:
                parts.append(f"=== Document: {did} ===\n{full_docs[did]}")
        context = "\n\n".join(parts)
        return context, {
            "context_mode": "full_documents",
            "num_docs": len(full_docs),
            "total_doc_tokens": total_doc_tokens,
            "token_budget": token_budget,
        }

    chunk_tokens = sum(_token_estimate(c.text) for c in chunks)
    remaining = token_budget - chunk_tokens

    meta = {
        "context_mode": "chunks_and_truncated_docs",
        "num_docs": len(full_docs),
        "num_chunks": len(chunks),
        "chunk_tokens": chunk_tokens,
        "total_doc_tokens": total_doc_tokens,
        "token_budget": token_budget,
    }

    if remaining <= 0:
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[{i}] ({c.doc_id})\n{c.text}")
        meta["context_mode"] = "chunks_only"
        meta["truncation_ratio"] = 0.0
        return "\n\n".join(parts), meta

    ratio = remaining / total_doc_tokens
    meta["truncation_ratio"] = round(ratio, 4)

    chunk_parts = []
    for i, c in enumerate(chunks, 1):
        chunk_parts.append(f"[{i}] ({c.doc_id})\n{c.text}")
    chunk_section = "\n\n".join(chunk_parts)

    doc_parts = []
    for did in doc_ids:
        if did not in full_docs:
            continue
        full_text = full_docs[did]
        truncated_chars = int(len(full_text) * ratio)
        doc_parts.append(f"=== Document: {did} (truncated) ===\n{full_text[:truncated_chars]}")
    doc_section = "\n\n".join(doc_parts)

    context = (
        "RETRIEVED PASSAGES:\n"
        f"{chunk_section}\n\n"
        "FULL DOCUMENT CONTEXT (truncated to fit token budget):\n"
        f"{doc_section}"
    )
    return context, meta


# ── Gemini helpers ────────────────────────────────────────────────────────

class CostTracker:
    def __init__(self):
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self._total_cost: float = 0.0
        self.call_count: int = 0

    def track(self, model: str, in_tok: int, out_tok: int) -> None:
        self.call_count += 1
        self.total_input_tokens += in_tok
        self.total_output_tokens += out_tok
        self._total_cost += _estimate_cost(model, in_tok, out_tok)


def generate_answer(client, query: str, context: str, model: str, tracker: CostTracker) -> str:
    from google.genai import types

    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    response = _generate_with_retry(
        client,
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=ANSWER_SYSTEM_PROMPT,
            temperature=0.0,
            max_output_tokens=2048,
        ),
    )
    usage = response.usage_metadata
    tracker.track(model, usage.prompt_token_count, usage.candidates_token_count)
    return response.text


def decompose_answer(client, answer: str, model: str, tracker: CostTracker) -> list[str]:
    from google.genai import types

    response = _generate_with_retry(
        client,
        model=model,
        contents=f"Decompose the following legal answer into atomic facts:\n\n{answer}",
        config=types.GenerateContentConfig(
            system_instruction=DECOMPOSE_SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    usage = response.usage_metadata
    tracker.track(model, usage.prompt_token_count, usage.candidates_token_count)
    return json.loads(response.text)


def judge_atomic_facts(
    client,
    generated_facts: list[str],
    ground_truth_facts: list[str],
    model: str,
    tracker: CostTracker,
) -> dict:
    from google.genai import types

    prompt = (
        f"generated_facts:\n{json.dumps(generated_facts, ensure_ascii=False)}\n\n"
        f"ground_truth_facts:\n{json.dumps(ground_truth_facts, ensure_ascii=False)}"
    )
    response = _generate_with_retry(
        client,
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=JUDGE_ATOMIC_SYSTEM_PROMPT,
            temperature=0.0,
            response_mime_type="application/json",
        ),
    )
    usage = response.usage_metadata
    tracker.track(model, usage.prompt_token_count, usage.candidates_token_count)
    return json.loads(response.text)


# ── Pipeline construction ─────────────────────────────────────────────────

def build_pipeline(cfg: ExperimentConfig):
    is_hybrid = "hybrid" in cfg.retriever.type.lower()
    is_agentic = cfg.agentic is not None

    if is_agentic:
        from benchmark_rag.pipeline.agentic_pipeline import AgenticRAGPipeline
        return AgenticRAGPipeline.from_config(cfg)
    elif is_hybrid:
        from benchmark_rag.pipeline.hybrid_pipeline import HybridRAGPipeline
        return HybridRAGPipeline.from_config(cfg)
    else:
        from benchmark_rag.pipeline.rag_pipeline import RAGPipeline
        return RAGPipeline.from_config(cfg)


def load_documents(dataset_path: str) -> dict[str, str]:
    """Load full document texts from parquet, keyed by citation."""
    df = pd.read_parquet(dataset_path, columns=["citation", "text"])
    return dict(zip(df["citation"], df["text"]))


def load_queries(queries_path: str) -> list[dict]:
    p = Path(queries_path)
    if p.suffix == ".json":
        return json.loads(p.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported query file format: {p.suffix}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate answer generation with atomic-fact judging.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config")
    parser.add_argument("--k-values", type=int, nargs="+", default=[10, 25], help="k values to evaluate at (default: 10 25)")
    parser.add_argument("--token-budget", type=int, default=128_000, help="Context token budget (default: 128000)")
    parser.add_argument("--atomic-facts-path", default="data/test_dataset/ground_truth_atomic_facts.json", help="Path to pre-computed ground-truth atomic facts")
    parser.add_argument("--generator-model", default="gemini-2.5-flash", help="Model for answer generation")
    parser.add_argument("--judge-model", default="gemini-2.5-pro", help="Model for atomic-fact judging")
    args = parser.parse_args()

    import os
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("ERROR: GOOGLE_API_KEY or GEMINI_API_KEY not set.")

    cfg = ExperimentConfig.from_yaml(args.config)

    setup_experiment_logging(
        experiment_id=cfg.experiment_id,
        log_dir=cfg.logging.log_dir,
        level=cfg.logging.level,
        resource_monitor_interval=0,
    )
    log = get_logger(__name__)

    gt_facts_path = Path(args.atomic_facts_path)
    if not gt_facts_path.exists():
        sys.exit(
            f"ERROR: Ground-truth atomic facts not found at {gt_facts_path}. "
            "Run scripts/decompose_atomic_facts.py first."
        )
    gt_atomic_facts: dict[str, list[str]] = json.loads(gt_facts_path.read_text(encoding="utf-8"))
    log.info("Loaded ground-truth atomic facts for %d queries", len(gt_atomic_facts))

    queries = load_queries(cfg.evaluation.queries_path)
    log.info("Loaded %d queries from %s", len(queries), cfg.evaluation.queries_path)

    doc_texts = load_documents(cfg.dataset.path)
    log.info("Loaded %d documents from %s", len(doc_texts), cfg.dataset.path)

    override_k = max(args.k_values)
    cfg.evaluation.k_values = args.k_values
    cfg.generator = None
    pipeline = build_pipeline(cfg)
    log.info("Pipeline built — retrieving at k=%d", override_k)

    client = genai.Client(api_key=api_key)
    gen_tracker = CostTracker()
    decompose_tracker = CostTracker()
    judge_tracker = CostTracker()

    results_dir = Path(f"runs/{cfg.experiment_id}/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for k in args.k_values:
        log.info("=== Evaluating at k=%d ===", k)
        rows: list[dict] = []
        precision_scores: list[float] = []
        recall_scores: list[float] = []
        failed: list[dict] = []

        for q in tqdm(queries, desc=f"k={k}"):
            query_text = str(q.get("query_text", ""))
            province = q.get("province", "")
            if province:
                query_text = f"I am in {province}. {query_text}"
            qid = str(q["query_id"])
            gold_citations = set(q.get("ground_truth_citations", []))

            if not query_text.strip() or not is_query_usable(q):
                continue

            gt_facts = gt_atomic_facts.get(qid, [])
            if not gt_facts:
                continue

            try:
                result = pipeline.query(query_text, k=override_k)
            except Exception as exc:
                log.error("Query %s retrieval failed: %s", qid, exc)
                failed.append({"query_id": qid, "error": f"retrieval: {exc}"})
                continue

            top_k_chunks = result.retrieved_chunks[:k]

            context, ctx_meta = build_context(top_k_chunks, doc_texts, args.token_budget)

            try:
                answer = generate_answer(
                    client, query_text, context, args.generator_model, gen_tracker,
                )
            except Exception as exc:
                log.error("Query %s generation failed: %s", qid, exc)
                failed.append({"query_id": qid, "error": f"generation: {exc}"})
                continue

            try:
                gen_facts = decompose_answer(
                    client, answer, args.generator_model, decompose_tracker,
                )
            except Exception as exc:
                log.error("Query %s decomposition failed: %s", qid, exc)
                failed.append({"query_id": qid, "error": f"decompose: {exc}"})
                continue

            try:
                judge_result = judge_atomic_facts(
                    client, gen_facts, gt_facts, args.judge_model, judge_tracker,
                )
            except Exception as exc:
                log.error("Query %s judging failed: %s", qid, exc)
                failed.append({"query_id": qid, "error": f"judge: {exc}"})
                continue

            gen_results = judge_result.get("generated_fact_results", [])
            gt_results = judge_result.get("ground_truth_fact_results", [])

            precision = (
                sum(1 for r in gen_results if r.get("in_ground_truth"))
                / len(gen_results)
                if gen_results else 0.0
            )
            recall = (
                sum(1 for r in gt_results if r.get("in_generated"))
                / len(gt_results)
                if gt_results else 0.0
            )
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0 else 0.0
            )

            precision_scores.append(precision)
            recall_scores.append(recall)

            rows.append({
                "query_id": qid,
                "query_text": query_text,
                "k": k,
                "gold_citations": list(gold_citations),
                "retrieved_ids": [c.doc_id for c in top_k_chunks],
                "context_meta": ctx_meta,
                "generated_answer": answer,
                "generated_facts": gen_facts,
                "ground_truth_facts": gt_facts,
                "judge_result": judge_result,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            })

        results_file = results_dir / f"answer_eval_k{k}.jsonl"
        with open(results_file, "w") as f:
            for rec in rows:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if failed:
            fail_file = results_dir / f"answer_eval_k{k}_failures.json"
            fail_file.write_text(json.dumps(failed, indent=2))

        avg_p = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        avg_r = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        avg_f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0

        log.info(
            "k=%d | queries=%d | failed=%d | precision=%.4f | recall=%.4f | f1=%.4f",
            k, len(rows), len(failed), avg_p, avg_r, avg_f1,
        )

    summary = {
        "experiment_id": cfg.experiment_id,
        "k_values": args.k_values,
        "token_budget": args.token_budget,
        "generator_model": args.generator_model,
        "judge_model": args.judge_model,
        "generator_cost": {
            "calls": gen_tracker.call_count,
            "input_tokens": gen_tracker.total_input_tokens,
            "output_tokens": gen_tracker.total_output_tokens,
            "cost_usd": round(gen_tracker._total_cost, 6),
        },
        "decompose_cost": {
            "calls": decompose_tracker.call_count,
            "input_tokens": decompose_tracker.total_input_tokens,
            "output_tokens": decompose_tracker.total_output_tokens,
            "cost_usd": round(decompose_tracker._total_cost, 6),
        },
        "judge_cost": {
            "calls": judge_tracker.call_count,
            "input_tokens": judge_tracker.total_input_tokens,
            "output_tokens": judge_tracker.total_output_tokens,
            "cost_usd": round(judge_tracker._total_cost, 6),
        },
    }
    summary_file = results_dir / "answer_eval_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))
    log.info("Summary saved to %s", summary_file)

    total_cost = gen_tracker._total_cost + decompose_tracker._total_cost + judge_tracker._total_cost
    append_cost_entry(DEFAULT_BENCHMARK_COST_CSV, cfg.experiment_id, total_cost)
    log.info("Total cost: $%.6f", total_cost)


if __name__ == "__main__":
    main()
