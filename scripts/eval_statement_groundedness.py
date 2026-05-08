"""
Evaluate groundedness of generated answers by checking each statement
against the corpus via vector embedding search.

For each statement in each generated answer:
  1. Embed the statement using the experiment's embedder (Qwen)
  2. Search the FAISS index for the top-k most similar chunks
  3. Have Gemini Flash judge whether the statement is supported by at
     least one of the retrieved chunks

Outputs:
  statement_groundedness.jsonl  — per-query, per-statement results
  statement_groundedness_summary.json — aggregate scores

Usage
-----
    # Using iterretgen answers:
    python scripts/eval_statement_groundedness.py \\
        --config configs/experiments/qwen_recursive_1024.yaml \\
        --answers runs/qwen_recursive_1024_iterretgen_1k-docs/results/query_results.jsonl \\
        --answer-field answer

    # Using answer_eval file (which has answers in 'generated_answer' field):
    python scripts/eval_statement_groundedness.py \\
        --config configs/experiments/qwen_recursive_1024.yaml \\
        --answers runs/qwen_recursive_1024_1k-docs/results/answer_eval_k25.jsonl \\
        --answer-field generated_answer

Requires: GOOGLE_API_KEY or GEMINI_API_KEY (for Gemini Flash judge)
Requires: a pre-built FAISS index (run run_indexing.py first)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import numpy as np
from tqdm import tqdm

from benchmark_rag.components.retrievers.faiss_retriever import FaissRetriever
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.registry import build_from_component_config


_JUDGE_SYSTEM_PROMPT = (
    "You are an impartial judge evaluating whether a statement is supported "
    "by the provided evidence passages from Canadian legal documents."
)

_JUDGE_PROMPT_TEMPLATE = """\
STATEMENT:
{statement}

EVIDENCE PASSAGES:
{evidence}

Is the STATEMENT supported (fully or partially) by at least one of the EVIDENCE \
PASSAGES above? A statement is "supported" if the evidence contains information \
that substantiates the claim — it does not need to be a word-for-word match.

First, explain your reasoning: identify which evidence passage(s) are relevant \
(if any) and whether they substantiate the claim in the statement.

Then, on a new line, give your final verdict as exactly one of:
  VERDICT: SUPPORTED — the statement is substantiated by the evidence
  VERDICT: NOT_SUPPORTED — the evidence does not contain information supporting this statement
  VERDICT: PARTIAL — the statement is partially supported but makes claims beyond the evidence\
"""


def _split_into_statements(answer: str) -> list[str]:
    """Split an answer into individual statements (sentences)."""
    answer = answer.strip()
    if not answer:
        return []
    lines = answer.split("\n")
    text_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if re.match(r"^\d+\.\s+(Opening Statements|Supporting Arguments|Final Conclusion)\s*$", stripped):
            continue
        text_lines.append(stripped)
    text = " ".join(text_lines)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _format_evidence(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] ({c['doc_id']})\n{c['text']}")
    return "\n\n".join(parts)


def _parse_verdict(judge_text: str) -> str:
    """Extract verdict from judge response (appears after 'VERDICT:')."""
    match = re.search(r"VERDICT:\s*(SUPPORTED|NOT_SUPPORTED|PARTIAL)", judge_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: check first word of last line
    lines = [l.strip() for l in judge_text.strip().split("\n") if l.strip()]
    if lines:
        last = lines[-1].upper()
        if "NOT_SUPPORTED" in last or "NOT SUPPORTED" in last:
            return "NOT_SUPPORTED"
        if "SUPPORTED" in last:
            return "SUPPORTED"
        if "PARTIAL" in last:
            return "PARTIAL"
    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True,
                        help="Experiment YAML (used to find the FAISS index and embedder)")
    parser.add_argument("--answers", required=True,
                        help="Path to JSONL file containing generated answers")
    parser.add_argument("--answer-field", default="answer",
                        help="JSON field name for the answer text (default: 'answer')")
    parser.add_argument("--k", type=int, default=25,
                        help="Number of chunks to retrieve per statement (default: 25)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: same dir as --answers)")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Limit number of queries to evaluate (for testing)")
    parser.add_argument("--judge-model", default="gemini-2.5-flash",
                        help="Gemini model for judging (default: gemini-2.5-flash)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    log = logging.getLogger(__name__)

    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        sys.exit("ERROR: set GOOGLE_API_KEY or GEMINI_API_KEY")

    cfg = ExperimentConfig.from_yaml(args.config)

    # --- Load answers ---
    answers_path = Path(args.answers)
    if not answers_path.exists():
        sys.exit(f"ERROR: answers file not found at {answers_path}")

    records = []
    with open(answers_path) as f:
        for line in f:
            rec = json.loads(line)
            answer = rec.get(args.answer_field) or ""
            if answer.strip():
                records.append(rec)
    log.info(f"Loaded {len(records)} answers from {answers_path}")

    if args.max_queries:
        records = records[:args.max_queries]
        log.info(f"Limited to {len(records)} queries")

    # --- Load embedder + FAISS index ---
    log.info("Loading embedder...")
    embedder = build_from_component_config(cfg.embedder.to_build_dict())

    log.info("Loading FAISS index...")
    retriever = FaissRetriever(metric=cfg.retriever.model_extra.get("metric", "cosine"))
    index_path = Path(cfg.indexing.output_dir) / "index"
    retriever.load_index(index_path)
    log.info(f"  {retriever._index.ntotal} vectors loaded")

    # --- Set up Gemini judge ---
    from google import genai
    from google.genai import types
    from benchmark_rag.components.generators.gemini import _generate_with_retry

    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=key)

    judge_cfg = types.GenerateContentConfig(
        system_instruction=_JUDGE_SYSTEM_PROMPT,
        temperature=0.0,
        max_output_tokens=512,
    )

    # --- Output setup ---
    output_dir = Path(args.output_dir) if args.output_dir else answers_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "statement_groundedness.jsonl"
    summary_path = output_dir / "statement_groundedness_summary.json"

    # --- Stats ---
    total_statements = 0
    total_supported = 0
    total_partial = 0
    total_not_supported = 0
    total_judge_calls = 0
    total_input_tokens = 0
    total_output_tokens = 0
    per_query_scores: list[dict] = []

    with open(results_path, "w") as out_f:
        for rec in tqdm(records, desc="Evaluating"):
            query_id = rec.get("query_id")
            answer_text = rec.get(args.answer_field, "")
            statements = _split_into_statements(answer_text)

            if not statements:
                continue

            # Embed all statements in this answer
            statement_embeddings = embedder.embed(statements)

            query_results: list[dict] = []
            query_supported = 0
            query_partial = 0

            for stmt, emb in zip(statements, statement_embeddings):
                # Search FAISS for top-k chunks
                retrieved = retriever.retrieve(emb, k=args.k)
                evidence_chunks = [
                    {"doc_id": c.doc_id, "chunk_idx": c.chunk_idx,
                     "text": c.text, "score": c.score}
                    for c in retrieved
                ]
                evidence_text = _format_evidence(evidence_chunks)

                # Judge (CoT: explanation first, then verdict)
                prompt = _JUDGE_PROMPT_TEMPLATE.format(
                    statement=stmt, evidence=evidence_text,
                )
                try:
                    response = _generate_with_retry(
                        client, model=args.judge_model,
                        contents=prompt, config=judge_cfg,
                    )
                    usage = response.usage_metadata
                    total_judge_calls += 1
                    total_input_tokens += (usage.prompt_token_count or 0)
                    total_output_tokens += (usage.candidates_token_count or 0)
                    judge_text = (response.text or "").strip()
                except Exception as e:
                    log.warning("Judge call failed for query %s: %s", query_id, e)
                    judge_text = "ERROR"

                verdict = _parse_verdict(judge_text)
                if verdict == "SUPPORTED":
                    query_supported += 1
                elif verdict == "PARTIAL":
                    query_partial += 1

                query_results.append({
                    "statement": stmt,
                    "verdict": verdict,
                    "judge_response": judge_text,
                    "top_evidence_doc_ids": [c["doc_id"] for c in evidence_chunks[:5]],
                })

            n_stmts = len(statements)
            total_statements += n_stmts
            total_supported += query_supported
            total_partial += query_partial
            total_not_supported += n_stmts - query_supported - query_partial

            groundedness = (query_supported + 0.5 * query_partial) / n_stmts if n_stmts else 0.0

            query_record = {
                "query_id": query_id,
                "num_statements": n_stmts,
                "supported": query_supported,
                "partial": query_partial,
                "not_supported": n_stmts - query_supported - query_partial,
                "groundedness_score": round(groundedness, 4),
                "statements": query_results,
            }
            out_f.write(json.dumps(query_record) + "\n")
            per_query_scores.append({
                "query_id": query_id,
                "groundedness": groundedness,
                "n": n_stmts,
            })

    # --- Summary ---
    overall_groundedness = (
        (total_supported + 0.5 * total_partial) / total_statements
        if total_statements else 0.0
    )
    groundedness_scores = [q["groundedness"] for q in per_query_scores]

    summary = {
        "config": args.config,
        "answers_source": str(answers_path),
        "answer_field": args.answer_field,
        "k": args.k,
        "judge_model": args.judge_model,
        "num_queries_evaluated": len(per_query_scores),
        "total_statements": total_statements,
        "total_supported": total_supported,
        "total_partial": total_partial,
        "total_not_supported": total_not_supported,
        "overall_groundedness": round(overall_groundedness, 4),
        "mean_query_groundedness": round(float(np.mean(groundedness_scores)), 4) if groundedness_scores else 0.0,
        "median_query_groundedness": round(float(np.median(groundedness_scores)), 4) if groundedness_scores else 0.0,
        "judge_cost": {
            "calls": total_judge_calls,
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    log.info(f"Results saved to {results_path}")
    log.info(f"Summary saved to {summary_path}")
    print(f"\n{'='*60}")
    print(f"Groundedness Evaluation — {cfg.experiment_id}")
    print(f"{'='*60}")
    print(f"Queries evaluated : {len(per_query_scores)}")
    print(f"Total statements  : {total_statements}")
    print(f"Supported         : {total_supported} ({100*total_supported/total_statements:.1f}%)")
    print(f"Partial           : {total_partial} ({100*total_partial/total_statements:.1f}%)")
    print(f"Not supported     : {total_not_supported} ({100*total_not_supported/total_statements:.1f}%)")
    print(f"Overall score     : {overall_groundedness:.4f}")
    print(f"Mean per-query    : {np.mean(groundedness_scores):.4f}")
    print(f"Judge calls       : {total_judge_calls}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
