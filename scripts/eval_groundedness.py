"""
Evaluate groundedness of generated answers against the decontextualized corpus.

For each generated answer:
  1. Split the answer into statements (sentences).
  2. Decontextualize each statement using Gemini Flash.
  3. Embed each decontextualized statement using Qwen.
  4. Search the decontextualized corpus FAISS index for top-k similar statements.
  5. Have Gemini Flash judge whether the statement is supported by at least one
     retrieved corpus statement.

Groundedness score: percentage of statements judged SUPPORTED or PARTIAL.
PARTIAL is counted equally as SUPPORTED for the main score, but reported
separately for further analysis.

Outputs (saved to runs/{experiment_id}/results/):
  groundedness_eval.jsonl         — per-query, per-statement results
  groundedness_summary.json       — aggregate scores + per-query scores (for histograms)
  decontextualized_answers.json   — decontextualized answer statements (reusable)

Usage
-----
    python scripts/eval_groundedness.py \\
        --config configs/experiments/qwen_recursive_1024.yaml \\
        --answers runs/qwen_recursive_1024_iterretgen_1k-docs/results/query_results.jsonl

    python scripts/eval_groundedness.py \\
        --config configs/experiments/qwen_recursive_1024.yaml \\
        --answers runs/qwen_recursive_1024_1k-docs/results/answer_eval_k25.jsonl \\
        --answer-field generated_answer

Requires: GOOGLE_API_KEY or GEMINI_API_KEY
Requires: GPU (Qwen embedder for statement embedding)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

import faiss
import numpy as np
from tqdm import tqdm

from benchmark_rag.components.decontextualizers.gemini_decontextualizer import (
    GeminiDecontextualizer,
)
from benchmark_rag.components.embedders.qwen import QwenEmbedder
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.logging import setup_experiment_logging, get_logger

# ── Paths to decontextualized corpus ──
_DECONTEXT_DIR = Path(__file__).parent.parent / "benchmark_rag" / "components" / "decontextualizers" / "batch_output"
_FAISS_INDEX_PATH = _DECONTEXT_DIR / "embeddings" / "decontext_statements.faiss"
_META_PATH = _DECONTEXT_DIR / "embeddings" / "decontext_statements_meta.pkl"

# ── Judge prompt ──
_JUDGE_SYSTEM = (
    "You are an impartial judge evaluating whether a statement from a generated "
    "legal answer is supported by evidence from Canadian legal documents."
)

_JUDGE_PROMPT = """\
STATEMENT (from a generated answer):
{statement}

EVIDENCE (retrieved from the legal corpus):
{evidence}

First, explain your reasoning: which evidence passage(s), if any, support the statement?
Does the evidence substantiate the specific claim made?

Then give your final verdict on a new line as exactly one of:
  VERDICT: SUPPORTED — the statement is substantiated by at least one evidence passage
  VERDICT: NOT_SUPPORTED — no evidence passage supports this statement
  VERDICT: PARTIAL — the statement is partially supported but makes claims beyond the evidence\
"""


def split_into_statements(answer: str) -> list[str]:
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


def parse_verdict(judge_text: str) -> str:
    match = re.search(r"VERDICT:\s*(SUPPORTED|NOT_SUPPORTED|PARTIAL)", judge_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
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
                        help="Experiment YAML (for experiment_id and results dir)")
    parser.add_argument("--answers", required=True,
                        help="JSONL file with generated answers")
    parser.add_argument("--answer-field", default="answer",
                        help="JSON field for answer text (default: 'answer')")
    parser.add_argument("--k", type=int, default=25,
                        help="Top-k corpus statements to retrieve per answer statement")
    parser.add_argument("--max-queries", type=int, default=None,
                        help="Limit queries (for testing)")
    parser.add_argument("--judge-model", default="gemini-2.5-pro")
    parser.add_argument("--decontext-model", default="gemini-2.5-flash")
    parser.add_argument("--embedder-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--embedder-device", default="cuda:0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    log = logging.getLogger(__name__)

    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        sys.exit("ERROR: set GOOGLE_API_KEY or GEMINI_API_KEY")

    cfg = ExperimentConfig.from_yaml(args.config)
    results_dir = Path(f"runs/{cfg.experiment_id}/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # --- Load answers ---
    answers_path = Path(args.answers)
    records = []
    with open(answers_path) as f:
        for line in f:
            rec = json.loads(line)
            if (rec.get(args.answer_field) or "").strip():
                records.append(rec)
    log.info(f"Loaded {len(records)} answers from {answers_path}")
    if args.max_queries:
        records = records[:args.max_queries]

    # --- Load decontextualized corpus FAISS index ---
    log.info(f"Loading corpus FAISS index from {_FAISS_INDEX_PATH}")
    corpus_index = faiss.read_index(str(_FAISS_INDEX_PATH))
    with open(_META_PATH, "rb") as f:
        corpus_meta: list[dict] = pickle.load(f)
    log.info(f"  {corpus_index.ntotal} corpus statements loaded")

    # --- Load components ---
    log.info("Loading Qwen embedder...")
    embedder = QwenEmbedder(model_name=args.embedder_model, device=args.embedder_device)

    decontextualizer = GeminiDecontextualizer(model_name=args.decontext_model)

    from google import genai
    from google.genai import types
    from benchmark_rag.components.generators.gemini import _generate_with_retry

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    judge_cfg = types.GenerateContentConfig(
        system_instruction=_JUDGE_SYSTEM,
        temperature=0.0,
        max_output_tokens=512,
    )

    # --- Stats ---
    total_stmts = 0
    total_supported = 0
    total_partial = 0
    total_not_supported = 0
    per_query_scores: list[dict] = []
    decontext_answers: list[dict] = []

    # Collect examples for display
    example_supported: dict | None = None
    example_partial: dict | None = None
    example_not_supported: dict | None = None

    eval_path = results_dir / "groundedness_eval.jsonl"
    checkpoint_interval = max(1, len(records) // 5)

    with open(eval_path, "w") as out_f:
        for qi, rec in enumerate(tqdm(records, desc="Evaluating groundedness")):
            query_id = rec.get("query_id")
            answer_text = rec.get(args.answer_field, "")
            statements = split_into_statements(answer_text)

            if not statements:
                continue

            # Step 1: Decontextualize answer statements
            decontext_result = decontextualizer.decontextualize(
                statements, answer_text,
            )
            if decontext_result is not None:
                decontext_stmts = decontext_result
            else:
                log.warning(f"Decontextualization failed for query {query_id}, using originals")
                decontext_stmts = statements

            decontext_answers.append({
                "query_id": query_id,
                "original_statements": statements,
                "decontextualized_statements": decontext_stmts,
            })

            # Step 2: Embed decontextualized statements
            embeddings = embedder.embed(decontext_stmts)
            query_vectors = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(query_vectors)

            # Step 3: Search corpus
            scores_matrix, indices_matrix = corpus_index.search(query_vectors, args.k)

            # Step 4: Judge each statement
            query_results: list[dict] = []
            q_supported = 0
            q_partial = 0

            for si, (stmt, d_stmt) in enumerate(zip(statements, decontext_stmts)):
                retrieved_indices = indices_matrix[si]
                evidence_parts = []
                evidence_citations = []
                for rank, idx in enumerate(retrieved_indices):
                    if idx < 0:
                        continue
                    meta = corpus_meta[idx]
                    evidence_parts.append(f"[{rank+1}] ({meta['citation']})\n{meta['text']}")
                    evidence_citations.append(meta["citation"])
                evidence_text = "\n\n".join(evidence_parts)

                prompt = _JUDGE_PROMPT.format(statement=d_stmt, evidence=evidence_text)
                try:
                    response = _generate_with_retry(
                        client, model=args.judge_model,
                        contents=prompt, config=judge_cfg,
                    )
                    judge_text = (response.text or "").strip()
                except Exception as e:
                    log.warning(f"Judge failed for query {query_id} stmt {si}: {e}")
                    judge_text = "ERROR"

                verdict = parse_verdict(judge_text)
                if verdict == "SUPPORTED":
                    q_supported += 1
                elif verdict == "PARTIAL":
                    q_partial += 1

                stmt_record = {
                    "original_statement": stmt,
                    "decontextualized_statement": d_stmt,
                    "verdict": verdict,
                    "judge_response": judge_text,
                    "top_evidence_citations": evidence_citations[:5],
                }
                query_results.append(stmt_record)

                # Capture examples
                if verdict == "SUPPORTED" and example_supported is None:
                    example_supported = {**stmt_record, "query_id": query_id}
                elif verdict == "PARTIAL" and example_partial is None:
                    example_partial = {**stmt_record, "query_id": query_id}
                elif verdict == "NOT_SUPPORTED" and example_not_supported is None:
                    example_not_supported = {**stmt_record, "query_id": query_id}

            n = len(statements)
            total_stmts += n
            total_supported += q_supported
            total_partial += q_partial
            total_not_supported += n - q_supported - q_partial

            # Grounded = SUPPORTED + PARTIAL (both count equally)
            grounded = q_supported + q_partial
            groundedness = grounded / n if n else 0.0

            query_record = {
                "query_id": query_id,
                "num_statements": n,
                "supported": q_supported,
                "partial": q_partial,
                "not_supported": n - q_supported - q_partial,
                "groundedness_score": round(groundedness, 4),
                "statements": query_results,
            }
            out_f.write(json.dumps(query_record) + "\n")
            per_query_scores.append({
                "query_id": query_id,
                "groundedness": groundedness,
                "n": n,
            })

            # Checkpoint
            if (qi + 1) % checkpoint_interval == 0:
                out_f.flush()
                decontext_path = results_dir / "decontextualized_answers.json"
                decontext_path.write_text(json.dumps(decontext_answers, indent=2, ensure_ascii=False))
                log.info(f"Checkpoint at {qi+1}/{len(records)}")

    # --- Save decontextualized answers ---
    decontext_path = results_dir / "decontextualized_answers.json"
    decontext_path.write_text(json.dumps(decontext_answers, indent=2, ensure_ascii=False))
    log.info(f"Saved decontextualized answers to {decontext_path}")

    # --- Summary ---
    total_grounded = total_supported + total_partial
    overall = total_grounded / total_stmts if total_stmts else 0.0
    g_scores = [q["groundedness"] for q in per_query_scores]

    summary = {
        "config": args.config,
        "answers_source": str(answers_path),
        "answer_field": args.answer_field,
        "k": args.k,
        "judge_model": args.judge_model,
        "decontext_model": args.decontext_model,
        "num_queries": len(per_query_scores),
        "total_statements": total_stmts,
        "total_supported": total_supported,
        "total_partial": total_partial,
        "total_not_supported": total_not_supported,
        "overall_groundedness": round(overall, 4),
        "mean_query_groundedness": round(float(np.mean(g_scores)), 4) if g_scores else 0.0,
        "median_query_groundedness": round(float(np.median(g_scores)), 4) if g_scores else 0.0,
        "per_query_groundedness": per_query_scores,
    }
    summary_path = results_dir / "groundedness_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    decontextualizer.log_usage_summary()

    # --- Print results ---
    print(f"\n{'='*70}")
    print(f"Groundedness Evaluation — {cfg.experiment_id}")
    print(f"{'='*70}")
    print(f"Queries evaluated : {len(per_query_scores)}")
    print(f"Total statements  : {total_stmts}")
    print(f"Grounded (S+P)    : {total_grounded} ({100*total_grounded/total_stmts:.1f}%)")
    print(f"  Supported       : {total_supported} ({100*total_supported/total_stmts:.1f}%)")
    print(f"  Partial         : {total_partial} ({100*total_partial/total_stmts:.1f}%)")
    print(f"Not supported     : {total_not_supported} ({100*total_not_supported/total_stmts:.1f}%)")
    print(f"Overall score     : {overall:.4f}")
    print(f"Mean per-query    : {np.mean(g_scores):.4f}")
    print(f"Median per-query  : {np.median(g_scores):.4f}")

    # --- Print examples ---
    def _print_example(label: str, ex: dict | None):
        print(f"\n--- Example: {label} ---")
        if ex is None:
            print("  (no example found)")
            return
        print(f"  Query ID: {ex['query_id']}")
        print(f"  Statement: {ex['original_statement'][:200]}")
        if ex['decontextualized_statement'] != ex['original_statement']:
            print(f"  Decontextualized: {ex['decontextualized_statement'][:200]}")
        print(f"  Top evidence from: {ex['top_evidence_citations']}")
        # Extract just the reasoning (before VERDICT line)
        judge = ex['judge_response']
        verdict_idx = judge.upper().find("VERDICT:")
        reasoning = judge[:verdict_idx].strip() if verdict_idx > 0 else judge[:300]
        print(f"  Judge reasoning: {reasoning[:300]}")

    _print_example("SUPPORTED", example_supported)
    _print_example("PARTIAL", example_partial)
    _print_example("NOT_SUPPORTED", example_not_supported)
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
