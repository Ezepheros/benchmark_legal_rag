"""
Oracle experiment: retrieve from GT-only documents, generate answers, evaluate.

For each query, FAISS search is restricted to only the ground-truth documents
(using IDSelectorBatch), then reranked, then Gemini Flash generates an answer.
Evaluation uses Gemini Pro via the batch API for cost efficiency.

Subcommands:
    generate      — Retrieve (GT-only) + rerank + generate answers (online)
    prepare-eval  — Decontextualize answers, embed, search decontext corpus,
                    prepare batch JSONL for Pro judging
    submit        — Submit batch jobs to Gemini batch API
    status        — Check batch job status
    collect       — Download results, compute groundedness + atomic fact metrics

Usage (from project root):
    # 1. Generate answers (needs GPU + API key)
    python scripts/oracle_experiment.py generate

    # 2. Prepare evaluation batches (needs GPU + API key for Flash)
    python scripts/oracle_experiment.py prepare-eval

    # 3. Submit batch jobs
    python scripts/oracle_experiment.py submit

    # 4. Check status (~24h wait)
    python scripts/oracle_experiment.py status

    # 5. Collect results
    python scripts/oracle_experiment.py collect
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENT_ID = "oracle_qwen_8192_rerank_1k-docs"
RESULTS_DIR = PROJECT_ROOT / "runs" / EXPERIMENT_ID / "results"
BATCH_DIR = RESULTS_DIR / "oracle_batch"

# Configs
INDEX_CONFIG = "configs/experiments/qwen_recursive_8192.yaml"
QUERIES_PATH = "data/test_dataset/queries.json"
DATASET_PATH = "data/test_dataset/test_dataset.parquet"
ATOMIC_FACTS_PATH = "data/test_dataset/ground_truth_atomic_facts.json"

DECONTEXT_FAISS = (
    PROJECT_ROOT / "benchmark_rag" / "components" / "decontextualizers"
    / "batch_output" / "embeddings" / "decontext_statements.faiss"
)
DECONTEXT_META = (
    PROJECT_ROOT / "benchmark_rag" / "components" / "decontextualizers"
    / "batch_output" / "embeddings" / "decontext_statements_meta.pkl"
)

GENERATOR_MODEL = "gemini-2.5-flash"
JUDGE_MODEL = "gemini-2.5-pro"
TOKEN_BUDGET = 128_000
CHARS_PER_TOKEN = 4
RERANK_K = 25
RETRIEVE_CANDIDATES = 100

JUDGE_TEMPERATURE = 0.0
GROUNDEDNESS_MAX_OUTPUT_TOKENS = 2048
ATOMIC_MAX_OUTPUT_TOKENS = 16384
MAX_JSONL_BYTES = 2 * 1024**3

PRO_INPUT_PRICE = 1.25 / 1_000_000
PRO_OUTPUT_PRICE = 10.0 / 1_000_000
EST_OUTPUT_TOKENS_PER_JUDGE = 500


# ── Prompts ──

GROUNDEDNESS_JUDGE_SYSTEM = (
    "You are an impartial judge evaluating whether a statement from a generated "
    "legal answer is supported by evidence from Canadian legal documents."
)

GROUNDEDNESS_JUDGE_PROMPT = """\
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

ATOMIC_JUDGE_SYSTEM = (
    "You are an impartial judge comparing atomic facts between a generated answer "
    "and ground-truth reference facts for a Canadian legal question."
)

ATOMIC_JUDGE_PROMPT = """\
Compare the generated facts against the ground truth facts.

For each generated fact, determine if it is present (semantically equivalent) in the ground truth.
For each ground truth fact, determine if it is present (semantically equivalent) in the generated answer.

generated_facts:
{generated_facts}

ground_truth_facts:
{ground_truth_facts}

Respond with a JSON object:
{{
  "generated_fact_results": [
    {{"fact": "...", "in_ground_truth": true/false}}
  ],
  "ground_truth_fact_results": [
    {{"fact": "...", "in_generated": true/false}}
  ]
}}\
"""


# ── Helpers ──

def _token_estimate(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


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


def build_context(chunks, doc_texts, token_budget):
    doc_ids = list(dict.fromkeys(c.doc_id for c in chunks))
    full_docs = {did: doc_texts[did] for did in doc_ids if did in doc_texts}
    total_doc_tokens = sum(_token_estimate(t) for t in full_docs.values())

    if total_doc_tokens <= token_budget:
        parts = [f"=== Document: {did} ===\n{full_docs[did]}" for did in doc_ids if did in full_docs]
        return "\n\n".join(parts), {"context_mode": "full_documents", "num_docs": len(full_docs)}

    chunk_tokens = sum(_token_estimate(c.text) for c in chunks)
    remaining = token_budget - chunk_tokens

    if remaining <= 0:
        parts = [f"[{i}] ({c.doc_id})\n{c.text}" for i, c in enumerate(chunks, 1)]
        return "\n\n".join(parts), {"context_mode": "chunks_only"}

    ratio = remaining / total_doc_tokens
    chunk_parts = [f"[{i}] ({c.doc_id})\n{c.text}" for i, c in enumerate(chunks, 1)]
    doc_parts = []
    for did in doc_ids:
        if did not in full_docs:
            continue
        truncated_chars = int(len(full_docs[did]) * ratio)
        doc_parts.append(f"=== Document: {did} (truncated) ===\n{full_docs[did][:truncated_chars]}")

    return (
        "RETRIEVED PASSAGES:\n" + "\n\n".join(chunk_parts) +
        "\n\nFULL DOCUMENT CONTEXT (truncated):\n" + "\n\n".join(doc_parts)
    ), {"context_mode": "chunks_and_truncated_docs", "truncation_ratio": round(ratio, 4)}


def _safe_key(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', s)


def _write_jsonl(path: Path, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def _try_parse_json(text: str):
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None


# ===========================================================================
# generate
# ===========================================================================

def cmd_generate(args: argparse.Namespace) -> None:
    import faiss
    import numpy as np
    import pandas as pd
    from tqdm import tqdm
    from google import genai
    from google.genai import types
    from benchmark_rag.components.base import EmbeddedChunk, RetrievedChunk
    from benchmark_rag.components.embedders.qwen import QwenEmbedder
    from benchmark_rag.components.generators.gemini import _generate_with_retry
    from benchmark_rag.components.rerankers.kanon2 import Kanon2Reranker
    from benchmark_rag.config.schemas import ExperimentConfig
    from benchmark_rag.prompts.answer_generator import ANSWER_SYSTEM_PROMPT

    cfg = ExperimentConfig.from_yaml(PROJECT_ROOT / INDEX_CONFIG)

    index_dir = PROJECT_ROOT / cfg.indexing.output_dir
    faiss_index = faiss.read_index(str(index_dir / "index.faiss"))
    with open(index_dir / "index.chunks.pkl", "rb") as f:
        chunks: list[EmbeddedChunk] = pickle.load(f)
    print(f"Loaded FAISS index: {faiss_index.ntotal} vectors, {len(chunks)} chunks")

    doc_to_indices: dict[str, list[int]] = {}
    for i, c in enumerate(chunks):
        doc_to_indices.setdefault(c.doc_id, []).append(i)

    embedder = QwenEmbedder(model_name="Qwen/Qwen3-Embedding-8B", device="cuda:0")
    reranker = Kanon2Reranker(model_name="kanon-2-reranker", batch_size=100)

    queries = json.loads((PROJECT_ROOT / QUERIES_PATH).read_text())
    doc_texts = dict(zip(
        *pd.read_parquet(PROJECT_ROOT / DATASET_PATH, columns=["citation", "text"]).values.T
    ))

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "query_results.jsonl"
    rows: list[dict] = []
    checkpoint_interval = max(1, len(queries) // 5)

    failed_queries: list[dict] = []

    for qi, q in enumerate(tqdm(queries, desc="Oracle generate")):
        query_text = str(q.get("query_text", ""))
        province = q.get("province", "")
        if province:
            query_text = f"I am in {province}. {query_text}"
        gold_citations = list(q.get("ground_truth_citations", []))
        if not query_text.strip() or not gold_citations:
            continue

        try:
            gt_indices = []
            for cit in gold_citations:
                gt_indices.extend(doc_to_indices.get(cit, []))

            if not gt_indices:
                rows.append({
                    "query_id": q["query_id"],
                    "query_text": query_text,
                    "gold_citations": gold_citations,
                    "retrieved_ids": [],
                    "retrieved_chunk_details": [],
                    "answer": None,
                    "note": "no GT documents found in index",
                })
                continue

            query_emb = np.array(embedder.embed([query_text]), dtype=np.float32)
            faiss.normalize_L2(query_emb)

            id_array = np.array(gt_indices, dtype=np.int64)
            selector = faiss.IDSelectorBatch(id_array)
            params = faiss.SearchParameters(sel=selector)
            k = min(RETRIEVE_CANDIDATES, len(gt_indices))
            scores, indices = faiss_index.search(query_emb, k, params=params)

            candidates = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                c = chunks[idx]
                candidates.append(RetrievedChunk(
                    text=c.text, doc_id=c.doc_id, chunk_idx=c.chunk_idx,
                    metadata=c.metadata, embedding=None, score=float(score),
                ))

            if candidates:
                try:
                    reranked = reranker.rerank(query_text, candidates)[:RERANK_K]
                except Exception as e:
                    log.warning(f"Rerank failed for query {q['query_id']}: {e}")
                    reranked = candidates[:RERANK_K]
            else:
                reranked = []

            answer = None
            ctx_meta = {}
            if reranked:
                context, ctx_meta = build_context(reranked, doc_texts, TOKEN_BUDGET)
                prompt = f"Context:\n{context}\n\nQuestion: {query_text}"
                response = _generate_with_retry(
                    client, model=GENERATOR_MODEL, contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=ANSWER_SYSTEM_PROMPT,
                        temperature=0.0, max_output_tokens=16384,
                    ),
                )
                answer = response.text

            rows.append({
                "query_id": q["query_id"],
                "query_text": query_text,
                "gold_citations": gold_citations,
                "retrieved_ids": [c.doc_id for c in reranked],
                "retrieved_chunk_details": [
                    {"doc_id": c.doc_id, "chunk_idx": c.chunk_idx, "score": round(c.score, 6)}
                    for c in reranked
                ],
                "num_unique_docs_retrieved": len(set(c.doc_id for c in reranked)),
                "answer": answer,
                "context_meta": ctx_meta,
            })

        except Exception as exc:
            log.error(
                "Query %s failed (%s: %s) — skipping",
                q.get("query_id"), type(exc).__name__, exc,
            )
            failed_queries.append({
                "query_id": q.get("query_id"),
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        if (qi + 1) % checkpoint_interval == 0:
            with open(results_file, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            print(f"  Checkpoint at {qi+1}")

    with open(results_file, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    if failed_queries:
        failed_file = RESULTS_DIR / "failed_queries.json"
        failed_file.write_text(json.dumps(failed_queries, indent=2))
        print(f"  {len(failed_queries)} failed queries saved to {failed_file}")

    n_with_answer = sum(1 for r in rows if r.get("answer"))
    print(f"\nSaved {len(rows)} results ({n_with_answer} with answers) to {results_file}")
    reranker.log_usage_summary()


# ===========================================================================
# prepare-eval
# ===========================================================================

def cmd_prepare_eval(args: argparse.Namespace) -> None:
    atomic_only = getattr(args, "atomic_only", False)

    if atomic_only:
        _prepare_atomic_only()
        return

    import faiss
    import numpy as np
    from tqdm import tqdm
    from benchmark_rag.components.decontextualizers.gemini_decontextualizer import GeminiDecontextualizer
    from benchmark_rag.components.embedders.qwen import QwenEmbedder
    from benchmark_rag.prompts.atomic_facts import DECOMPOSE_SYSTEM_PROMPT

    results_file = RESULTS_DIR / "query_results.jsonl"
    if not results_file.exists():
        sys.exit("Run 'generate' first.")

    with open(results_file) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    rows_with_answers = [r for r in rows if r.get("answer")]
    print(f"Loaded {len(rows_with_answers)} answers")

    gt_facts = json.loads((PROJECT_ROOT / ATOMIC_FACTS_PATH).read_text())

    print("Loading decontextualized corpus FAISS...")
    corpus_index = faiss.read_index(str(DECONTEXT_FAISS))
    with open(DECONTEXT_META, "rb") as f:
        corpus_meta = pickle.load(f)
    print(f"  {corpus_index.ntotal} statements")

    embedder = QwenEmbedder(model_name="Qwen/Qwen3-Embedding-8B", device="cuda:0")
    decontextualizer = GeminiDecontextualizer(model_name="gemini-2.5-flash")

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    from google import genai
    from google.genai import types
    from benchmark_rag.components.generators.gemini import _generate_with_retry as _retry
    client = genai.Client(api_key=api_key)

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    groundedness_lines: list[str] = []
    atomic_lines: list[str] = []
    intermediate: list[dict] = []
    checkpoint_interval = max(1, len(rows_with_answers) // 5)

    for ri, row in enumerate(tqdm(rows_with_answers, desc="Preparing eval")):
        qid = str(row["query_id"])
        answer = row["answer"]
        statements = split_into_statements(answer)
        if not statements:
            continue

        # --- Groundedness: decontextualize + embed + search ---
        decontext_result = decontextualizer.decontextualize(statements, answer)
        decontext_stmts = decontext_result if decontext_result else statements

        embeddings = embedder.embed(decontext_stmts)
        query_vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(query_vectors)
        scores_matrix, indices_matrix = corpus_index.search(query_vectors, 25)

        stmt_data = []
        for si, (stmt, d_stmt) in enumerate(zip(statements, decontext_stmts)):
            evidence_parts = []
            for rank, idx in enumerate(indices_matrix[si]):
                if idx < 0:
                    continue
                meta = corpus_meta[idx]
                evidence_parts.append(f"[{rank+1}] ({meta['citation']})\n{meta['text']}")
            evidence_text = "\n\n".join(evidence_parts)

            prompt = GROUNDEDNESS_JUDGE_PROMPT.format(statement=d_stmt, evidence=evidence_text)
            key = f"ground_{_safe_key(qid)}_s{si}"

            groundedness_lines.append(json.dumps({
                "key": key,
                "request": {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "system_instruction": {"parts": [{"text": GROUNDEDNESS_JUDGE_SYSTEM}]},
                    "generation_config": {
                        "temperature": JUDGE_TEMPERATURE,
                        "max_output_tokens": GROUNDEDNESS_MAX_OUTPUT_TOKENS,
                    },
                },
            }, ensure_ascii=False))

            stmt_data.append({
                "original": stmt,
                "decontextualized": d_stmt,
                "batch_key": key,
            })

        # --- Atomic facts: decompose + prepare comparison ---
        try:
            response = _retry(
                client, model="gemini-2.5-flash", contents=f"Decompose:\n\n{answer}",
                config=types.GenerateContentConfig(
                    system_instruction=DECOMPOSE_SYSTEM_PROMPT,
                    temperature=0.0, response_mime_type="application/json",
                ),
            )
            gen_facts = json.loads(response.text)
        except Exception as e:
            log.warning(f"Decompose failed for {qid}: {e}")
            gen_facts = []

        gt_qid_facts = gt_facts.get(qid, gt_facts.get(str(qid), []))

        if gen_facts and gt_qid_facts:
            atomic_prompt = ATOMIC_JUDGE_PROMPT.format(
                generated_facts=json.dumps(gen_facts, ensure_ascii=False),
                ground_truth_facts=json.dumps(gt_qid_facts, ensure_ascii=False),
            )
            atomic_key = f"atomic_{_safe_key(qid)}"
            atomic_lines.append(json.dumps({
                "key": atomic_key,
                "request": {
                    "contents": [{"parts": [{"text": atomic_prompt}]}],
                    "system_instruction": {"parts": [{"text": ATOMIC_JUDGE_SYSTEM}]},
                    "generation_config": {
                        "temperature": JUDGE_TEMPERATURE,
                        "max_output_tokens": ATOMIC_MAX_OUTPUT_TOKENS,
                        "response_mime_type": "application/json",
                    },
                },
            }, ensure_ascii=False))

        intermediate.append({
            "query_id": qid,
            "query_text": row["query_text"],
            "statements": stmt_data,
            "generated_facts": gen_facts,
            "gt_facts_count": len(gt_qid_facts),
        })

        if (ri + 1) % checkpoint_interval == 0:
            with open(BATCH_DIR / "intermediate_checkpoint.json", "w") as f:
                json.dump(intermediate, f, ensure_ascii=False)
            log.info(f"Checkpoint at {ri+1}/{len(rows_with_answers)}")

    _write_jsonl(BATCH_DIR / "groundedness_requests.jsonl", groundedness_lines)
    _write_jsonl(BATCH_DIR / "atomic_requests.jsonl", atomic_lines)

    with open(BATCH_DIR / "intermediate.json", "w") as f:
        json.dump(intermediate, f, indent=2, ensure_ascii=False)

    decontext_path = RESULTS_DIR / "decontextualized_answers.json"
    decontext_path.write_text(json.dumps(intermediate, indent=2, ensure_ascii=False))

    total_requests = len(groundedness_lines) + len(atomic_lines)
    est_in_tokens = sum(len(l) for l in groundedness_lines + atomic_lines) // 4
    est_out_tokens = total_requests * EST_OUTPUT_TOKENS_PER_JUDGE
    est_cost = est_in_tokens * PRO_INPUT_PRICE + est_out_tokens * PRO_OUTPUT_PRICE

    manifest = {
        "created": datetime.now().isoformat(),
        "groundedness_requests": len(groundedness_lines),
        "atomic_requests": len(atomic_lines),
        "total_requests": total_requests,
        "est_input_tokens": est_in_tokens,
        "est_output_tokens": est_out_tokens,
        "est_cost_usd": round(est_cost, 2),
        "judge_model": JUDGE_MODEL,
    }
    with open(BATCH_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    decontextualizer.log_usage_summary()
    print(f"\nPrepared {len(groundedness_lines)} groundedness + {len(atomic_lines)} atomic requests")
    print(f"Estimated batch cost: ${est_cost:.2f}")
    print(f"Files saved to {BATCH_DIR}")


# ===========================================================================
# submit
# ===========================================================================

def cmd_submit(args: argparse.Namespace) -> None:
    from google import genai
    from google.genai import types

    manifest_path = BATCH_DIR / "manifest.json"
    if not manifest_path.exists():
        sys.exit("Run 'prepare-eval' first.")

    client = genai.Client()
    jobs = []

    for fname in ["groundedness_requests.jsonl", "atomic_requests.jsonl"]:
        fpath = BATCH_DIR / fname
        if not fpath.exists() or fpath.stat().st_size == 0:
            print(f"Skipping {fname} (empty or missing)")
            continue

        print(f"Uploading {fname}...")
        uploaded = client.files.upload(
            file=str(fpath),
            config=types.UploadFileConfig(display_name=fname, mime_type="jsonl"),
        )
        print(f"  Uploaded: {uploaded.name}")

        job = client.batches.create(
            model=JUDGE_MODEL,
            src=uploaded.name,
            config={"display_name": f"oracle_{fname}"},
        )
        print(f"  Job: {job.name}")
        jobs.append({
            "name": job.name,
            "file": fname,
            "uploaded": uploaded.name,
            "submitted": datetime.now().isoformat(),
        })

    with open(BATCH_DIR / "jobs.json", "w") as f:
        json.dump(jobs, f, indent=2)
    print(f"\n{len(jobs)} batch job(s) submitted.")


# ===========================================================================
# status
# ===========================================================================

def cmd_status(args: argparse.Namespace) -> None:
    from google import genai

    jobs_path = BATCH_DIR / "jobs.json"
    if not jobs_path.exists():
        sys.exit("Run 'submit' first.")

    jobs = json.loads(jobs_path.read_text())
    client = genai.Client()

    for j in jobs:
        job = client.batches.get(name=j["name"])
        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        print(f"  {j['file']:40s} {state}")


# ===========================================================================
# collect
# ===========================================================================

def cmd_collect(args: argparse.Namespace) -> None:
    from google import genai
    import numpy as np

    jobs_path = BATCH_DIR / "jobs.json"
    intermediate_path = BATCH_DIR / "intermediate.json"
    if not jobs_path.exists() or not intermediate_path.exists():
        sys.exit("Run 'submit' and 'prepare-eval' first.")

    jobs = json.loads(jobs_path.read_text())
    intermediate = json.loads(intermediate_path.read_text())
    gt_facts = json.loads((PROJECT_ROOT / ATOMIC_FACTS_PATH).read_text())
    client = genai.Client()

    all_results: dict[str, dict] = {}
    for j in jobs:
        job = client.batches.get(name=j["name"])
        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        if state != "JOB_STATE_SUCCEEDED":
            print(f"WARNING: {j['file']} state={state}, skipping")
            continue

        if not job.dest or not job.dest.file_name:
            print(f"WARNING: {j['file']} no output file")
            continue

        print(f"Downloading {j['file']} results...")
        result_data = client.files.download(file=job.dest.file_name)
        text = result_data.decode("utf-8") if isinstance(result_data, bytes) else str(result_data)

        raw_path = BATCH_DIR / f"raw_results_{j['file']}"
        raw_path.write_text(text, encoding="utf-8")

        for line in text.strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            key = entry["key"]
            try:
                resp_text = entry["response"]["candidates"][0]["content"]["parts"][0]["text"]
                all_results[key] = {"text": resp_text, "success": True}
            except (KeyError, IndexError, TypeError) as e:
                all_results[key] = {"text": None, "success": False, "error": str(e)}

    # --- Process groundedness ---
    groundedness_rows = []
    total_stmts = 0
    total_supported = 0
    total_partial = 0
    total_not_supported = 0

    for entry in intermediate:
        qid = entry["query_id"]
        q_grounded = 0
        q_partial = 0
        stmt_results = []

        for sd in entry["statements"]:
            key = sd["batch_key"]
            result = all_results.get(key, {})
            judge_text = result.get("text", "") or ""
            verdict = parse_verdict(judge_text) if judge_text else "MISSING"

            if verdict in ("SUPPORTED", "PARTIAL"):
                q_grounded += 1
            if verdict == "PARTIAL":
                q_partial += 1

            stmt_results.append({
                "original": sd["original"],
                "decontextualized": sd["decontextualized"],
                "verdict": verdict,
            })

        n = len(entry["statements"])
        total_stmts += n
        total_supported += q_grounded - q_partial
        total_partial += q_partial
        total_not_supported += n - q_grounded

        groundedness = q_grounded / n if n else 0.0
        groundedness_rows.append({
            "query_id": qid,
            "num_statements": n,
            "supported": q_grounded - q_partial,
            "partial": q_partial,
            "not_supported": n - q_grounded,
            "groundedness_score": round(groundedness, 4),
            "statements": stmt_results,
        })

    # --- Process atomic facts ---
    atomic_rows = []
    precision_scores = []
    recall_scores = []

    for entry in intermediate:
        qid = entry["query_id"]
        atomic_key = f"atomic_{_safe_key(qid)}"
        result = all_results.get(atomic_key, {})

        if not result.get("success"):
            continue

        parsed = _try_parse_json(result["text"])
        if not parsed:
            continue

        gen_results = parsed.get("generated_fact_results", [])
        gt_results = parsed.get("ground_truth_fact_results", [])

        precision = (
            sum(1 for r in gen_results if r.get("in_ground_truth")) / len(gen_results)
            if gen_results else 0.0
        )
        recall = (
            sum(1 for r in gt_results if r.get("in_generated")) / len(gt_results)
            if gt_results else 0.0
        )
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)

        atomic_rows.append({
            "query_id": qid,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "num_generated_facts": len(gen_results),
            "num_gt_facts": len(gt_results),
        })

    # --- Save ---
    with open(RESULTS_DIR / "groundedness_eval.jsonl", "w") as f:
        for r in groundedness_rows:
            f.write(json.dumps(r) + "\n")

    with open(RESULTS_DIR / "atomic_eval.jsonl", "w") as f:
        for r in atomic_rows:
            f.write(json.dumps(r) + "\n")

    g_scores = [r["groundedness_score"] for r in groundedness_rows]
    total_grounded = total_supported + total_partial
    overall_groundedness = total_grounded / total_stmts if total_stmts else 0.0
    avg_p = float(np.mean(precision_scores)) if precision_scores else 0.0
    avg_r = float(np.mean(recall_scores)) if recall_scores else 0.0
    avg_f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "groundedness": {
            "num_queries": len(groundedness_rows),
            "total_statements": total_stmts,
            "total_supported": total_supported,
            "total_partial": total_partial,
            "total_not_supported": total_not_supported,
            "overall_groundedness": round(overall_groundedness, 4),
            "mean_per_query": round(float(np.mean(g_scores)), 4) if g_scores else 0.0,
            "median_per_query": round(float(np.median(g_scores)), 4) if g_scores else 0.0,
            "per_query_scores": [{"query_id": r["query_id"], "score": r["groundedness_score"]} for r in groundedness_rows],
        },
        "atomic_facts": {
            "num_queries": len(atomic_rows),
            "mean_precision": round(avg_p, 4),
            "mean_recall": round(avg_r, 4),
            "mean_f1": round(avg_f1, 4),
        },
    }
    with open(RESULTS_DIR / "oracle_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Oracle Experiment Results — {EXPERIMENT_ID}")
    print(f"{'='*60}")
    print(f"\nGroundedness (SUPPORTED + PARTIAL = grounded):")
    print(f"  Queries:     {len(groundedness_rows)}")
    print(f"  Statements:  {total_stmts}")
    print(f"  Grounded:    {total_grounded} ({100*overall_groundedness:.1f}%)")
    print(f"    Supported: {total_supported}")
    print(f"    Partial:   {total_partial}")
    print(f"  Not supp:    {total_not_supported}")
    print(f"  Mean/query:  {np.mean(g_scores):.4f}")
    print(f"\nAtomic Facts:")
    print(f"  Queries:     {len(atomic_rows)}")
    print(f"  Precision:   {avg_p:.4f}")
    print(f"  Recall:      {avg_r:.4f}")
    print(f"  F1:          {avg_f1:.4f}")
    print(f"{'='*60}")


# ===========================================================================
# atomic-only helpers
# ===========================================================================

def _prepare_atomic_only() -> None:
    """Regenerate atomic_requests.jsonl from existing intermediate.json."""
    intermediate_path = BATCH_DIR / "intermediate.json"
    if not intermediate_path.exists():
        sys.exit("No intermediate.json found. Run full 'prepare-eval' first.")

    intermediate = json.loads(intermediate_path.read_text())
    gt_facts = json.loads((PROJECT_ROOT / ATOMIC_FACTS_PATH).read_text())

    atomic_lines: list[str] = []
    for entry in intermediate:
        qid = entry["query_id"]
        gen_facts = entry.get("generated_facts", [])
        gt_qid_facts = gt_facts.get(str(qid), gt_facts.get(qid, []))

        if not gen_facts or not gt_qid_facts:
            continue

        atomic_prompt = ATOMIC_JUDGE_PROMPT.format(
            generated_facts=json.dumps(gen_facts, ensure_ascii=False),
            ground_truth_facts=json.dumps(gt_qid_facts, ensure_ascii=False),
        )
        atomic_key = f"atomic_{_safe_key(str(qid))}"
        atomic_lines.append(json.dumps({
            "key": atomic_key,
            "request": {
                "contents": [{"parts": [{"text": atomic_prompt}]}],
                "system_instruction": {"parts": [{"text": ATOMIC_JUDGE_SYSTEM}]},
                "generation_config": {
                    "temperature": JUDGE_TEMPERATURE,
                    "max_output_tokens": ATOMIC_MAX_OUTPUT_TOKENS,
                    "response_mime_type": "application/json",
                },
            },
        }, ensure_ascii=False))

    _write_jsonl(BATCH_DIR / "atomic_requests.jsonl", atomic_lines)
    print(f"Regenerated {len(atomic_lines)} atomic requests → {BATCH_DIR / 'atomic_requests.jsonl'}")


def cmd_submit_atomic(args: argparse.Namespace) -> None:
    """Submit only the atomic batch job."""
    from google import genai
    from google.genai import types

    fpath = BATCH_DIR / "atomic_requests.jsonl"
    if not fpath.exists() or fpath.stat().st_size == 0:
        sys.exit("No atomic_requests.jsonl found. Run 'prepare-eval --atomic-only' first.")

    client = genai.Client()

    print(f"Uploading atomic_requests.jsonl...")
    uploaded = client.files.upload(
        file=str(fpath),
        config=types.UploadFileConfig(display_name="atomic_requests.jsonl", mime_type="jsonl"),
    )
    print(f"  Uploaded: {uploaded.name}")

    job = client.batches.create(
        model=JUDGE_MODEL,
        src=uploaded.name,
        config={"display_name": "oracle_atomic_requests.jsonl"},
    )
    print(f"  Job: {job.name}")

    # Update jobs.json — keep groundedness job, replace atomic
    jobs_path = BATCH_DIR / "jobs.json"
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []
    jobs = [j for j in jobs if j["file"] != "atomic_requests.jsonl"]
    jobs.append({
        "name": job.name,
        "file": "atomic_requests.jsonl",
        "uploaded": uploaded.name,
        "submitted": datetime.now().isoformat(),
    })
    with open(jobs_path, "w") as f:
        json.dump(jobs, f, indent=2)
    print(f"Updated {jobs_path}")


# ===========================================================================
# main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Oracle experiment: GT-only retrieval + answer eval")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("generate", help="Retrieve (GT-only) + rerank + generate answers")
    p_prep = sub.add_parser("prepare-eval", help="Prepare batch JSONL for Pro judging")
    p_prep.add_argument("--atomic-only", action="store_true",
                        help="Only regenerate atomic_requests.jsonl from existing intermediate data")
    sub.add_parser("submit", help="Submit batch jobs to Gemini API")
    sub.add_parser("submit-atomic", help="Submit only the atomic batch job")
    sub.add_parser("status", help="Check batch job status")
    sub.add_parser("collect", help="Download results and compute metrics")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    {"generate": cmd_generate, "prepare-eval": cmd_prepare_eval,
     "submit": cmd_submit, "submit-atomic": cmd_submit_atomic,
     "status": cmd_status, "collect": cmd_collect}[args.command](args)


if __name__ == "__main__":
    main()
