"""
Generate answers by sending requests to a vLLM server.

Reads shared_retrieval.jsonl, builds prompts, sends to a running vLLM
server (started by serve_qwen.sbatch or serve_gemma.sbatch), and saves
answers. The server stays up between runs so model loading (~20 min)
only happens once.

Usage:
    python scripts/multi_gen_client.py --generator qwen
    python scripts/multi_gen_client.py --generator gemma

    # Override server location:
    python scripts/multi_gen_client.py --generator qwen --host node01 --port 8100

Requires: a running vLLM server (see slurm/multi_gen/serve_*.sbatch)
Requires: runs/multi_generator_50/results/shared_retrieval.jsonl
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

from tqdm import tqdm

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "runs" / "multi_generator_50" / "results"
RETRIEVAL_PATH = RESULTS_DIR / "shared_retrieval.jsonl"

MAX_INPUT_TOKENS = 123_000
CONTEXT_TOKEN_BUDGET = 120_000
CHARS_PER_TOKEN = 4
MAX_INPUT_CHARS = MAX_INPUT_TOKENS * CHARS_PER_TOKEN

_INSTRUCTION_PREFIX = """\
Question: {query_text}

Answer the question using ONLY the documents provided below. Structure your \
answer with: 1. Opening Statements, 2. Supporting Arguments, 3. Final Conclusion. \
Cite sources using the exact citation string shown in each document header. \
If the provided documents are insufficient to fully answer the question, \
explicitly note what information is missing and why it matters.\
"""

_INSTRUCTION_SUFFIX = """\

Reminder: Answer the above question using ONLY the provided documents. \
Do not invent information. If the documents are insufficient, note what \
specific information is missing and how it would affect the answer.
Question: {query_text}\
"""


def _token_estimate(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def build_context(chunk_details, gt_doc_texts, token_budget):
    doc_ids = list(dict.fromkeys(c["doc_id"] for c in chunk_details))
    full_docs = {did: gt_doc_texts[did] for did in doc_ids if did in gt_doc_texts}
    total_doc_tokens = sum(_token_estimate(t) for t in full_docs.values())

    if total_doc_tokens <= token_budget:
        parts = [f"=== Document: {did} ===\n{full_docs[did]}" for did in doc_ids if did in full_docs]
        return "\n\n".join(parts), {"context_mode": "full_documents", "num_docs": len(full_docs), "total_doc_tokens": total_doc_tokens}

    chunk_tokens = sum(_token_estimate(c["text"]) for c in chunk_details)
    remaining = token_budget - chunk_tokens
    if remaining <= 0:
        parts = [f"[{i}] ({c['doc_id']})\n{c['text']}" for i, c in enumerate(chunk_details, 1)]
        return "\n\n".join(parts), {"context_mode": "chunks_only"}

    ratio = remaining / total_doc_tokens
    chunk_parts = [f"[{i}] ({c['doc_id']})\n{c['text']}" for i, c in enumerate(chunk_details, 1)]
    doc_parts = [f"=== Document: {did} (truncated) ===\n{full_docs[did][:int(len(full_docs[did]) * ratio)]}"
                 for did in doc_ids if did in full_docs]
    return ("RETRIEVED PASSAGES:\n" + "\n\n".join(chunk_parts) +
            "\n\nFULL DOCUMENT CONTEXT (truncated):\n" + "\n\n".join(doc_parts)
    ), {"context_mode": "chunks_and_truncated_docs", "truncation_ratio": round(ratio, 4)}


def _build_prompt(query_text, context):
    return f"{_INSTRUCTION_PREFIX.format(query_text=query_text)}\n\n{context}\n\n{_INSTRUCTION_SUFFIX.format(query_text=query_text)}"


def wait_for_server(base_url, timeout=1200):
    """Poll the server health endpoint until it's ready."""
    import urllib.request
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{base_url}/health", timeout=5)
            return True
        except Exception:
            time.sleep(10)
    return False


def main():
    parser = argparse.ArgumentParser(description="Send generation requests to a vLLM server.")
    parser.add_argument("--generator", required=True, choices=["qwen", "gemma"])
    parser.add_argument("--host", default=None, help="Server hostname (default: read from server info file)")
    parser.add_argument("--port", type=int, default=None, help="Server port (default: read from server info file)")
    parser.add_argument("--max-tokens", type=int, default=8000, help="Max output tokens (default: 8000)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    if not RETRIEVAL_PATH.exists():
        sys.exit(f"Run multi_gen_retrieve.py first. Missing: {RETRIEVAL_PATH}")

    with open(RETRIEVAL_PATH) as f:
        retrieval_data = [json.loads(l) for l in f if l.strip()]
    log.info(f"Loaded {len(retrieval_data)} queries")

    # Resolve server address
    if args.host and args.port:
        host, port = args.host, args.port
    else:
        server_file = PROJECT_ROOT / "runs" / "multi_generator_50" / f"{args.generator}_server.json"
        if not server_file.exists():
            sys.exit(f"No server info at {server_file}. Start the server first or pass --host/--port.")
        info = json.loads(server_file.read_text())
        host = args.host or info["host"]
        port = args.port or info["port"]
        model_name = info["model"]
        log.info(f"Server: {host}:{port} model={model_name}")

    base_url = f"http://{host}:{port}/v1"

    log.info(f"Waiting for server at {host}:{port}...")
    if not wait_for_server(f"http://{host}:{port}"):
        sys.exit(f"Server at {host}:{port} not responding after 10 minutes.")
    log.info("Server is ready.")

    from openai import OpenAI
    from benchmark_rag.prompts.answer_generator import ANSWER_SYSTEM_PROMPT

    client = OpenAI(base_url=base_url, api_key="unused")

    # Get model name from server
    models = client.models.list()
    model_id = models.data[0].id
    log.info(f"Using model: {model_id}")

    gen_name = args.generator
    output_file = RESULTS_DIR / f"{gen_name}_answers.jsonl"

    # Checkpoint resume
    completed_ids = set()
    existing_rows = []
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("answer"):
                    completed_ids.add(rec["query_id"])
                    existing_rows.append(rec)
        if completed_ids:
            log.info(f"Resuming: {len(completed_ids)} already completed")

    with open(output_file, "w") as out_f:
        for rec in existing_rows:
            out_f.write(json.dumps(rec) + "\n")

        for qc in tqdm(retrieval_data, desc=f"Generating ({gen_name})"):
            qid = qc["query_id"]
            if qid in completed_ids:
                continue

            query_text = qc["query_text"]
            chunk_details = qc.get("retrieved_chunk_details", [])
            gt_doc_texts = qc.get("gt_doc_texts", {})

            if not chunk_details:
                row = {"query_id": qid, "query_text": query_text, "gold_citations": qc["gold_citations"],
                       "retrieved_chunk_details": chunk_details, "retrieved_documents": qc.get("retrieved_documents", []),
                       "context_meta": {}, "generator": gen_name, "answer": None, "usage": {},
                       "note": "no chunks retrieved"}
                out_f.write(json.dumps(row) + "\n"); out_f.flush()
                continue

            context, ctx_meta = build_context(chunk_details, gt_doc_texts, CONTEXT_TOKEN_BUDGET)
            prompt = _build_prompt(query_text, context)

            if len(prompt) > MAX_INPUT_CHARS:
                log.warning(f"Query {qid}: prompt too long, skipping")
                row = {"query_id": qid, "query_text": query_text, "gold_citations": qc["gold_citations"],
                       "retrieved_chunk_details": chunk_details, "retrieved_documents": qc.get("retrieved_documents", []),
                       "context_meta": ctx_meta, "generator": gen_name, "answer": None,
                       "usage": {"est_input_tokens": len(prompt) // CHARS_PER_TOKEN},
                       "note": f"skipped: prompt exceeds {MAX_INPUT_TOKENS} token limit"}
                out_f.write(json.dumps(row) + "\n"); out_f.flush()
                continue

            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                )
                answer = response.choices[0].message.content
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                }
            except Exception as e:
                log.exception(f"Failed for query {qid}: {type(e).__name__}: {e}")
                answer = None
                usage = {"error": f"{type(e).__name__}: {e}"}

            row = {"query_id": qid, "query_text": query_text, "gold_citations": qc["gold_citations"],
                   "retrieved_chunk_details": chunk_details, "retrieved_documents": qc.get("retrieved_documents", []),
                   "context_meta": ctx_meta, "generator": gen_name, "answer": answer, "usage": usage}
            out_f.write(json.dumps(row) + "\n"); out_f.flush()

    with open(output_file) as f:
        all_rows = [json.loads(l) for l in f if l.strip()]
    n_answered = sum(1 for r in all_rows if r.get("answer"))
    n_skipped = sum(1 for r in all_rows if r.get("note"))
    log.info(f"{gen_name}: {n_answered}/{len(all_rows)} answers, {n_skipped} skipped → {output_file}")


if __name__ == "__main__":
    main()
