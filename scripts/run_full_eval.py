"""
Final paper evaluation: answer generation + atomic-fact judging.

Two conditions (oracle / pipeline) × three generators (gemini / qwen / gemma).
Gemini generation and all judging use the Gemini Batch API.

Subcommands
-----------
    generate      Generate answers (online for local models, batch for Gemini)
    prepare-eval  Decompose answers into atomic facts + build judge batch JSONL
    submit        Submit batch jobs to Gemini Flash
    status        Check batch job status
    collect       Download results, compute per-query precision/recall/F1

Usage (from project root)
-------------------------
    # Oracle condition — Gemini + Qwen now, Gemma later:
    python scripts/run_full_eval.py generate --condition oracle --generators gemini,qwen

    # Pipeline condition:
    python scripts/run_full_eval.py generate --condition pipeline --generators gemini

    # Prepare + submit judging:
    python scripts/run_full_eval.py prepare-eval --condition oracle
    python scripts/run_full_eval.py submit --condition oracle

    # Check status / collect:
    python scripts/run_full_eval.py status --condition oracle
    python scripts/run_full_eval.py collect --condition oracle
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass

from benchmark_rag.cost_logging import (
    DEFAULT_BENCHMARK_COST_CSV,
    append_cost_entry,
)

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_BASE = PROJECT_ROOT / "runs" / "final_eval"
QUERIES_PATH = PROJECT_ROOT / "data" / "test_dataset" / "queries.json"
DATASET_PATH = PROJECT_ROOT / "data" / "test_dataset" / "test_dataset.parquet"
ATOMIC_FACTS_PATH = PROJECT_ROOT / "data" / "test_dataset" / "ground_truth_atomic_facts.json"

ORACLE_INDEX_CONFIG = "configs/experiments/qwen_recursive_8192.yaml"

PIPELINE_CONFIGS = {
    "gemini": "configs/experiments/gemini2_recursive_4096_iterretgen.yaml",
    "qwen": "configs/experiments/qwen_recursive_8192_iterretgen_rerank.yaml",
    "gemma": "configs/experiments/gemma_recursive_8192_rerank.yaml",
}

GENERATOR_MODEL = "gemini-2.5-flash"
JUDGE_MODEL = "gemini-2.5-flash"
TOKEN_BUDGET = 115_000
CHARS_PER_TOKEN = 4
RERANK_K = 25
RETRIEVE_CANDIDATES = 100

DECONTEXT_FAISS = (
    PROJECT_ROOT / "benchmark_rag" / "components" / "decontextualizers"
    / "batch_output" / "embeddings" / "decontext_statements.faiss"
)
DECONTEXT_META = (
    PROJECT_ROOT / "benchmark_rag" / "components" / "decontextualizers"
    / "batch_output" / "embeddings" / "decontext_statements_meta.pkl"
)

GROUNDEDNESS_JUDGE_SYSTEM = (
    "You are an impartial judge evaluating whether a statement from a generated "
    "legal answer is supported by evidence from Canadian legal documents."
)

GROUNDEDNESS_JUDGE_PROMPT = """\
STATEMENT (from a generated answer):
{statement}

EVIDENCE (retrieved from the legal corpus):
{evidence}

First, briefly summarize in 2-4 sentences whether the evidence supports the statement. \
Then give your early verdict as exactly one of:
  VERDICT: SUPPORTED
  VERDICT: NOT_SUPPORTED
  VERDICT: PARTIAL

Now provide your full detailed reasoning: which specific evidence passage(s) support \
or contradict the statement? What claims, if any, go beyond the evidence?

Finally, after your full analysis, give your final verdict as exactly one of:
  FINAL_VERDICT: SUPPORTED
  FINAL_VERDICT: NOT_SUPPORTED
  FINAL_VERDICT: PARTIAL\
"""

_RETRY_DELAYS = [5, 10, 30, 60, 120]

_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.30 / 1_000_000, 0.30 / 1_000_000),
    "gemini-2.5-pro":   (1.25 / 1_000_000, 10.0 / 1_000_000),
}


def _estimate_cost(model: str, in_tok: int, out_tok: int) -> float:
    for prefix, (ip, op) in _PRICING.items():
        if model.startswith(prefix):
            return in_tok * ip + out_tok * op
    return 0.0


class CostTracker:
    def __init__(self, label: str):
        self.label = label
        self.calls: int = 0
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.cost_usd: float = 0.0

    def track(self, model: str, in_tok: int, out_tok: int) -> None:
        self.calls += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.cost_usd += _estimate_cost(model, in_tok, out_tok)

    def summary(self) -> dict:
        return {
            "label": self.label,
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }

    def log_summary(self) -> None:
        log.info(
            "%s cost: calls=%d | input_tokens=%d | output_tokens=%d | $%.6f",
            self.label, self.calls, self.input_tokens, self.output_tokens, self.cost_usd,
        )


# ── Helpers ───────────────────────────────────────────────────────────────

def _token_estimate(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def build_context(chunks, doc_texts: dict[str, str], token_budget: int) -> tuple[str, dict]:
    doc_ids = list(dict.fromkeys(c.doc_id for c in chunks))
    full_docs = {did: doc_texts[did] for did in doc_ids if did in doc_texts}
    total_doc_tokens = sum(_token_estimate(t) for t in full_docs.values())

    if total_doc_tokens <= token_budget:
        parts = [f"=== Document: {did} ===\n{full_docs[did]}" for did in doc_ids if did in full_docs]
        return "\n\n".join(parts), {
            "context_mode": "full_documents",
            "num_docs": len(full_docs),
            "total_doc_tokens": total_doc_tokens,
        }

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
        "RETRIEVED PASSAGES:\n" + "\n\n".join(chunk_parts)
        + "\n\nFULL DOCUMENT CONTEXT (truncated):\n" + "\n\n".join(doc_parts)
    ), {"context_mode": "chunks_and_truncated_docs", "truncation_ratio": round(ratio, 4)}


def _safe_key(s: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(s))


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
    if not judge_text:
        return "MISSING"
    # 1. Check for FINAL_VERDICT in the last 200 chars (best signal — after full reasoning)
    tail = judge_text.strip()[-200:]
    match = re.search(r"FINAL_VERDICT:\s*(SUPPORTED|NOT_SUPPORTED|PARTIAL)", tail, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # 2. Check for FINAL_VERDICT anywhere (in case tail missed it)
    match = re.search(r"FINAL_VERDICT:\s*(SUPPORTED|NOT_SUPPORTED|PARTIAL)", judge_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # 3. Fall back to early VERDICT (truncated before full reasoning finished)
    match = re.search(r"VERDICT:\s*(SUPPORTED|NOT_SUPPORTED|PARTIAL)", judge_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # 4. Last resort: check last line for keyword
    last_line = judge_text.strip().split("\n")[-1].upper()
    if "NOT_SUPPORTED" in last_line or "NOT SUPPORTED" in last_line:
        return "NOT_SUPPORTED"
    if "SUPPORTED" in last_line:
        return "SUPPORTED"
    if "PARTIAL" in last_line:
        return "PARTIAL"
    return "UNKNOWN"


class SafeJSONLWriter:
    """Write JSONL with fsync after each row to prevent NFS corruption."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._f = open(self.path, "w", encoding="utf-8")

    def write_row(self, row: dict) -> None:
        self._f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self) -> None:
        self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


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
            log.warning("Gemini %d — retrying in %ds (attempt %d)...", code, delay, attempt)
            time.sleep(delay)


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


def _load_completed_ids(path: Path) -> tuple[set, list[dict]]:
    completed_ids: set = set()
    existing_rows: list[dict] = []
    if path.exists():
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("generated_answer"):
                    completed_ids.add(rec["query_id"])
                    existing_rows.append(rec)
    return completed_ids, existing_rows


def _results_dir(condition: str) -> Path:
    return RESULTS_BASE / condition


# ── Oracle retrieval ──────────────────────────────────────────────────────

# def _oracle_retrieve(query_text: str, gold_citations: list[str],
#                      faiss_index, chunks, doc_to_indices: dict,
#                      embedder, reranker) -> list:
#     import faiss as faiss_mod
#     import numpy as np
#     from benchmark_rag.components.base import RetrievedChunk

#     gt_indices = []
#     for cit in gold_citations:
#         gt_indices.extend(doc_to_indices.get(cit, []))
#     if not gt_indices:
#         return []

#     query_emb = np.array(embedder.embed([query_text]), dtype=np.float32)
#     faiss_mod.normalize_L2(query_emb)

#     id_array = np.array(gt_indices, dtype=np.int64)
#     selector = faiss_mod.IDSelectorBatch(id_array)
#     params = faiss_mod.SearchParameters(sel=selector)
#     k = min(RETRIEVE_CANDIDATES, len(gt_indices))
#     scores, indices = faiss_index.search(query_emb, k, params=params)

#     candidates = []
#     for score, idx in zip(scores[0], indices[0]):
#         if idx < 0:
#             continue
#         c = chunks[idx]
#         candidates.append(RetrievedChunk(
#             text=c.text, doc_id=c.doc_id, chunk_idx=c.chunk_idx,
#             metadata=c.metadata, embedding=None, score=float(score),
#         ))

#     if candidates and reranker:
#         try:
#             return reranker.rerank(query_text, candidates)[:RERANK_K]
#         except Exception as e:
#             log.warning("Rerank failed: %s", e)
#             return candidates[:RERANK_K]
#     return candidates[:RERANK_K]


# ===========================================================================
# generate
# ===========================================================================

def cmd_generate(args: argparse.Namespace) -> None:
    import pandas as pd
    from tqdm import tqdm
    from benchmark_rag.prompts.answer_generator import ANSWER_SYSTEM_PROMPT

    condition = args.condition
    generators = [g.strip() for g in args.generators.split(",")]
    out_dir = _results_dir(condition)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = json.loads(QUERIES_PATH.read_text())
    doc_texts = dict(zip(
        *pd.read_parquet(DATASET_PATH, columns=["citation", "text"]).values.T
    ))
    log.info("Loaded %d queries, %d documents", len(queries), len(doc_texts))

    # -- Oracle retrieval setup (use pre-computed results) --
    oracle_precomputed: dict[int, dict] = {}
    oracle_chunk_text_lookup: dict[tuple, str] = {}

    if condition == "oracle":
        ORACLE_RESULTS_PATH = PROJECT_ROOT / "runs" / "oracle_qwen_8192_rerank_1k-docs" / "results" / "query_results.jsonl"
        if ORACLE_RESULTS_PATH.exists():
            from benchmark_rag.components.base import RetrievedChunk
            with open(ORACLE_RESULTS_PATH) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    qid = rec.get("query_id")
                    oracle_precomputed[qid] = {
                        "retrieved_ids": rec.get("retrieved_ids", []),
                        "chunk_details": rec.get("retrieved_chunk_details", []),
                    }
            log.info("Loaded pre-computed oracle retrieval for %d queries", len(oracle_precomputed))

            # Load chunk texts from index
            from benchmark_rag.config.schemas import ExperimentConfig
            cfg = ExperimentConfig.from_yaml(PROJECT_ROOT / ORACLE_INDEX_CONFIG)
            index_dir = PROJECT_ROOT / cfg.indexing.output_dir
            chunks_pkl = index_dir / "index.chunks.pkl"
            if chunks_pkl.exists():
                with open(chunks_pkl, "rb") as f:
                    all_chunks = pickle.load(f)
                oracle_chunk_text_lookup = {(c.doc_id, c.chunk_idx): c.text for c in all_chunks}
                log.info("Loaded %d chunk texts for oracle context building", len(oracle_chunk_text_lookup))
        else:
            log.error("No pre-computed oracle retrieval at %s", ORACLE_RESULTS_PATH)
            sys.exit(1)

    # -- Generate per generator --
    for gen_name in generators:
        log.info("=== Generating: condition=%s generator=%s ===", condition, gen_name)
        output_file = out_dir / f"{gen_name}_answers.jsonl"
        completed_ids, existing_rows = _load_completed_ids(output_file)
        if completed_ids:
            log.info("Resuming: %d already completed", len(completed_ids))

        # Build pipeline for pipeline condition — prefer pre-computed results
        pipeline = None
        precomputed_retrieval: dict[int, dict] = {}
        if condition == "pipeline":
            config_path = PIPELINE_CONFIGS.get(gen_name)
            if config_path is None:
                log.warning("No pipeline config for %s — skipping", gen_name)
                continue

            from benchmark_rag.config.schemas import ExperimentConfig
            cfg = ExperimentConfig.from_yaml(PROJECT_ROOT / config_path)
            precomputed_path = PROJECT_ROOT / "runs" / cfg.experiment_id / "results" / "query_results.jsonl"

            if precomputed_path.exists():
                from benchmark_rag.components.base import RetrievedChunk
                with open(precomputed_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        qid = rec.get("query_id")
                        chunks_data = rec.get("retrieved_chunk_details", [])
                        retrieved_ids = rec.get("retrieved_ids", [])
                        precomputed_retrieval[qid] = {
                            "retrieved_ids": retrieved_ids,
                            "chunk_details": chunks_data,
                        }
                log.info("Loaded pre-computed retrieval for %d queries from %s",
                         len(precomputed_retrieval), precomputed_path)

                # Load chunk text from the index for context building
                index_dir = PROJECT_ROOT / cfg.indexing.output_dir
                chunks_pkl = index_dir / "index.chunks.pkl"
                if chunks_pkl.exists():
                    with open(chunks_pkl, "rb") as f:
                        all_chunks = pickle.load(f)
                    chunk_text_lookup = {(c.doc_id, c.chunk_idx): c.text for c in all_chunks}
                    log.info("Loaded %d chunks from %s for text lookup", len(chunk_text_lookup), chunks_pkl)
                else:
                    chunk_text_lookup = {}
                    log.warning("No chunks pickle at %s — will use full doc text as fallback", chunks_pkl)
            else:
                log.info("No pre-computed retrieval at %s — building pipeline live", precomputed_path)
                cfg.generator = None
                is_hybrid = "hybrid" in cfg.retriever.type.lower()
                is_iterretgen = cfg.iterretgen is not None
                if is_iterretgen:
                    from benchmark_rag.pipeline.iterretgen_pipeline import IterRetGenPipeline
                    pipeline = IterRetGenPipeline.from_config(cfg)
                elif is_hybrid:
                    from benchmark_rag.pipeline.hybrid_pipeline import HybridRAGPipeline
                    pipeline = HybridRAGPipeline.from_config(cfg)
                else:
                    from benchmark_rag.pipeline.rag_pipeline import RAGPipeline
                    pipeline = RAGPipeline.from_config(cfg)

        gen_cost = CostTracker(f"generate_{gen_name}_{condition}")

        # Init generator clients
        gemini_client = None
        vllm_client = None
        vllm_model_id = None
        gemma_tokenizer = None
        qwen_model = None
        qwen_tokenizer = None

        if gen_name == "gemini":
            from google import genai
            api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                log.error("GOOGLE_API_KEY not set — skipping gemini")
                continue
            gemini_client = genai.Client(api_key=api_key)
        elif gen_name == "gemma":
            server_file = PROJECT_ROOT / "runs" / "final_eval" / "gemma_server.json"
            if not server_file.exists():
                server_file = PROJECT_ROOT / "runs" / "multi_generator_50" / "gemma_server.json"
            if server_file.exists():
                info = json.loads(server_file.read_text())
                base_url = f"http://{info['host']}:{info['port']}/v1"
                log.info("Waiting for Gemma vLLM server at %s (up to 20 min)...", base_url)
                import urllib.request
                server_ready = False
                for attempt in range(120):
                    try:
                        urllib.request.urlopen(f"http://{info['host']}:{info['port']}/health", timeout=5)
                        server_ready = True
                        break
                    except Exception:
                        time.sleep(10)
                if not server_ready:
                    log.error("Gemma vLLM server not responding after 20 min — skipping")
                    continue
                log.info("Gemma vLLM server is ready.")
                from openai import OpenAI
                from transformers import AutoTokenizer
                vllm_client = OpenAI(base_url=base_url, api_key="unused")
                models = vllm_client.models.list()
                vllm_model_id = models.data[0].id
                gemma_tokenizer = AutoTokenizer.from_pretrained(vllm_model_id)
                log.info("Gemma vLLM model: %s, tokenizer loaded", vllm_model_id)
            else:
                log.error("No gemma_server.json found — start vLLM server first")
                continue
        elif gen_name == "qwen":
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                model_name = "Qwen/Qwen3.5-9B"
                load_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
                if getattr(args, "quantize", False):
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
                    log.info("Loading %s (8-bit quantized)...", model_name)
                else:
                    log.info("Loading %s ...", model_name)
                qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
                qwen_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
                log.info("Qwen loaded. device_map: %s", getattr(qwen_model, 'hf_device_map', 'N/A'))
            except Exception as exc:
                log.error("Failed to load Qwen: %s: %s — skipping", type(exc).__name__, exc, exc_info=True)
                continue

        with SafeJSONLWriter(output_file) as out_f:
            for rec in existing_rows:
                out_f.write_row(rec)

            for q in tqdm(queries, desc=f"{gen_name} ({condition})"):
                qid = q["query_id"]
                if qid in completed_ids:
                    continue

                query_text = str(q.get("query_text", ""))
                province = q.get("province", "")
                if province:
                    query_text = f"I am in {province}. {query_text}"
                gold_citations = list(q.get("ground_truth_citations", []))
                if not query_text.strip() or not gold_citations:
                    continue

                # Retrieve
                try:
                    if condition == "oracle" and qid in oracle_precomputed:
                        from benchmark_rag.components.base import RetrievedChunk
                        pc = oracle_precomputed[qid]
                        retrieved = []
                        for i, doc_id in enumerate(pc["retrieved_ids"][:RERANK_K]):
                            detail = pc["chunk_details"][i] if i < len(pc["chunk_details"]) else {}
                            cidx = detail.get("chunk_idx", i)
                            text = oracle_chunk_text_lookup.get((doc_id, cidx), "")
                            retrieved.append(RetrievedChunk(
                                text=text, doc_id=doc_id, chunk_idx=cidx,
                                metadata={}, score=detail.get("score", 0.0),
                            ))
                    elif condition == "oracle":
                        log.warning("Query %s not in oracle pre-computed results — skipping", qid)
                        continue
                    elif precomputed_retrieval and qid in precomputed_retrieval:
                        from benchmark_rag.components.base import RetrievedChunk
                        pc = precomputed_retrieval[qid]
                        retrieved = []
                        for i, doc_id in enumerate(pc["retrieved_ids"][:RERANK_K]):
                            detail = pc["chunk_details"][i] if i < len(pc["chunk_details"]) else {}
                            cidx = detail.get("chunk_idx", i)
                            text = chunk_text_lookup.get((doc_id, cidx), "")
                            retrieved.append(RetrievedChunk(
                                text=text,
                                doc_id=doc_id,
                                chunk_idx=cidx,
                                metadata={},
                                score=detail.get("score", 0.0),
                            ))
                    elif pipeline is not None:
                        result = pipeline.query(query_text, k=RERANK_K)
                        retrieved = result.retrieved_chunks
                    else:
                        log.warning("Query %s: no retrieval available — skipping", qid)
                        continue
                except Exception as exc:
                    log.error("Query %s retrieval failed: %s: %s", qid, type(exc).__name__, exc, exc_info=True)
                    out_f.write_row({
                        "query_id": qid, "query_text": query_text,
                        "condition": condition, "generator": gen_name,
                        "gold_citations": gold_citations,
                        "generated_answer": None,
                        "error": f"retrieval: {exc}",
                    })
                    continue

                if not retrieved:
                    out_f.write_row({
                        "query_id": qid, "query_text": query_text,
                        "condition": condition, "generator": gen_name,
                        "gold_citations": gold_citations,
                        "generated_answer": None,
                        "note": "no chunks retrieved",
                    })
                    continue

                context, ctx_meta = build_context(retrieved, doc_texts, TOKEN_BUDGET)
                prompt = f"Context:\n{context}\n\nQuestion: {query_text}"

                retrieval_method = (
                    "oracle_qwen_8192_rerank" if condition == "oracle"
                    else PIPELINE_CONFIGS[gen_name].split("/")[-1].replace(".yaml", "")
                )

                # Generate
                import torch
                answer = None
                usage = {}
                try:
                    if gen_name == "gemini":
                        from google.genai import types
                        response = _generate_with_retry(
                            gemini_client, model=GENERATOR_MODEL, contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=ANSWER_SYSTEM_PROMPT,
                                temperature=0.0, max_output_tokens=16384,
                            ),
                        )
                        answer = response.text
                        um = response.usage_metadata
                        usage = {"input_tokens": um.prompt_token_count, "output_tokens": um.candidates_token_count}
                        gen_cost.track(GENERATOR_MODEL, um.prompt_token_count, um.candidates_token_count)
                    elif gen_name == "gemma" and vllm_client is not None:
                        GEMMA_MAX_CTX = 131072
                        gemma_messages = [
                            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ]
                        input_tokens = len(gemma_tokenizer.encode(
                            ANSWER_SYSTEM_PROMPT + "\n" + prompt
                        ))
                        max_out = min(8000, GEMMA_MAX_CTX - input_tokens - 100)
                        if max_out < 100:
                            log.warning("Query %s: Gemma input %d tokens — no room", qid, input_tokens)
                            answer = None
                            usage = {"error": "prompt_exceeds_gemma_context", "input_tokens": input_tokens}
                        else:
                            completion = vllm_client.chat.completions.create(
                                model=vllm_model_id, messages=gemma_messages,
                                temperature=0.0, max_tokens=max_out,
                            )
                            answer = completion.choices[0].message.content
                            usage = {
                                "input_tokens": completion.usage.prompt_tokens,
                                "output_tokens": completion.usage.completion_tokens,
                            }
                    elif gen_name == "qwen" and qwen_model is not None:
                        import torch
                        messages = [
                            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ]
                        qwen_device = next(qwen_model.parameters()).device
                        tokenized = qwen_tokenizer.apply_chat_template(
                            messages, add_generation_prompt=True,
                            tokenize=True, return_tensors="pt",
                        )
                        if hasattr(tokenized, "input_ids"):
                            input_ids = tokenized["input_ids"].to(qwen_device)
                        else:
                            input_ids = tokenized.to(qwen_device)
                        in_len = input_ids.shape[-1]
                        qwen_temp = getattr(args, "temperature", 0.0)
                        gen_kwargs = dict(max_new_tokens=20000)
                        if qwen_temp > 0:
                            gen_kwargs["do_sample"] = True
                            gen_kwargs["temperature"] = qwen_temp
                        else:
                            gen_kwargs["do_sample"] = False
                        with torch.no_grad():
                            out_ids = qwen_model.generate(input_ids, **gen_kwargs)
                        out_len = out_ids.shape[-1] - in_len
                        raw = qwen_tokenizer.decode(out_ids[0][in_len:], skip_special_tokens=False)
                        if "</think>" in raw:
                            answer = raw.split("</think>", 1)[1].strip()
                        else:
                            answer = raw.strip()
                        for tok in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
                            answer = answer.replace(tok, "")
                        answer = answer.strip()
                        usage = {"input_tokens": in_len, "output_tokens": out_len}
                    else:
                        log.warning("Generator %s not available in this process", gen_name)
                        answer = None
                        usage = {"note": "generator_not_available"}
                except torch.cuda.OutOfMemoryError as exc:
                    log.warning("Query %s OOM: %s — clearing cache and continuing", qid, exc)
                    torch.cuda.empty_cache()
                    answer = None
                    usage = {"error": "OOM", "detail": str(exc)}
                except Exception as exc:
                    log.error("Query %s generation failed: %s: %s", qid, type(exc).__name__, exc, exc_info=True)
                    answer = None
                    usage = {"error": f"{type(exc).__name__}: {exc}"}

                # Validate Qwen answer contains at least the Opening Statements section
                if answer and gen_name == "qwen":
                    if "opening statement" not in answer.lower():
                        log.warning("Query %s: Qwen answer has no 'Opening Statements' header — likely thinking leak, marking as failed", qid)
                        answer = None
                        usage["error"] = "thinking_leak_no_headers"

                row = {
                    "query_id": qid,
                    "query_text": query_text,
                    "condition": condition,
                    "retrieval_method": retrieval_method,
                    "generator": gen_name,
                    "gold_citations": gold_citations,
                    "retrieved_ids": [c.doc_id for c in retrieved],
                    "retrieved_chunk_details": [
                        {"doc_id": c.doc_id, "chunk_idx": c.chunk_idx, "score": round(c.score, 6)}
                        for c in retrieved
                    ],
                    "context_meta": ctx_meta,
                    "generated_answer": answer,
                    "usage": usage,
                }
                out_f.write_row(row)

        # Count and log cost
        all_rows = []
        with open(output_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    all_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        n_answered = sum(1 for r in all_rows if r.get("generated_answer"))
        log.info("%s (%s): %d/%d answered → %s", gen_name, condition, n_answered, len(all_rows), output_file)
        gen_cost.log_summary()

        if gen_cost.calls > 0:
            cost_file = out_dir / f"{gen_name}_generation_cost.json"
            cost_file.write_text(json.dumps(gen_cost.summary(), indent=2))
            append_cost_entry(
                DEFAULT_BENCHMARK_COST_CSV,
                f"final_eval_{condition}_{gen_name}_generate",
                gen_cost.cost_usd,
            )


# ===========================================================================
# prepare-eval
# ===========================================================================

def cmd_prepare_eval(args: argparse.Namespace) -> None:
    from benchmark_rag.prompts.atomic_facts import DECOMPOSE_SYSTEM_PROMPT, JUDGE_ATOMIC_SYSTEM_PROMPT

    condition = args.condition
    out_dir = _results_dir(condition)
    batch_dir = out_dir / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    gt_facts = json.loads(ATOMIC_FACTS_PATH.read_text())

    # Collect all answers across generators
    all_answers: list[dict] = []
    for gen_file in out_dir.glob("*_answers.jsonl"):
        gen_name = gen_file.stem.replace("_answers", "")
        with open(gen_file) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("generated_answer"):
                    rec["generator"] = gen_name
                    all_answers.append(rec)
    log.info("Loaded %d answers across generators", len(all_answers))

    # Decompose generated answers into atomic facts (online Flash calls)
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("ERROR: GOOGLE_API_KEY not set")
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=api_key)

    # Load existing decompositions for resume
    atomic_path = out_dir / "atomic_facts.jsonl"
    existing_decomp: dict[str, dict] = {}
    if atomic_path.exists():
        with open(atomic_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = f"{rec['query_id']}_{rec['generator']}"
                existing_decomp[key] = rec
        log.info("Loaded %d existing decompositions — resuming", len(existing_decomp))

    from tqdm import tqdm
    decomp_rows: list[dict] = list(existing_decomp.values())
    decomp_cost = CostTracker(f"decompose_{condition}")

    for rec in tqdm(all_answers, desc="Decomposing"):
        qid = rec["query_id"]
        gen_name = rec["generator"]
        key = f"{qid}_{gen_name}"
        if key in existing_decomp:
            continue

        try:
            response = _generate_with_retry(
                client, model=GENERATOR_MODEL,
                contents=f"Decompose the following legal answer into atomic facts:\n\n{rec['generated_answer']}",
                config=types.GenerateContentConfig(
                    system_instruction=DECOMPOSE_SYSTEM_PROMPT,
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )
            gen_facts = json.loads(response.text)
            um = response.usage_metadata
            decomp_cost.track(GENERATOR_MODEL, um.prompt_token_count, um.candidates_token_count)
        except Exception as e:
            log.error("Decompose failed for qid=%s gen=%s: %s: %s", qid, gen_name, type(e).__name__, e, exc_info=True)
            gen_facts = []

        gt_qid_facts = gt_facts.get(str(qid), gt_facts.get(qid, []))

        row = {
            "query_id": qid,
            "condition": condition,
            "generator": gen_name,
            "generated_facts": gen_facts,
            "ground_truth_facts": gt_qid_facts,
        }
        decomp_rows.append(row)
        existing_decomp[key] = row

        if len(decomp_rows) % 50 == 0:
            with open(atomic_path, "w") as f:
                for r in decomp_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(atomic_path, "w") as f:
        for r in decomp_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Saved %d decompositions to %s", len(decomp_rows), atomic_path)

    # Load already-judged keys to avoid re-judging
    already_judged: set[str] = set()
    # Check collected judge_results.jsonl
    judge_results_path = out_dir / "judge_results.jsonl"
    if judge_results_path.exists():
        with open(judge_results_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                k = f"judge_{_safe_key(str(rec['query_id']))}_{_safe_key(rec['generator'])}"
                already_judged.add(k)
    # Also check raw batch results not yet collected
    jobs_path = batch_dir / "jobs.json"
    if jobs_path.exists():
        prev_jobs = json.loads(jobs_path.read_text())
        for j in prev_jobs:
            raw_path = batch_dir / f"raw_results_{j['file']}"
            if raw_path.exists():
                with open(raw_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        entry = json.loads(line)
                        if entry.get("response", {}).get("candidates"):
                            already_judged.add(entry["key"])
    if already_judged:
        log.info("Found %d already-judged keys — will skip", len(already_judged))

    # Build judge batch JSONL — only unjudged decompositions
    judge_lines: list[str] = []
    skipped = 0
    for rec in decomp_rows:
        gen_facts = rec.get("generated_facts", [])
        gt_qid_facts = rec.get("ground_truth_facts", [])
        if not gen_facts or not gt_qid_facts:
            continue

        key = f"judge_{_safe_key(str(rec['query_id']))}_{_safe_key(rec['generator'])}"
        if key in already_judged:
            skipped += 1
            continue

        prompt = (
            f"generated_facts:\n{json.dumps(gen_facts, ensure_ascii=False)}\n\n"
            f"ground_truth_facts:\n{json.dumps(gt_qid_facts, ensure_ascii=False)}"
        )
        judge_lines.append(json.dumps({
            "key": key,
            "request": {
                "contents": [{"parts": [{"text": prompt}]}],
                "system_instruction": {"parts": [{"text": JUDGE_ATOMIC_SYSTEM_PROMPT}]},
                "generation_config": {
                    "temperature": 0.0,
                    "max_output_tokens": 16384,
                    "response_mime_type": "application/json",
                },
            },
        }, ensure_ascii=False))
    log.info("Judge batch: %d new requests (%d skipped — already judged)", len(judge_lines), skipped)

    judge_path = batch_dir / "judge_requests.jsonl"
    with open(judge_path, "w", encoding="utf-8") as f:
        for line in judge_lines:
            f.write(line + "\n")

    manifest = {
        "created": datetime.now().isoformat(),
        "condition": condition,
        "judge_requests": len(judge_lines),
        "judge_model": JUDGE_MODEL,
        "decompositions": len(decomp_rows),
    }
    with open(batch_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    decomp_cost.log_summary()
    if decomp_cost.calls > 0:
        cost_file = batch_dir / "decompose_cost.json"
        cost_file.write_text(json.dumps(decomp_cost.summary(), indent=2))
        append_cost_entry(
            DEFAULT_BENCHMARK_COST_CSV,
            f"final_eval_{condition}_decompose",
            decomp_cost.cost_usd,
        )

    log.info("Prepared %d judge requests → %s", len(judge_lines), judge_path)


# ===========================================================================
# submit
# ===========================================================================

def cmd_submit(args: argparse.Namespace) -> None:
    from google import genai
    from google.genai import types

    batch_dir = _results_dir(args.condition) / "batch"
    if not (batch_dir / "manifest.json").exists():
        sys.exit("Run 'prepare-eval' first.")

    client = genai.Client()

    # Load existing jobs to append (not overwrite)
    jobs_path = batch_dir / "jobs.json"
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []

    fpath = batch_dir / "judge_requests.jsonl"
    if not fpath.exists() or fpath.stat().st_size == 0:
        print("No new judge requests — all answers already judged. Nothing to submit.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    display_name = f"final_eval_{args.condition}_judge_{timestamp}"

    print(f"Uploading judge_requests.jsonl...")
    uploaded = client.files.upload(
        file=str(fpath),
        config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
    )
    print(f"  Uploaded: {uploaded.name}")

    job = client.batches.create(
        model=JUDGE_MODEL,
        src=uploaded.name,
        config={"display_name": display_name},
    )
    print(f"  Job: {job.name}")
    jobs.append({
        "name": job.name,
        "file": "judge_requests.jsonl",
        "uploaded": uploaded.name,
        "submitted": datetime.now().isoformat(),
        "display_name": display_name,
    })

    with open(jobs_path, "w") as f:
        json.dump(jobs, f, indent=2)
    print(f"Submitted. Total jobs tracked: {len(jobs)}")


# ===========================================================================
# status
# ===========================================================================

def cmd_status(args: argparse.Namespace) -> None:
    from google import genai

    batch_dir = _results_dir(args.condition) / "batch"
    jobs_path = batch_dir / "jobs.json"
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

    condition = args.condition
    out_dir = _results_dir(condition)
    batch_dir = out_dir / "batch"

    jobs_path = batch_dir / "jobs.json"
    atomic_path = out_dir / "atomic_facts.jsonl"
    if not jobs_path.exists() or not atomic_path.exists():
        sys.exit("Run 'submit' and 'prepare-eval' first.")

    jobs = json.loads(jobs_path.read_text())
    client = genai.Client()

    # Download batch results
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

        raw_path = batch_dir / f"raw_results_{j['file']}"
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

    # Load decompositions
    decomp_rows = []
    with open(atomic_path) as f:
        for line in f:
            if line.strip():
                decomp_rows.append(json.loads(line))

    # Process judge results
    judge_rows: list[dict] = []
    precision_by_gen: dict[str, list[float]] = {}
    recall_by_gen: dict[str, list[float]] = {}

    for rec in decomp_rows:
        qid = rec["query_id"]
        gen_name = rec["generator"]
        key = f"judge_{_safe_key(str(qid))}_{_safe_key(gen_name)}"
        result = all_results.get(key, {})

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

        precision_by_gen.setdefault(gen_name, []).append(precision)
        recall_by_gen.setdefault(gen_name, []).append(recall)

        judge_rows.append({
            "query_id": qid,
            "condition": condition,
            "generator": gen_name,
            "generated_facts": rec.get("generated_facts", []),
            "ground_truth_facts": rec.get("ground_truth_facts", []),
            "judge_result": parsed,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        })

    judge_path = out_dir / "judge_results.jsonl"
    with open(judge_path, "w") as f:
        for r in judge_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    summary = {"condition": condition, "generators": {}}
    for gen_name in sorted(set(r["generator"] for r in judge_rows)):
        p_list = precision_by_gen.get(gen_name, [])
        r_list = recall_by_gen.get(gen_name, [])
        avg_p = float(np.mean(p_list)) if p_list else 0.0
        avg_r = float(np.mean(r_list)) if r_list else 0.0
        avg_f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0.0
        summary["generators"][gen_name] = {
            "num_queries": len(p_list),
            "precision": round(avg_p, 4),
            "recall": round(avg_r, 4),
            "f1": round(avg_f1, 4),
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"Final Eval Results — {condition}")
    print(f"{'='*60}")
    for gen_name, scores in summary["generators"].items():
        print(f"  {gen_name:10s}  P={scores['precision']:.4f}  R={scores['recall']:.4f}  F1={scores['f1']:.4f}  (n={scores['num_queries']})")
    print(f"{'='*60}")
    print(f"Details: {judge_path}")
    print(f"Summary: {summary_path}")


# ===========================================================================
# prepare-groundedness  (needs GPU for Qwen embedder)
# ===========================================================================

def cmd_prepare_groundedness(args: argparse.Namespace) -> None:
    import faiss
    import numpy as np
    from tqdm import tqdm
    from benchmark_rag.components.decontextualizers.gemini_decontextualizer import GeminiDecontextualizer
    from benchmark_rag.components.embedders.qwen import QwenEmbedder

    condition = args.condition
    generators = [g.strip() for g in args.generators.split(",")] if args.generators else None
    out_dir = _results_dir(condition)
    batch_dir = out_dir / "batch"
    batch_dir.mkdir(parents=True, exist_ok=True)

    if not DECONTEXT_FAISS.exists() or not DECONTEXT_META.exists():
        sys.exit(f"Decontextualized corpus not found at {DECONTEXT_FAISS}. "
                 "Run the decontextualization pipeline first.")

    log.info("Loading decontextualized corpus FAISS...")
    corpus_index = faiss.read_index(str(DECONTEXT_FAISS))
    with open(DECONTEXT_META, "rb") as f:
        corpus_meta = pickle.load(f)
    log.info("Loaded %d decontextualized statements", corpus_index.ntotal)

    embedder = QwenEmbedder(model_name="Qwen/Qwen3-Embedding-8B", device="cuda:0")
    decontextualizer = GeminiDecontextualizer(model_name="gemini-2.5-flash")

    # Collect answers (filtered by --generators if specified)
    all_answers: list[dict] = []
    for gen_file in out_dir.glob("*_answers.jsonl"):
        gen_name = gen_file.stem.replace("_answers", "")
        if generators and gen_name not in generators:
            continue
        with open(gen_file) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("generated_answer"):
                    rec["generator"] = gen_name
                    all_answers.append(rec)
    log.info("Loaded %d answers across generators", len(all_answers))

    # Load existing intermediate for resume
    intermediate_path = out_dir / "groundedness_intermediate.jsonl"
    existing_keys: set = set()
    existing_rows: list[dict] = []
    if intermediate_path.exists():
        with open(intermediate_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                key = f"{rec['query_id']}_{rec['generator']}"
                existing_keys.add(key)
                existing_rows.append(rec)
        log.info("Loaded %d existing groundedness intermediates — resuming", len(existing_rows))

    groundedness_lines: list[str] = []
    intermediate_rows: list[dict] = list(existing_rows)

    for ri, row in enumerate(tqdm(all_answers, desc="Preparing groundedness")):
        qid = str(row["query_id"])
        gen_name = row["generator"]
        resume_key = f"{qid}_{gen_name}"
        if resume_key in existing_keys:
            continue

        answer = row["generated_answer"]
        statements = split_into_statements(answer)
        if not statements:
            continue

        try:
            decontext_result = decontextualizer.decontextualize(statements, answer)
            decontext_stmts = decontext_result if decontext_result else statements
        except Exception as e:
            log.error("Decontextualize failed for qid=%s gen=%s: %s", qid, gen_name, e, exc_info=True)
            decontext_stmts = statements

        import torch
        EMBED_BATCH = 8
        embeddings = []
        try:
            for i in range(0, len(decontext_stmts), EMBED_BATCH):
                batch = decontext_stmts[i:i + EMBED_BATCH]
                embeddings.extend(embedder.embed(batch))
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            log.warning("Query %s gen=%s: OOM during embedding (%d statements) — skipping",
                        qid, gen_name, len(decontext_stmts))
            torch.cuda.empty_cache()
            continue
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
            batch_key = f"ground_{_safe_key(qid)}_{_safe_key(gen_name)}_s{si}"

            groundedness_lines.append(json.dumps({
                "key": batch_key,
                "request": {
                    "contents": [{"parts": [{"text": prompt}]}],
                    "system_instruction": {"parts": [{"text": GROUNDEDNESS_JUDGE_SYSTEM}]},
                    "generation_config": {
                        "temperature": 0.0,
                        "max_output_tokens": 2048,
                    },
                },
            }, ensure_ascii=False))

            stmt_data.append({
                "original": stmt,
                "decontextualized": d_stmt,
                "batch_key": batch_key,
            })

        intermediate_rows.append({
            "query_id": qid,
            "generator": gen_name,
            "condition": condition,
            "statements": stmt_data,
        })
        existing_keys.add(resume_key)

        if len(intermediate_rows) % 50 == 0:
            with open(intermediate_path, "w") as f:
                for r in intermediate_rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(intermediate_path, "w") as f:
        for r in intermediate_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Saved %d groundedness intermediates to %s", len(intermediate_rows), intermediate_path)

    # Write batch JSONL — only contains new items from this run
    # Old items from previous runs are in separate batch submissions
    gen_suffix = "_".join(generators) if generators else "all"
    gpath = batch_dir / f"groundedness_requests_{gen_suffix}.jsonl"
    with open(gpath, "w", encoding="utf-8") as f:
        for line in groundedness_lines:
            f.write(line + "\n")

    log.info("Wrote %d groundedness judge requests to %s", len(groundedness_lines), gpath)
    decontextualizer.log_usage_summary()


# ===========================================================================
# submit-groundedness
# ===========================================================================

def cmd_submit_groundedness(args: argparse.Namespace) -> None:
    from google import genai
    from google.genai import types

    batch_dir = _results_dir(args.condition) / "batch"
    generators = [g.strip() for g in args.generators.split(",")] if args.generators else None
    gen_suffix = "_".join(generators) if generators else "all"
    fpath = batch_dir / f"groundedness_requests_{gen_suffix}.jsonl"
    if not fpath.exists() or fpath.stat().st_size == 0:
        sys.exit(f"No {fpath.name} found. Run 'prepare-groundedness --generators {gen_suffix}' first.")

    client = genai.Client()

    jobs_path = batch_dir / "groundedness_jobs.json"
    jobs = json.loads(jobs_path.read_text()) if jobs_path.exists() else []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    display_name = f"final_eval_{args.condition}_groundedness_{timestamp}"

    print(f"Uploading groundedness_requests.jsonl...")
    uploaded = client.files.upload(
        file=str(fpath),
        config=types.UploadFileConfig(display_name=display_name, mime_type="jsonl"),
    )
    print(f"  Uploaded: {uploaded.name}")

    job = client.batches.create(
        model=JUDGE_MODEL,
        src=uploaded.name,
        config={"display_name": display_name},
    )
    print(f"  Job: {job.name}")
    jobs.append({
        "name": job.name,
        "file": "groundedness_requests.jsonl",
        "uploaded": uploaded.name,
        "submitted": datetime.now().isoformat(),
        "display_name": display_name,
    })

    with open(jobs_path, "w") as f:
        json.dump(jobs, f, indent=2)
    print(f"Submitted. Total groundedness jobs tracked: {len(jobs)}")


# ===========================================================================
# collect-groundedness
# ===========================================================================

def cmd_collect_groundedness(args: argparse.Namespace) -> None:
    from google import genai
    import numpy as np

    condition = args.condition
    out_dir = _results_dir(condition)
    batch_dir = out_dir / "batch"

    jobs_path = batch_dir / "groundedness_jobs.json"
    intermediate_path = out_dir / "groundedness_intermediate.jsonl"
    if not jobs_path.exists() or not intermediate_path.exists():
        sys.exit("Run 'submit-groundedness' and 'prepare-groundedness' first.")

    jobs = json.loads(jobs_path.read_text())
    client = genai.Client()

    all_results: dict[str, dict] = {}
    for j in jobs:
        job = client.batches.get(name=j["name"])
        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        if state != "JOB_STATE_SUCCEEDED":
            print(f"WARNING: {j['display_name']} state={state}, skipping")
            continue
        if not job.dest or not job.dest.file_name:
            print(f"WARNING: {j['display_name']} no output file")
            continue

        print(f"Downloading {j['display_name']} results...")
        result_data = client.files.download(file=job.dest.file_name)
        text = result_data.decode("utf-8") if isinstance(result_data, bytes) else str(result_data)

        raw_path = batch_dir / f"raw_results_groundedness_{j['submitted'][:10]}.jsonl"
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

    intermediate = []
    with open(intermediate_path) as f:
        for line in f:
            if line.strip():
                intermediate.append(json.loads(line))

    # Process results from intermediate (has statement text)
    groundedness_rows: list[dict] = []
    processed_keys: set[str] = set()

    for entry in intermediate:
        qid = entry["query_id"]
        gen_name = entry["generator"]
        total = len(entry["statements"])
        supported = 0
        partial = 0
        not_supported = 0
        unknown = 0
        stmt_results = []

        for sd in entry["statements"]:
            key = sd["batch_key"]
            processed_keys.add(key)
            result = all_results.get(key, {})
            judge_text = result.get("text", "") or ""
            verdict = parse_verdict(judge_text) if judge_text else "MISSING"

            if verdict == "SUPPORTED":
                supported += 1
            elif verdict == "PARTIAL":
                partial += 1
            elif verdict == "NOT_SUPPORTED":
                not_supported += 1
            else:
                unknown += 1

            stmt_results.append({
                "original": sd.get("original", ""),
                "decontextualized": sd.get("decontextualized", ""),
                "verdict": verdict,
            })

        groundedness = (supported + partial) / total if total else 0.0
        groundedness_rows.append({
            "query_id": qid,
            "condition": condition,
            "generator": gen_name,
            "num_statements": total,
            "supported": supported,
            "partial": partial,
            "not_supported": not_supported,
            "unknown": unknown,
            "groundedness_score": round(groundedness, 4),
            "statements": stmt_results,
        })

    # Process remaining results not in intermediate (e.g., Gemma entries lost from intermediate)
    # Group unprocessed keys by (qid, generator)
    unprocessed: dict[tuple, list] = {}
    key_pattern = re.compile(r"^ground_(\d+)_(\w+)_s(\d+)$")
    for key, result in all_results.items():
        if key in processed_keys:
            continue
        m = key_pattern.match(key)
        if not m:
            continue
        qid_str, gen_name, stmt_idx = m.group(1), m.group(2), int(m.group(3))
        unprocessed.setdefault((qid_str, gen_name), []).append((stmt_idx, key, result))

    if unprocessed:
        log.info("Found %d queries with results not in intermediate — reconstructing", len(unprocessed))

    for (qid_str, gen_name), stmts in unprocessed.items():
        stmts.sort(key=lambda x: x[0])
        total = len(stmts)
        supported = 0
        partial = 0
        not_supported = 0
        unknown = 0
        stmt_results = []

        for stmt_idx, key, result in stmts:
            judge_text = result.get("text", "") or ""
            verdict = parse_verdict(judge_text) if judge_text else "MISSING"

            if verdict == "SUPPORTED":
                supported += 1
            elif verdict == "PARTIAL":
                partial += 1
            elif verdict == "NOT_SUPPORTED":
                not_supported += 1
            else:
                unknown += 1

            stmt_results.append({
                "original": "",
                "decontextualized": "",
                "verdict": verdict,
            })

        groundedness = (supported + partial) / total if total else 0.0
        groundedness_rows.append({
            "query_id": qid_str,
            "condition": condition,
            "generator": gen_name,
            "num_statements": total,
            "supported": supported,
            "partial": partial,
            "not_supported": not_supported,
            "unknown": unknown,
            "groundedness_score": round(groundedness, 4),
            "statements": stmt_results,
        })

    gpath = out_dir / "groundedness_results.jsonl"
    with open(gpath, "w") as f:
        for r in groundedness_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    by_gen: dict[str, list[float]] = {}
    for r in groundedness_rows:
        by_gen.setdefault(r["generator"], []).append(r["groundedness_score"])

    print(f"\n{'='*60}")
    print(f"Groundedness Results — {condition}")
    print(f"{'='*60}")
    for gen_name in sorted(by_gen):
        scores = by_gen[gen_name]
        avg = float(np.mean(scores))
        print(f"  {gen_name:10s}  groundedness={avg:.4f}  (n={len(scores)})")
    print(f"{'='*60}")
    print(f"Details: {gpath}")


# ===========================================================================
# main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="Final paper evaluation: answer generation + atomic-fact judging + groundedness")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generate", help="Generate answers")
    p_gen.add_argument("--condition", required=True, choices=["oracle", "pipeline"])
    p_gen.add_argument("--generators", required=True, help="Comma-separated: gemini,qwen,gemma")
    p_gen.add_argument("--quantize", action="store_true", help="Load Qwen in 8-bit (for OOM retry phase)")
    p_gen.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for local models (default: 0.0 = greedy)")

    p_prep = sub.add_parser("prepare-eval", help="Decompose answers + build atomic fact judge JSONL")
    p_prep.add_argument("--condition", required=True, choices=["oracle", "pipeline"])

    p_sub = sub.add_parser("submit", help="Submit atomic fact batch jobs")
    p_sub.add_argument("--condition", required=True, choices=["oracle", "pipeline"])

    p_stat = sub.add_parser("status", help="Check batch status")
    p_stat.add_argument("--condition", required=True, choices=["oracle", "pipeline"])

    p_col = sub.add_parser("collect", help="Collect atomic fact results + compute scores")
    p_col.add_argument("--condition", required=True, choices=["oracle", "pipeline"])

    p_pg = sub.add_parser("prepare-groundedness", help="Decontextualize + embed + build groundedness judge JSONL (needs GPU)")
    p_pg.add_argument("--condition", required=True, choices=["oracle", "pipeline"])
    p_pg.add_argument("--generators", default=None, help="Comma-separated generators to process (default: all available)")

    p_sg = sub.add_parser("submit-groundedness", help="Submit groundedness batch jobs")
    p_sg.add_argument("--condition", required=True, choices=["oracle", "pipeline"])
    p_sg.add_argument("--generators", default=None, help="Comma-separated generators (must match prepare-groundedness)")

    p_cg = sub.add_parser("collect-groundedness", help="Collect groundedness results")
    p_cg.add_argument("--condition", required=True, choices=["oracle", "pipeline"])

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    cmds = {
        "generate": cmd_generate,
        "prepare-eval": cmd_prepare_eval,
        "submit": cmd_submit,
        "status": cmd_status,
        "collect": cmd_collect,
        "prepare-groundedness": cmd_prepare_groundedness,
        "submit-groundedness": cmd_submit_groundedness,
        "collect-groundedness": cmd_collect_groundedness,
    }
    cmds[args.command](args)


if __name__ == "__main__":
    main()
