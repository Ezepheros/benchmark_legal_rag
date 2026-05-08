"""
Generate answers for the multi-generator experiment using pre-computed retrieval.

Loads shared_retrieval.jsonl (from multi_gen_retrieve.py) and generates answers
with the specified generator. Each generator runs independently so they can
be parallelized across separate jobs.

Output: runs/multi_generator_50/results/{generator}_answers.jsonl

Usage:
    python scripts/multi_gen_generate.py --generator gemini
    python scripts/multi_gen_generate.py --generator qwen
    python scripts/multi_gen_generate.py --generator gemma

Requires: GPU (for qwen/gemma), GOOGLE_API_KEY (for gemini)
Requires: runs/multi_generator_50/results/shared_retrieval.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
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

MAX_INPUT_TOKENS = 125_000
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


def _count_tokens(text: str, tokenizer=None) -> int:
    if tokenizer is not None:
        return len(tokenizer.encode(text, add_special_tokens=False))
    return len(text) // CHARS_PER_TOKEN


def build_context(chunk_details: list[dict], gt_doc_texts: dict[str, str],
                   token_budget: int, tokenizer=None) -> tuple[str, dict]:
    """Build context from retrieved chunks + full/truncated GT documents."""
    doc_ids = list(dict.fromkeys(c["doc_id"] for c in chunk_details))
    full_docs = {did: gt_doc_texts[did] for did in doc_ids if did in gt_doc_texts}
    total_doc_tokens = sum(_count_tokens(t, tokenizer) for t in full_docs.values())

    if total_doc_tokens <= token_budget:
        parts = [f"=== Document: {did} ===\n{full_docs[did]}" for did in doc_ids if did in full_docs]
        return "\n\n".join(parts), {
            "context_mode": "full_documents",
            "num_docs": len(full_docs),
            "total_doc_tokens": total_doc_tokens,
        }

    chunk_tokens = sum(_count_tokens(c["text"], tokenizer) for c in chunk_details)
    remaining = token_budget - chunk_tokens

    if remaining <= 0:
        parts = [f"[{i}] ({c['doc_id']})\n{c['text']}" for i, c in enumerate(chunk_details, 1)]
        return "\n\n".join(parts), {"context_mode": "chunks_only", "num_chunks": len(chunk_details)}

    ratio = remaining / total_doc_tokens
    chunk_parts = [f"[{i}] ({c['doc_id']})\n{c['text']}" for i, c in enumerate(chunk_details, 1)]
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


def _build_prompt(query_text: str, context: str) -> str:
    prefix = _INSTRUCTION_PREFIX.format(query_text=query_text)
    suffix = _INSTRUCTION_SUFFIX.format(query_text=query_text)
    return f"{prefix}\n\n{context}\n\n{suffix}"


def generate_gemini(prompt: str, system_prompt: str, client) -> tuple[str | None, dict]:
    from google.genai import types
    from benchmark_rag.components.generators.gemini import _generate_with_retry

    response = _generate_with_retry(
        client, model="gemini-2.5-flash", contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0, max_output_tokens=16384,
        ),
    )
    usage = response.usage_metadata
    return response.text, {
        "input_tokens": usage.prompt_token_count or 0,
        "output_tokens": usage.candidates_token_count or 0,
    }


QWEN_MAX_NEW_TOKENS = 10000


def generate_qwen(prompt: str, system_prompt: str, model, tokenizer) -> tuple[str | None, dict]:
    """Minimal Qwen generation: tokenize, generate, decode, strip thinking."""
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = input_ids.shape[-1]
    if input_len > MAX_INPUT_TOKENS:
        return None, {"input_tokens": int(input_len), "skipped": "exceeds_max_input_tokens"}

    log.info(f"Qwen generating: input_tokens={input_len}")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=QWEN_MAX_NEW_TOKENS,
            do_sample=False,
        )

    output_len = output_ids.shape[-1] - input_len
    raw = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=False)

    if "</think>" in raw:
        answer = raw.split("</think>", 1)[1].strip()
    else:
        answer = raw.strip()

    for tok in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
        answer = answer.replace(tok, "")
    answer = answer.strip()

    log.info(f"Qwen done: output_tokens={output_len}, thinking={'</think>' in raw}, answer_len={len(answer)}")

    return answer, {
        "input_tokens": int(input_len),
        "output_tokens": int(output_len),
        "had_thinking": "</think>" in raw,
    }


def run_qwen_sanity_tests(model, tokenizer, retrieval_data: list[dict]) -> bool:
    """Run a suite of sanity tests before real generation to catch broken outputs early."""
    import torch

    test_queries = []
    for qc in retrieval_data[:2]:
        test_queries.append(qc["query_text"])

    configs = [
        {"label": "greedy, no thinking",   "do_sample": False, "temperature": None, "enable_thinking": False},
        {"label": "greedy, with thinking",  "do_sample": False, "temperature": None, "enable_thinking": True},
        {"label": "t=0.05, no thinking",    "do_sample": True,  "temperature": 0.05, "enable_thinking": False},
        {"label": "t=0.05, with thinking",  "do_sample": True,  "temperature": 0.05, "enable_thinking": True},
    ]

    log.info("=" * 60)
    log.info("QWEN SANITY TESTS (no context, short answers)")
    log.info("=" * 60)

    all_passed = True
    for cfg in configs:
        for qi, query_text in enumerate(test_queries):
            messages = [
                {"role": "user", "content": f"Answer in 2-3 sentences: {query_text}"},
            ]
            template_kwargs = dict(
                add_generation_prompt=True, tokenize=True, return_tensors="pt",
            )
            if cfg["enable_thinking"]:
                template_kwargs["enable_thinking"] = True
            else:
                template_kwargs["enable_thinking"] = False

            input_ids = tokenizer.apply_chat_template(messages, **template_kwargs).to(model.device)
            input_len = input_ids.shape[-1]

            gen_kwargs = dict(max_new_tokens=500)
            if cfg["do_sample"]:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = cfg["temperature"]
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                output_ids = model.generate(input_ids, **gen_kwargs)

            output_len = output_ids.shape[-1] - input_len
            raw = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=False)

            if "</think>" in raw:
                answer = raw.split("</think>", 1)[1].strip()
            else:
                answer = raw.strip()
            for tok in ["<|im_end|>", "<|im_start|>", "<|endoftext|>"]:
                answer = answer.replace(tok, "")
            answer = answer.strip()

            is_ascii = all(ord(c) < 128 or c in '–—''""…' for c in answer[:200]) if answer else False
            is_garbage = not is_ascii or len(answer) < 10

            status = "PASS" if not is_garbage else "FAIL"
            if is_garbage:
                all_passed = False

            log.info(f"[{status}] {cfg['label']} | query {qi} | "
                     f"out_tok={output_len} | answer_len={len(answer)}")
            log.info(f"  raw: {raw!r}")
            log.info(f"  answer: {answer!r}")

    log.info("=" * 60)
    if all_passed:
        log.info("ALL SANITY TESTS PASSED")
    else:
        log.warning("SOME SANITY TESTS FAILED — output may be garbage")
    log.info("=" * 60)
    return all_passed


def generate_local(prompt: str, system_prompt: str, generator) -> tuple[str | None, dict]:
    """Call a local model (gemma) via the generator wrapper."""
    import torch

    generator._load()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    chat_kwargs = dict(add_generation_prompt=True, tokenize=True, return_tensors="pt")
    tokenized = generator._tokenizer.apply_chat_template(messages, **chat_kwargs)
    if hasattr(tokenized, "input_ids"):
        inputs = {k: v.to(generator._model.device) for k, v in tokenized.items()}
        input_ids = inputs["input_ids"]
    else:
        input_ids = tokenized.to(generator._model.device)
        inputs = None

    input_len = input_ids.shape[-1]
    if input_len > MAX_INPUT_TOKENS:
        return None, {"input_tokens": int(input_len), "skipped": "exceeds_max_input_tokens"}

    gen_kwargs = dict(max_new_tokens=generator.max_new_tokens)
    if generator.temperature == 0.0:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = generator.temperature

    with torch.no_grad():
        if inputs is not None:
            output_ids = generator._model.generate(**inputs, **gen_kwargs)
        else:
            output_ids = generator._model.generate(input_ids, **gen_kwargs)

    output_len = output_ids.shape[-1] - input_len
    generator._track_and_log(input_len, output_len)

    answer = generator._tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    return answer, {
        "input_tokens": int(input_len),
        "output_tokens": int(output_len),
    }


def _log_config(gen_name, retrieval_data, generator_obj=None):
    import torch, transformers
    log.info("=" * 60)
    log.info("CONFIGURATION")
    log.info(f"  generator:            {gen_name}")
    log.info(f"  num_queries:          {len(retrieval_data)}")
    log.info(f"  MAX_INPUT_TOKENS:     {MAX_INPUT_TOKENS}")
    log.info(f"  CONTEXT_TOKEN_BUDGET: {CONTEXT_TOKEN_BUDGET}")
    log.info(f"  torch:                {torch.__version__}")
    log.info(f"  transformers:         {transformers.__version__}")
    log.info(f"  CUDA available:       {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"  CUDA version:         {torch.version.cuda}")
        log.info(f"  GPU count:            {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_mem / 1e9
            log.info(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.1f} GB)")
    log.info(f"  SLURM_JOB_ID:         {os.environ.get('SLURM_JOB_ID', 'not set')}")
    log.info(f"  SLURM_NODELIST:       {os.environ.get('SLURM_NODELIST', 'not set')}")
    log.info(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    if generator_obj:
        log.info(f"  model_name:           {generator_obj.model_name}")
        log.info(f"  device:               {generator_obj.device}")
        log.info(f"  temperature:          {generator_obj.temperature}")
        log.info(f"  max_new_tokens:       {generator_obj.max_new_tokens}")
        log.info(f"  torch_dtype:          {generator_obj.torch_dtype}")
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate answers with one generator.")
    parser.add_argument("--generator", required=True, choices=["gemini", "qwen", "gemma"],
                        help="Which generator to run")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    if not RETRIEVAL_PATH.exists():
        sys.exit(f"Run multi_gen_retrieve.py first. Missing: {RETRIEVAL_PATH}")

    with open(RETRIEVAL_PATH) as f:
        retrieval_data = [json.loads(line) for line in f if line.strip()]
    log.info(f"Loaded {len(retrieval_data)} queries from {RETRIEVAL_PATH}")

    gen_name = args.generator
    from benchmark_rag.prompts.answer_generator import ANSWER_SYSTEM_PROMPT

    # Initialize generator / client
    generator_obj = None
    client = None
    qwen_model = None
    qwen_tokenizer = None

    if gen_name == "gemini":
        if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
            sys.exit("ERROR: set GOOGLE_API_KEY or GEMINI_API_KEY")
        from google import genai
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        client = genai.Client(api_key=api_key)
    elif gen_name == "qwen":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = "Qwen/Qwen3.5-9B"
        log.info(f"Loading {model_name}...")
        qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
        qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        log.info(f"Model loaded. device_map: {qwen_model.hf_device_map}")
    elif gen_name == "gemma":
        from benchmark_rag.components.generators.gemma import GemmaGenerator
        generator_obj = GemmaGenerator(
            model_name="google/gemma-4-E4B-it", device="auto",
            temperature=0.0, max_new_tokens=8000,
        )

    _log_config(gen_name, retrieval_data, generator_obj)

    if gen_name == "qwen":
        run_qwen_sanity_tests(qwen_model, qwen_tokenizer, retrieval_data)

    suffix = "_v2" if gen_name == "qwen" else ""
    output_file = RESULTS_DIR / f"{gen_name}{suffix}_answers.jsonl"

    # Load existing results for checkpointing
    completed_ids: set = set()
    existing_rows: list[dict] = []
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
            log.info(f"Resuming: {len(completed_ids)} queries already completed")

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
                row = {
                    "query_id": qid, "query_text": query_text,
                    "gold_citations": qc["gold_citations"],
                    "retrieved_chunk_details": chunk_details,
                    "retrieved_documents": qc.get("retrieved_documents", []),
                    "context_meta": {},
                    "generator": gen_name, "answer": None, "usage": {},
                    "note": "no chunks retrieved",
                }
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
                continue

            tok = qwen_tokenizer if qwen_tokenizer is not None else None
            context, ctx_meta = build_context(chunk_details, gt_doc_texts, CONTEXT_TOKEN_BUDGET, tokenizer=tok)
            prompt = _build_prompt(query_text, context)

            prompt_tokens = _count_tokens(prompt, tok)
            if prompt_tokens > MAX_INPUT_TOKENS:
                log.warning(f"Query {qid}: prompt too long ({prompt_tokens} tokens), skipping")
                row = {
                    "query_id": qid, "query_text": query_text,
                    "gold_citations": qc["gold_citations"],
                    "retrieved_chunk_details": chunk_details,
                    "retrieved_documents": qc.get("retrieved_documents", []),
                    "context_meta": ctx_meta,
                    "generator": gen_name, "answer": None,
                    "usage": {"input_tokens": prompt_tokens},
                    "note": f"skipped: prompt exceeds {MAX_INPUT_TOKENS} token limit",
                }
                out_f.write(json.dumps(row) + "\n")
                out_f.flush()
                continue

            try:
                if gen_name == "gemini":
                    answer, usage = generate_gemini(prompt, ANSWER_SYSTEM_PROMPT, client)
                elif gen_name == "qwen":
                    answer, usage = generate_qwen(prompt, ANSWER_SYSTEM_PROMPT, qwen_model, qwen_tokenizer)
                else:
                    answer, usage = generate_local(prompt, ANSWER_SYSTEM_PROMPT, generator_obj)

                if usage.get("skipped"):
                    log.warning(f"Query {qid}: {usage['skipped']} "
                                f"({usage['input_tokens']} tokens), skipping")
                    row = {
                        "query_id": qid, "query_text": query_text,
                        "gold_citations": qc["gold_citations"],
                        "retrieved_chunk_details": chunk_details,
                        "retrieved_documents": qc.get("retrieved_documents", []),
                        "context_meta": ctx_meta,
                        "generator": gen_name, "answer": None, "usage": usage,
                        "note": f"skipped: {usage['skipped']}",
                    }
                    out_f.write(json.dumps(row) + "\n")
                    out_f.flush()
                    continue
            except Exception as e:
                log.exception(f"Failed for query {qid} ({gen_name}): {type(e).__name__}: {e}")
                answer = None
                usage = {"error": f"{type(e).__name__}: {e}"}

            row = {
                "query_id": qid, "query_text": query_text,
                "gold_citations": qc["gold_citations"],
                "retrieved_chunk_details": chunk_details,
                "retrieved_documents": qc.get("retrieved_documents", []),
                "context_meta": ctx_meta,
                "generator": gen_name, "answer": answer, "usage": usage,
            }
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()

    with open(output_file) as f:
        all_rows = [json.loads(l) for l in f if l.strip()]
    n_answered = sum(1 for r in all_rows if r.get("answer"))
    n_skipped = sum(1 for r in all_rows if r.get("note"))
    log.info(f"{gen_name}: {n_answered}/{len(all_rows)} answers, {n_skipped} skipped → {output_file}")


if __name__ == "__main__":
    main()
