"""
Pre-compute atomic facts for all ground-truth answers in the test dataset.

Usage
-----
    python scripts/decompose_atomic_facts.py
    python scripts/decompose_atomic_facts.py --queries data/test_dataset/queries.json \
        --output data/test_dataset/ground_truth_atomic_facts.json

Reads each query's ``user_answer`` from queries.json, sends it to Gemini Flash
for decomposition into atomic facts, and saves the result as a JSON mapping
``{query_id: [fact, ...]}``.

This only needs to run once — the output is reused by ``run_answer_eval.py``.
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

from benchmark_rag.cost_logging import (
    DEFAULT_BENCHMARK_COST_CSV,
    append_cost_entry,
)
from benchmark_rag.evaluation.metrics import EXCLUDED_QUERY_IDS, is_query_usable
from benchmark_rag.prompts.atomic_facts import DECOMPOSE_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

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
            log.warning("Gemini %d — retrying in %ds (attempt %d)...", code, delay, attempt)
            time.sleep(delay)


def decompose_answer(client, answer: str, model: str = "gemini-2.5-flash", cost: dict | None = None) -> list[str]:
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
    if cost is not None:
        usage = response.usage_metadata
        cost["calls"] += 1
        cost["input_tokens"] += usage.prompt_token_count
        cost["output_tokens"] += usage.candidates_token_count
        cost["cost_usd"] += _estimate_cost(model, usage.prompt_token_count, usage.candidates_token_count)
    return json.loads(response.text)


def main():
    parser = argparse.ArgumentParser(description="Decompose ground-truth answers into atomic facts.")
    parser.add_argument(
        "--queries",
        default="data/test_dataset/queries.json",
        help="Path to queries.json",
    )
    parser.add_argument(
        "--output",
        default="data/test_dataset/ground_truth_atomic_facts.json",
        help="Output path for atomic facts JSON",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        help="Gemini model for decomposition",
    )
    args = parser.parse_args()

    import os
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        sys.exit("ERROR: GOOGLE_API_KEY or GEMINI_API_KEY not set.")
    client = genai.Client(api_key=api_key)

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    log.info("Loaded %d queries from %s", len(queries), args.queries)

    output_path = Path(args.output)
    existing: dict[str, list[str]] = {}
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        log.info("Loaded %d existing decompositions — resuming", len(existing))

    results = dict(existing)
    failed: list[dict] = []
    cost: dict = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    for q in tqdm(queries, desc="Decomposing"):
        qid = str(q["query_id"])
        if not is_query_usable(q):
            continue
        if qid in results:
            continue

        answer = q.get("user_answer", "")

        try:
            facts = decompose_answer(client, answer, model=args.model, cost=cost)
            results[qid] = facts
        except Exception as exc:
            log.error("Query %s failed: %s — skipping", qid, exc)
            failed.append({"query_id": qid, "error": str(exc)})
            continue

        if len(results) % 50 == 0:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    log.info("Saved %d decompositions to %s", len(results), output_path)

    if failed:
        log.warning("%d queries failed decomposition", len(failed))
        fail_path = output_path.parent / "decompose_failures.json"
        fail_path.write_text(json.dumps(failed, indent=2))

    log.info(
        "Cost: calls=%d | input_tokens=%d | output_tokens=%d | cost_usd=$%.6f",
        cost["calls"], cost["input_tokens"], cost["output_tokens"], cost["cost_usd"],
    )
    if cost["calls"] > 0:
        append_cost_entry(DEFAULT_BENCHMARK_COST_CSV, "decompose_atomic_facts", cost["cost_usd"])


if __name__ == "__main__":
    main()
