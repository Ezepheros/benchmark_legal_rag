#!/usr/bin/env python
"""
Retry failed batch decontextualization requests and merge all results.

Subcommands:
    prepare  — Scan failures from the first batch run, create retry JSONL
               (JSON parse failures + timeouts, with higher max_output_tokens).
    submit   — Upload retry JSONL and create batch job.
    status   — Check retry batch job status.
    merge    — Combine original + retry raw results into final dataset.
               Uses option-3 logic: length mismatches are accepted as-is
               (no 1:1 pairing required — just a flat list per document).

Usage (from project root, with .venv activated):

    python -m benchmark_rag.components.decontextualizers.retry_and_merge prepare
    python -m benchmark_rag.components.decontextualizers.retry_and_merge submit
    python -m benchmark_rag.components.decontextualizers.retry_and_merge status
    python -m benchmark_rag.components.decontextualizers.retry_and_merge merge
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import nltk
import pandas as pd
from dotenv import load_dotenv

from benchmark_rag.prompts.decontextualizer import (
    DECONTEXTUALIZE_INSTRUCTION,
    DECONTEXTUALIZE_REMINDER,
    DECONTEXTUALIZE_SYSTEM_PROMPT,
)

log = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
BATCH_DIR = SCRIPT_DIR / "batch_output"
RETRY_DIR = BATCH_DIR / "retry"

MODEL = "gemini-2.5-flash"
RETRY_MAX_OUTPUT_TOKENS = 65_536  # doubled from original 32k
TEMPERATURE = 0.05

FLASH_INPUT_PRICE = 0.30 / 1_000_000
FLASH_OUTPUT_PRICE = 0.30 / 1_000_000
EST_OUTPUT_TOKENS_PER_REQUEST = 5_000


# ---------------------------------------------------------------------------
# Helpers (shared with batch_decontextualize.py)
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    return tokenizer.tokenize(text)


def _group_into_snippets(sentences: list[str], target_size: int) -> list[list[str]]:
    snippets: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for sent in sentences:
        added = len(sent) + (1 if current else 0)
        if current_len + added > target_size and current:
            snippets.append(list(current))
            current = [sent]
            current_len = len(sent)
        else:
            current.append(sent)
            current_len += added
    if current:
        snippets.append(list(current))
    return snippets


def _try_parse_json_array(text: str) -> list[str] | None:
    """Parse model output as a JSON array of strings.

    Handles: markdown fences, arrays of dicts with a "statement" or
    "revised_statement" key, and truncated JSON (salvages complete entries).
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    # Try direct parse first
    parsed = _try_json_loads(cleaned)

    # If truncated, try closing the array
    if parsed is None and cleaned.startswith("[") and not cleaned.endswith("]"):
        # Find last complete entry (last complete quoted string or dict)
        last_comma = cleaned.rfind(",")
        if last_comma > 0:
            salvaged = cleaned[:last_comma] + "\n]"
            parsed = _try_json_loads(salvaged)
            if parsed is not None:
                log.info("Salvaged truncated JSON: kept %d entries", len(parsed))

    if parsed is None:
        return None

    if not isinstance(parsed, list):
        return None

    # Already a string array — done
    if all(isinstance(s, str) for s in parsed):
        return parsed

    # Array of dicts — extract "statement" or "revised_statement" values
    if all(isinstance(s, dict) for s in parsed):
        extracted = []
        for d in parsed:
            val = d.get("statement") or d.get("revised_statement") or d.get("text")
            if isinstance(val, str):
                extracted.append(val)
        if extracted:
            log.info("Extracted %d statements from dict-array response", len(extracted))
            return extracted

    return None


def _try_json_loads(text: str):
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# prepare — extract failed keys from original raw results, build retry JSONL
# ---------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> None:
    raw_path = BATCH_DIR / "raw_results_batch_requests_000.jsonl"
    manifest_path = BATCH_DIR / "manifest.json"
    if not raw_path.exists() or not manifest_path.exists():
        print("Missing raw_results or manifest.json in batch_output/. Run the first batch first.")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())

    # Identify which keys failed due to JSON parse or timeout
    # by scanning the raw results directly
    retry_keys: set[str] = set()
    timeout_keys: set[str] = set()
    parse_fail_keys: set[str] = set()

    print("Scanning raw results for failures...")
    with open(raw_path) as f:
        for line in f:
            entry = json.loads(line)
            key = entry["key"]

            # Timeout: no "response" key, has "error" with code 4
            if "response" not in entry:
                timeout_keys.add(key)
                retry_keys.add(key)
                continue

            # Check if response parses as valid JSON array
            resp = entry["response"]
            try:
                text = resp["candidates"][0]["content"]["parts"][0]["text"]
                parsed = _try_parse_json_array(text)
                if parsed is None:
                    parse_fail_keys.add(key)
                    retry_keys.add(key)
            except (KeyError, IndexError, TypeError):
                parse_fail_keys.add(key)
                retry_keys.add(key)

    print(f"  Timeouts:           {len(timeout_keys)}")
    print(f"  JSON parse failures: {len(parse_fail_keys)}")
    print(f"  Total to retry:     {len(retry_keys)}")

    if not retry_keys:
        print("No failures to retry.")
        return

    # Extract matching requests from original JSONL, bump max_output_tokens
    RETRY_DIR.mkdir(parents=True, exist_ok=True)
    retry_lines: list[str] = []
    total_prompt_chars = 0

    print("Extracting requests from original JSONL...")
    original_jsonl = BATCH_DIR / "batch_requests_000.jsonl"
    with open(original_jsonl) as f:
        for line in f:
            entry = json.loads(line)
            if entry["key"] in retry_keys:
                entry["request"]["generation_config"]["max_output_tokens"] = RETRY_MAX_OUTPUT_TOKENS
                retry_lines.append(json.dumps(entry, ensure_ascii=False))
                prompt_text = entry["request"]["contents"][0]["parts"][0]["text"]
                total_prompt_chars += len(prompt_text)

    retry_jsonl_path = RETRY_DIR / "retry_requests.jsonl"
    with open(retry_jsonl_path, "w", encoding="utf-8") as f:
        for line in retry_lines:
            f.write(line + "\n")

    # Cost estimate
    est_input_tokens = total_prompt_chars // 4
    est_output_tokens = len(retry_lines) * EST_OUTPUT_TOKENS_PER_REQUEST
    in_cost = est_input_tokens * FLASH_INPUT_PRICE
    out_cost = est_output_tokens * FLASH_OUTPUT_PRICE
    total_cost = in_cost + out_cost

    retry_meta = {
        "created": datetime.now().isoformat(),
        "model": MODEL,
        "max_output_tokens": RETRY_MAX_OUTPUT_TOKENS,
        "total_requests": len(retry_lines),
        "timeout_keys": sorted(timeout_keys),
        "parse_fail_keys": sorted(parse_fail_keys),
        "est_input_tokens": est_input_tokens,
        "est_output_tokens": est_output_tokens,
        "est_cost_usd": round(total_cost, 2),
    }
    with open(RETRY_DIR / "retry_meta.json", "w") as f:
        json.dump(retry_meta, f, indent=2)

    size_mb = retry_jsonl_path.stat().st_size / 1e6
    print(f"\n=== Retry Preparation Complete ===")
    print(f"Retry JSONL: {retry_jsonl_path.name} ({size_mb:.1f} MB)")
    print(f"Requests:    {len(retry_lines)}")
    print(f"max_output_tokens bumped to {RETRY_MAX_OUTPUT_TOKENS}")
    print(f"\n--- Cost Estimate ---")
    print(f"Est. input tokens:  {est_input_tokens:>12,}  (${in_cost:.2f})")
    print(f"Est. output tokens: {est_output_tokens:>12,}  (${out_cost:.2f})")
    print(f"Est. total cost:                 ${total_cost:.2f}")


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def cmd_submit(args: argparse.Namespace) -> None:
    from google import genai
    from google.genai import types

    retry_jsonl = RETRY_DIR / "retry_requests.jsonl"
    if not retry_jsonl.exists():
        print("No retry_requests.jsonl found. Run 'prepare' first.")
        sys.exit(1)

    client = genai.Client()

    print(f"Uploading {retry_jsonl.name}...")
    uploaded = client.files.upload(
        file=str(retry_jsonl),
        config=types.UploadFileConfig(display_name="retry_requests", mime_type="jsonl"),
    )
    print(f"  Uploaded as: {uploaded.name}")

    print("Creating batch job...")
    job = client.batches.create(
        model=MODEL,
        src=uploaded.name,
        config={"display_name": "decontext_retry"},
    )
    print(f"  Job created: {job.name}")

    jobs_info = {
        "name": job.name,
        "uploaded_file": uploaded.name,
        "submitted": datetime.now().isoformat(),
    }
    jobs_path = RETRY_DIR / "retry_job.json"
    with open(jobs_path, "w") as f:
        json.dump(jobs_info, f, indent=2)

    print(f"\nRetry job submitted. Saved to {jobs_path}")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    from google import genai

    jobs_path = RETRY_DIR / "retry_job.json"
    if not jobs_path.exists():
        print("No retry_job.json found. Run 'submit' first.")
        sys.exit(1)

    job_info = json.loads(jobs_path.read_text())
    client = genai.Client()

    job = client.batches.get(name=job_info["name"])
    state = job.state.name if hasattr(job.state, "name") else str(job.state)
    print(f"  {job_info['name']}: {state}")
    if state == "JOB_STATE_FAILED" and hasattr(job, "error") and job.error:
        print(f"    Error: {job.error}")
    elif state == "JOB_STATE_EXPIRED":
        print(f"    Expired (>48h). Resubmit.")


# ---------------------------------------------------------------------------
# merge — combine original + retry results into final dataset (option 3)
# ---------------------------------------------------------------------------

def cmd_merge(args: argparse.Namespace) -> None:
    from google import genai

    manifest_path = BATCH_DIR / "manifest.json"
    original_raw = BATCH_DIR / "raw_results_batch_requests_000.jsonl"
    retry_job_path = RETRY_DIR / "retry_job.json"

    if not manifest_path.exists() or not original_raw.exists():
        print("Missing manifest.json or raw_results in batch_output/.")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())

    # --- Load original raw results ---
    print("Loading original raw results...")
    all_results: dict[str, dict] = {}
    with open(original_raw) as f:
        for line in f:
            entry = json.loads(line)
            key = entry["key"]
            all_results[key] = _parse_batch_entry(entry)

    # --- Load retry raw results (override originals) ---
    retry_raw_path = RETRY_DIR / "raw_results_retry.jsonl"
    if retry_raw_path.exists():
        print("Loading retry raw results (already downloaded)...")
        _load_raw_into(retry_raw_path, all_results)
    elif retry_job_path.exists():
        print("Downloading retry results from API...")
        client = genai.Client()
        job_info = json.loads(retry_job_path.read_text())
        job = client.batches.get(name=job_info["name"])
        state = job.state.name if hasattr(job.state, "name") else str(job.state)

        if state != "JOB_STATE_SUCCEEDED":
            print(f"Retry job state={state}. Cannot merge yet.")
            if state == "JOB_STATE_FAILED" and hasattr(job, "error"):
                print(f"  Error: {job.error}")
            sys.exit(1)

        if not job.dest or not job.dest.file_name:
            print("Retry job has no output file.")
            sys.exit(1)

        result_data = client.files.download(file=job.dest.file_name)
        result_text = result_data.decode("utf-8") if isinstance(result_data, bytes) else str(result_data)
        retry_raw_path.write_text(result_text, encoding="utf-8")
        print(f"  Saved to {retry_raw_path.name}")
        _load_raw_into(retry_raw_path, all_results)
    else:
        print("No retry results found. Merging with original results only.")

    # --- Re-parse documents from parquet ---
    print("Re-parsing documents from parquet...")
    df = pd.read_parquet(PROJECT_ROOT / manifest["metadata"]["dataset"])
    gt = df[df["is_ground_truth"]]
    snippet_size = manifest["metadata"]["snippet_size"]

    doc_snippets: dict[str, list[list[str]]] = {}
    for _, row in gt.iterrows():
        citation = row["citation"]
        if citation not in manifest["documents"]:
            continue
        sentences = _split_sentences(row["text"])
        doc_snippets[citation] = _group_into_snippets(sentences, snippet_size)

    # --- Build final dataset (option 3: accept length mismatches) ---
    print("Building dataset...")
    dataset: dict = {
        "metadata": {
            **manifest["metadata"],
            "merged": datetime.now().isoformat(),
        },
        "documents": [],
    }

    total_success = 0
    total_failed = 0
    total_decon_statements = 0
    total_original_statements = 0

    for citation, doc_info in manifest["documents"].items():
        snippet_groups = doc_snippets.get(citation, [])

        doc_out: dict = {
            "citation": doc_info["citation"],
            "name": doc_info["name"],
            "court": doc_info["court"],
            "url": doc_info["url"],
            "char_count": doc_info["char_count"],
            "is_statute": doc_info["is_statute"],
            "original_statements": [],
            "decontextualized_statements": [],
            "failed_snippets": [],
        }

        for snip_meta, snip_sents in zip(doc_info["snippets"], snippet_groups):
            key = snip_meta["key"]
            doc_out["original_statements"].extend(snip_sents)

            if doc_info["is_statute"]:
                doc_out["decontextualized_statements"].extend(snip_sents)
                total_success += 1
                continue

            result = all_results.get(key)
            if result and result["success"]:
                doc_out["decontextualized_statements"].extend(result["decontextualized"])
                total_success += 1
            else:
                doc_out["decontextualized_statements"].extend(snip_sents)
                reason = "missing from results"
                if result:
                    reason = result.get("error", "JSON parse failure")
                doc_out["failed_snippets"].append({"key": key, "reason": reason})
                total_failed += 1

        doc_out["num_original_statements"] = len(doc_out["original_statements"])
        doc_out["num_decontextualized_statements"] = len(doc_out["decontextualized_statements"])
        total_original_statements += doc_out["num_original_statements"]
        total_decon_statements += doc_out["num_decontextualized_statements"]
        dataset["documents"].append(doc_out)

    remaining_failures = sum(len(d["failed_snippets"]) for d in dataset["documents"])
    dataset["metadata"]["total_successful_snippets"] = total_success
    dataset["metadata"]["total_failed_snippets"] = remaining_failures
    dataset["metadata"]["total_original_statements"] = total_original_statements
    dataset["metadata"]["total_decontextualized_statements"] = total_decon_statements

    output_path = BATCH_DIR / "final_decontextualized_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n=== Merge Complete ===")
    print(f"Output: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Documents:              {len(dataset['documents'])}")
    print(f"Successful snippets:    {total_success:,}")
    print(f"Still-failed snippets:  {remaining_failures}")
    print(f"Original statements:    {total_original_statements:,}")
    print(f"Decontextualized stmts: {total_decon_statements:,}")


def _parse_batch_entry(entry: dict) -> dict:
    """Parse a single batch result entry into a success/failure dict."""
    if "response" not in entry:
        msg = entry.get("error", {}).get("message", "unknown error")
        return {"success": False, "error": f"timeout: {msg}"}

    resp = entry["response"]
    try:
        text = resp["candidates"][0]["content"]["parts"][0]["text"]
        parsed = _try_parse_json_array(text)
        if parsed is not None:
            return {"decontextualized": parsed, "success": True}
        return {"success": False, "error": "JSON parse failure", "raw": text[:500]}
    except (KeyError, IndexError, TypeError) as e:
        return {"success": False, "error": str(e)}


def _load_raw_into(path: Path, results: dict[str, dict]) -> None:
    """Load a raw JSONL results file, overriding existing entries."""
    count = 0
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            key = entry["key"]
            results[key] = _parse_batch_entry(entry)
            count += 1
    print(f"  Loaded {count} retry results.")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retry failed batch requests and merge results",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare", help="Scan failures, create retry JSONL")
    sub.add_parser("submit", help="Upload retry JSONL and create batch job")
    sub.add_parser("status", help="Check retry batch job status")
    sub.add_parser("merge", help="Combine original + retry into final dataset")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    load_dotenv(PROJECT_ROOT / ".env")

    {"prepare": cmd_prepare, "submit": cmd_submit,
     "status": cmd_status, "merge": cmd_merge}[args.command](args)


if __name__ == "__main__":
    main()
