#!/usr/bin/env python
"""
Batch decontextualization of ground truth legal documents via Gemini batch API.

Subcommands:
    prepare  — Parse all GT documents, build JSONL batch request files + manifest.
    submit   — Upload JSONL files and create batch jobs.
    status   — Check batch job status.
    collect  — Download results, validate JSON, stitch into final dataset.

Usage (from project root, with .venv activated):

    # 1. Prepare request files
    python -m benchmark_rag.components.decontextualizers.batch_decontextualize prepare

    # 2. Submit to Gemini batch API
    python -m benchmark_rag.components.decontextualizers.batch_decontextualize submit

    # 3. Check status (min 24h wait)
    python -m benchmark_rag.components.decontextualizers.batch_decontextualize status

    # 4. Collect results and build dataset
    python -m benchmark_rag.components.decontextualizers.batch_decontextualize collect
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

SNIPPET_SIZE = 10_000
MODEL = "gemini-2.5-flash"
TEMPERATURE = 0.05
MAX_OUTPUT_TOKENS = 32_000
MAX_JSONL_BYTES = 2 * 1024**3  # 2 GB per file

# Gemini 2.5 Flash pricing ($/token) — update if pricing changes
FLASH_INPUT_PRICE = 0.30 / 1_000_000
FLASH_OUTPUT_PRICE = 0.30 / 1_000_000
EST_OUTPUT_TOKENS_PER_REQUEST = 5_000


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    return tokenizer.tokenize(text)


def _group_into_snippets(sentences: list[str], target_size: int) -> list[list[str]]:
    """Group sentences into snippet batches of ~target_size characters."""
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


def _build_prompt(document_text: str, statements: list[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(statements))
    return (
        f"{DECONTEXTUALIZE_INSTRUCTION}\n\n"
        f"STATEMENTS TO DECONTEXTUALIZE:\n{numbered}\n\n"
        f"===== DOCUMENT =====\n{document_text}\n===== END DOCUMENT =====\n\n"
        f"STATEMENTS TO DECONTEXTUALIZE (repeated for reference):\n{numbered}\n\n"
        f"{DECONTEXTUALIZE_REMINDER}"
    )


def _safe_key(citation: str) -> str:
    return (
        citation.replace(" ", "_").replace(",", "").replace("/", "_")
        .replace("(", "").replace(")", "").replace(".", "_")
    )


def _try_parse_json_array(text: str) -> list[str] | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
        return parsed
    return None


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> None:
    """Parse all GT documents, create JSONL request files and a manifest."""
    df = pd.read_parquet(PROJECT_ROOT / args.dataset)
    gt = df[df["is_ground_truth"]].sort_values("char_count", ascending=False).reset_index(drop=True)

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "metadata": {
            "model": MODEL,
            "snippet_size": args.snippet_size,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "created": datetime.now().isoformat(),
            "dataset": args.dataset,
        },
        "documents": {},
    }

    jsonl_lines: list[str] = []
    total_prompt_chars = 0

    print(f"Processing {len(gt)} ground-truth documents...")
    for idx, (_, row) in enumerate(gt.iterrows()):
        citation = row["citation"]
        text = row["text"]
        is_statute = row["court"] == "STATUTE"

        sentences = _split_sentences(text)
        snippet_groups = _group_into_snippets(sentences, args.snippet_size)

        safe = _safe_key(citation)
        doc_entry: dict = {
            "citation": citation,
            "name": row.get("name", ""),
            "court": row.get("court", ""),
            "url": row.get("url", ""),
            "char_count": int(row["char_count"]),
            "is_statute": is_statute,
            "snippets": [],
            "total_sentences": sum(len(sg) for sg in snippet_groups),
        }

        sentence_offset = 0
        for snip_idx, snip_sents in enumerate(snippet_groups):
            key = f"{safe}__snip{snip_idx:04d}_of{len(snippet_groups):04d}"
            doc_entry["snippets"].append({
                "key": key,
                "sentence_offset": sentence_offset,
                "num_sentences": len(snip_sents),
            })

            if not is_statute:
                prompt = _build_prompt(text, snip_sents)
                total_prompt_chars += len(prompt) + len(DECONTEXTUALIZE_SYSTEM_PROMPT)

                request_obj = {
                    "key": key,
                    "request": {
                        "contents": [{"parts": [{"text": prompt}]}],
                        "system_instruction": {"parts": [{"text": DECONTEXTUALIZE_SYSTEM_PROMPT}]},
                        "generation_config": {
                            "temperature": TEMPERATURE,
                            "max_output_tokens": MAX_OUTPUT_TOKENS,
                            "response_mime_type": "application/json",
                        },
                    },
                }
                jsonl_lines.append(json.dumps(request_obj, ensure_ascii=False))

            sentence_offset += len(snip_sents)

        manifest["documents"][citation] = doc_entry

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(gt)} documents parsed...")

    # --- Write JSONL files, splitting at 2 GB ---
    jsonl_files: list[str] = []
    current_lines: list[str] = []
    current_bytes = 0
    file_idx = 0

    for line in jsonl_lines:
        line_bytes = len(line.encode("utf-8")) + 1  # +1 for newline
        if current_bytes + line_bytes > MAX_JSONL_BYTES and current_lines:
            fname = f"batch_requests_{file_idx:03d}.jsonl"
            _write_jsonl(BATCH_DIR / fname, current_lines)
            jsonl_files.append(fname)
            file_idx += 1
            current_lines = []
            current_bytes = 0
        current_lines.append(line)
        current_bytes += line_bytes

    if current_lines:
        fname = f"batch_requests_{file_idx:03d}.jsonl"
        _write_jsonl(BATCH_DIR / fname, current_lines)
        jsonl_files.append(fname)

    # --- Cost estimation ---
    n_requests = len(jsonl_lines)
    est_input_tokens = total_prompt_chars // 4
    est_output_tokens = n_requests * EST_OUTPUT_TOKENS_PER_REQUEST
    in_cost = est_input_tokens * FLASH_INPUT_PRICE
    out_cost = est_output_tokens * FLASH_OUTPUT_PRICE
    total_cost = in_cost + out_cost

    total_jsonl_bytes = sum((BATCH_DIR / f).stat().st_size for f in jsonl_files)

    n_statutes = sum(1 for d in manifest["documents"].values() if d["is_statute"])
    n_non_statutes = len(manifest["documents"]) - n_statutes

    manifest["metadata"].update({
        "jsonl_files": jsonl_files,
        "total_requests": n_requests,
        "total_documents": len(manifest["documents"]),
        "total_statute_documents": n_statutes,
        "total_non_statute_documents": n_non_statutes,
        "est_input_tokens": est_input_tokens,
        "est_output_tokens_per_request": EST_OUTPUT_TOKENS_PER_REQUEST,
        "est_total_output_tokens": est_output_tokens,
        "est_cost_usd": round(total_cost, 2),
    })

    with open(BATCH_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n=== Batch Preparation Complete ===")
    print(f"Output directory: {BATCH_DIR}")
    print(f"Documents:    {len(manifest['documents'])} "
          f"({n_non_statutes} non-statute, {n_statutes} statute)")
    print(f"API requests: {n_requests} (non-statute snippets only)")
    print(f"JSONL files:  {len(jsonl_files)}")
    for fname in jsonl_files:
        size_mb = (BATCH_DIR / fname).stat().st_size / 1e6
        print(f"  {fname}: {size_mb:.1f} MB")
    print(f"Total JSONL:  {total_jsonl_bytes / 1e6:.1f} MB")
    print(f"Manifest:     {(BATCH_DIR / 'manifest.json').stat().st_size / 1e6:.1f} MB")
    print(f"\n--- Cost Estimate (Gemini 2.5 Flash @ ${FLASH_INPUT_PRICE*1e6:.2f}/1M in, "
          f"${FLASH_OUTPUT_PRICE*1e6:.2f}/1M out) ---")
    print(f"Est. input tokens:  {est_input_tokens:>14,}  (${in_cost:>8.2f})")
    print(f"Est. output tokens: {est_output_tokens:>14,}  (${out_cost:>8.2f})")
    print(f"Est. total cost:    {'':>14s}   ${total_cost:>8.2f}")


def _write_jsonl(path: Path, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

def cmd_submit(args: argparse.Namespace) -> None:
    """Upload JSONL files and create batch jobs."""
    from google import genai
    from google.genai import types

    manifest_path = BATCH_DIR / "manifest.json"
    if not manifest_path.exists():
        print("No manifest.json found. Run 'prepare' first.")
        sys.exit(1)

    manifest = json.loads(manifest_path.read_text())
    client = genai.Client()

    jobs: list[dict] = []
    for fname in manifest["metadata"]["jsonl_files"]:
        fpath = BATCH_DIR / fname
        print(f"Uploading {fname} ({fpath.stat().st_size / 1e6:.1f} MB)...")
        uploaded = client.files.upload(
            file=str(fpath),
            config=types.UploadFileConfig(display_name=fname, mime_type="jsonl"),
        )
        print(f"  Uploaded as: {uploaded.name}")

        print(f"Creating batch job...")
        job = client.batches.create(
            model=manifest["metadata"]["model"],
            src=uploaded.name,
            config={"display_name": f"decontext_{fname}"},
        )
        print(f"  Job created: {job.name}")
        jobs.append({
            "name": job.name,
            "jsonl_file": fname,
            "uploaded_file": uploaded.name,
            "submitted": datetime.now().isoformat(),
        })

    jobs_path = BATCH_DIR / "jobs.json"
    with open(jobs_path, "w") as f:
        json.dump(jobs, f, indent=2)

    print(f"\n{len(jobs)} batch job(s) submitted. Saved to {jobs_path}")
    print("Minimum wait: ~24 hours. Check with: ... status")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

def cmd_status(args: argparse.Namespace) -> None:
    """Check batch job status (one-shot)."""
    from google import genai

    jobs_path = BATCH_DIR / "jobs.json"
    if not jobs_path.exists():
        print("No jobs.json found. Run 'submit' first.")
        sys.exit(1)

    jobs = json.loads(jobs_path.read_text())
    client = genai.Client()

    for job_info in jobs:
        job = client.batches.get(name=job_info["name"])
        state = job.state.name if hasattr(job.state, "name") else str(job.state)
        print(f"  {job_info['name']}: {state}  ({job_info['jsonl_file']})")
        if state == "JOB_STATE_FAILED" and hasattr(job, "error") and job.error:
            print(f"    Error: {job.error}")
        elif state == "JOB_STATE_EXPIRED":
            print(f"    Expired (>48h). Resubmit or split into smaller batches.")


# ---------------------------------------------------------------------------
# collect
# ---------------------------------------------------------------------------

def cmd_collect(args: argparse.Namespace) -> None:
    """Download batch results, validate JSON, stitch into final dataset."""
    from google import genai

    jobs_path = BATCH_DIR / "jobs.json"
    manifest_path = BATCH_DIR / "manifest.json"
    if not jobs_path.exists() or not manifest_path.exists():
        print("Missing jobs.json or manifest.json.")
        sys.exit(1)

    jobs = json.loads(jobs_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    client = genai.Client()

    # --- Re-parse documents from parquet to get original sentences ---
    print("Loading dataset and re-parsing sentences...")
    df = pd.read_parquet(PROJECT_ROOT / manifest["metadata"]["dataset"])
    gt = df[df["is_ground_truth"]]
    snippet_size = manifest["metadata"]["snippet_size"]

    doc_sentences: dict[str, list[list[str]]] = {}
    for _, row in gt.iterrows():
        citation = row["citation"]
        if citation not in manifest["documents"]:
            continue
        sentences = _split_sentences(row["text"])
        snippet_groups = _group_into_snippets(sentences, snippet_size)
        doc_sentences[citation] = snippet_groups

    # --- Download batch results ---
    all_results: dict[str, dict] = {}
    for job_info in jobs:
        job = client.batches.get(name=job_info["name"])
        state = job.state.name if hasattr(job.state, "name") else str(job.state)

        if state != "JOB_STATE_SUCCEEDED":
            print(f"WARNING: Job {job_info['name']} state={state}, skipping.")
            if state == "JOB_STATE_FAILED" and hasattr(job, "error") and job.error:
                print(f"  Error: {job.error}")
            continue

        if not job.dest or not job.dest.file_name:
            print(f"WARNING: Job {job_info['name']} has no output file, skipping.")
            continue

        print(f"Downloading results for {job_info['jsonl_file']}...")
        result_data = client.files.download(file=job.dest.file_name)
        if isinstance(result_data, bytes):
            result_text = result_data.decode("utf-8")
        else:
            result_text = str(result_data)

        # Save raw results for debugging
        raw_path = BATCH_DIR / f"raw_results_{job_info['jsonl_file']}"
        raw_path.write_text(result_text, encoding="utf-8")
        print(f"  Raw results saved to {raw_path.name}")

        for line in result_text.strip().split("\n"):
            if not line.strip():
                continue
            entry = json.loads(line)
            key = entry["key"]
            try:
                resp_text = entry["response"]["candidates"][0]["content"]["parts"][0]["text"]
                parsed = _try_parse_json_array(resp_text)
                if parsed is not None:
                    all_results[key] = {"decontextualized": parsed, "success": True}
                else:
                    all_results[key] = {
                        "decontextualized": None, "success": False,
                        "raw": resp_text[:1000],
                    }
            except (KeyError, IndexError, TypeError) as e:
                all_results[key] = {
                    "decontextualized": None, "success": False,
                    "error": str(e),
                }

    # --- Stitch into final dataset ---
    print("Stitching results...")
    dataset: dict = {
        "metadata": {
            **manifest["metadata"],
            "collected": datetime.now().isoformat(),
            "total_batch_results": len(all_results),
            "successful_results": sum(1 for r in all_results.values() if r["success"]),
            "failed_results": sum(1 for r in all_results.values() if not r["success"]),
        },
        "documents": [],
    }

    total_changed = 0
    total_unchanged = 0
    total_failed_snippets = 0

    for citation, doc_info in manifest["documents"].items():
        snippet_groups = doc_sentences.get(citation, [])
        doc_out: dict = {
            "citation": doc_info["citation"],
            "name": doc_info["name"],
            "court": doc_info["court"],
            "url": doc_info["url"],
            "char_count": doc_info["char_count"],
            "is_statute": doc_info["is_statute"],
            "statements": [],
            "failed_snippets": [],
        }

        for snip_meta, snip_sents in zip(doc_info["snippets"], snippet_groups):
            key = snip_meta["key"]

            if doc_info["is_statute"]:
                for sent in snip_sents:
                    doc_out["statements"].append({
                        "original": sent,
                        "decontextualized": sent,
                        "changed": False,
                    })
                    total_unchanged += 1
                continue

            result = all_results.get(key)
            if (
                result
                and result["success"]
                and len(result["decontextualized"]) == len(snip_sents)
            ):
                for orig, decon in zip(snip_sents, result["decontextualized"]):
                    changed = orig.strip() != decon.strip()
                    doc_out["statements"].append({
                        "original": orig,
                        "decontextualized": decon,
                        "changed": changed,
                    })
                    if changed:
                        total_changed += 1
                    else:
                        total_unchanged += 1
            else:
                reason = "missing from results"
                if result and not result["success"]:
                    reason = result.get("error", "JSON parse failure")
                elif result and result["success"]:
                    reason = (
                        f"length mismatch: expected {len(snip_sents)}, "
                        f"got {len(result['decontextualized'])}"
                    )
                doc_out["failed_snippets"].append({
                    "key": key,
                    "reason": reason,
                })
                total_failed_snippets += 1
                for sent in snip_sents:
                    doc_out["statements"].append({
                        "original": sent,
                        "decontextualized": sent,
                        "changed": False,
                    })
                    total_unchanged += 1

        doc_out["num_statements"] = len(doc_out["statements"])
        dataset["documents"].append(doc_out)

    dataset["metadata"]["total_changed_statements"] = total_changed
    dataset["metadata"]["total_unchanged_statements"] = total_unchanged
    dataset["metadata"]["total_failed_snippets"] = total_failed_snippets

    output_path = BATCH_DIR / "decontextualized_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n=== Collection Complete ===")
    print(f"Output: {output_path} ({output_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Documents:            {len(dataset['documents'])}")
    print(f"Statements changed:   {total_changed:,}")
    print(f"Statements unchanged: {total_unchanged:,}")
    print(f"Failed snippets:      {total_failed_snippets}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch decontextualization via Gemini batch API",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Create JSONL request files and manifest")
    p_prep.add_argument(
        "--dataset", default="data/test_dataset/test_dataset.parquet",
        help="Path to dataset parquet (relative to project root)",
    )
    p_prep.add_argument(
        "--snippet-size", type=int, default=SNIPPET_SIZE,
        help=f"Target snippet size in characters (default: {SNIPPET_SIZE})",
    )

    sub.add_parser("submit", help="Upload JSONL and create batch jobs")
    sub.add_parser("status", help="Check batch job status")
    sub.add_parser("collect", help="Download results and create dataset")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    load_dotenv(PROJECT_ROOT / ".env")

    {"prepare": cmd_prepare, "submit": cmd_submit,
     "status": cmd_status, "collect": cmd_collect}[args.command](args)


if __name__ == "__main__":
    main()
