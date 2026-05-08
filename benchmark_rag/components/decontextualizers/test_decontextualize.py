#!/usr/bin/env python
"""
Test decontextualization quality on long legal documents.

Retrieves the top 5 longest documents from the test dataset, lets the user
pick one, extracts snippets at sentence boundaries, and runs the
GeminiDecontextualizer on the first N snippets.  Results (original vs.
decontextualized) are saved to a timestamped text file with cost at the top.

Usage (from project root, with .venv activated):

    python -m benchmark_rag.components.decontextualizers.test_decontextualize

    # Pick second-longest doc, 2000-char snippets, skip first 5000 chars
    python -m benchmark_rag.components.decontextualizers.test_decontextualize \
        --doc-index 1 --snippet-size 2000 --offset 5000
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import nltk
import pandas as pd
from dotenv import load_dotenv

log = logging.getLogger(__name__)

DEFAULT_SNIPPET_SIZE = 1500  # ~one legal paragraph (numbered [N] blocks avg 500-1500 chars)
SCRIPT_DIR = Path(__file__).resolve().parent


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using NLTK Punkt tokenizer."""
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    return tokenizer.tokenize(text)


def _group_into_snippets(
    sentences: list[str], target_size: int
) -> list[tuple[str, list[str]]]:
    """Group sentences into snippets of approximately *target_size* characters.

    Returns a list of ``(snippet_text, constituent_sentences)`` tuples.
    """
    snippets: list[tuple[str, list[str]]] = []
    current_sents: list[str] = []
    current_len = 0

    for sent in sentences:
        added_len = len(sent) + (1 if current_sents else 0)
        if current_len + added_len > target_size and current_sents:
            snippets.append((" ".join(current_sents), list(current_sents)))
            current_sents = [sent]
            current_len = len(sent)
        else:
            current_sents.append(sent)
            current_len += added_len

    if current_sents:
        snippets.append((" ".join(current_sents), list(current_sents)))

    return snippets


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test decontextualization on long legal documents",
    )
    parser.add_argument(
        "--doc-index", type=int, default=0,
        help="Index into documents sorted by length descending (0 = longest, default: 0)",
    )
    parser.add_argument(
        "--snippet-size", type=int, default=DEFAULT_SNIPPET_SIZE,
        help=f"Target snippet size in characters (default: {DEFAULT_SNIPPET_SIZE})",
    )
    parser.add_argument(
        "--offset", type=int, default=0,
        help="Character offset from document start to skip (default: 0)",
    )
    parser.add_argument(
        "--num-snippets", type=int, default=5,
        help="Number of snippets to decontextualize (default: 5)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(SCRIPT_DIR),
        help=f"Output directory (default: {SCRIPT_DIR})",
    )
    parser.add_argument(
        "--dataset", type=str,
        default="data/test_dataset/test_dataset.parquet",
        help="Path to dataset parquet file (relative to project root)",
    )
    parser.add_argument(
        "--include-statutes", action="store_true", default=False,
        help="Include statute documents in ranking (statutes are excluded by default)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    project_root = Path(__file__).resolve().parents[3]
    load_dotenv(project_root / ".env")

    # ---- Load dataset, sort by length, show neighbourhood around selected index ----
    dataset_path = project_root / args.dataset
    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        sys.exit(1)

    df = pd.read_parquet(dataset_path)
    if not args.include_statutes:
        n_before = len(df)
        df = df[df["court"] != "STATUTE"]
        print(f"(Skipped {n_before - len(df)} statute documents)")
    df_sorted = df.sort_values("char_count", ascending=False).reset_index(drop=True)

    if args.doc_index < 0 or args.doc_index >= len(df_sorted):
        print(f"--doc-index must be 0–{len(df_sorted) - 1}")
        sys.exit(1)

    # Show a window of 5 documents around the selected index
    window_start = max(0, args.doc_index - 2)
    window_end = min(len(df_sorted), window_start + 5)
    window_start = max(0, window_end - 5)

    print(f"\n=== Documents by length (showing [{window_start}–{window_end - 1}] "
          f"of {len(df_sorted)}) ===")
    for i in range(window_start, window_end):
        row = df_sorted.iloc[i]
        marker = "  <-- selected" if i == args.doc_index else ""
        print(
            f"  [{i}] {row['citation']:30s} | "
            f"{row['char_count']:>8,} chars | "
            f"{row['name'][:60]}{marker}"
        )
    print()

    doc = df_sorted.iloc[args.doc_index]
    full_text: str = doc["text"]
    citation: str = doc["citation"]
    url: str = doc.get("url", "N/A")
    name: str = doc.get("name", "N/A")

    print(f"Selected: {citation} ({len(full_text):,} chars)")
    print(f"URL: {url}")
    print(f"Snippet size: {args.snippet_size} chars | Offset: {args.offset} chars")
    print()

    # ---- Extract snippets ----
    text_after_offset = full_text[args.offset:]
    if not text_after_offset.strip():
        print("No text remaining after offset.")
        sys.exit(1)

    sentences = _split_sentences(text_after_offset)
    snippets = _group_into_snippets(sentences, args.snippet_size)

    if not snippets:
        print("No snippets could be extracted.")
        sys.exit(1)

    n_snippets = min(args.num_snippets, len(snippets))
    print(
        f"Extracted {len(snippets)} total snippets; "
        f"decontextualizing first {n_snippets}.\n"
    )

    # ---- Decontextualize ----
    from benchmark_rag.components.decontextualizers.gemini_decontextualizer import (
        GeminiDecontextualizer,
    )

    decontextualizer = GeminiDecontextualizer()

    results: list[dict] = []
    for i in range(n_snippets):
        snippet_text, snippet_sents = snippets[i]
        print(
            f"--- Snippet {i + 1}/{n_snippets} "
            f"({len(snippet_sents)} sentences, {len(snippet_text)} chars) ---"
        )

        decontextualized = decontextualizer.decontextualize(
            statements=snippet_sents,
            document_text=full_text,
        )

        if decontextualized is None:
            print("  [FAILED] Could not parse model response; keeping originals.")
            decontextualized = list(snippet_sents)
            failed = True
            raw_responses = list(decontextualizer._last_raw_responses)
        else:
            failed = False
            raw_responses = []

        results.append({
            "snippet_idx": i,
            "original_sentences": snippet_sents,
            "raw_responses": raw_responses,
            "decontextualized_sentences": decontextualized,
            "original_text": snippet_text,
            "failed": failed,
        })

        for j, (orig, decon) in enumerate(zip(snippet_sents, decontextualized)):
            if orig.strip() != decon.strip():
                print(f"  [{j + 1}] CHANGED:")
                o = orig[:120] + ("..." if len(orig) > 120 else "")
                d = decon[:120] + ("..." if len(decon) > 120 else "")
                print(f"       ORIG:  {o}")
                print(f"       DECON: {d}")
        print()

    # ---- Save results ----
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_citation = (
        citation.replace(" ", "_").replace(",", "").replace("/", "_")
    )
    output_path = output_dir / f"decontext_{safe_citation}_{timestamp}.txt"

    total_cost = decontextualizer._total_cost or 0.0
    total_in = decontextualizer._total_input_tokens
    total_out = decontextualizer._total_output_tokens

    with open(output_path, "w") as f:
        f.write("=== Decontextualization Test Run ===\n")
        f.write(f"Estimated total cost: ${total_cost:.6f}\n")
        f.write(f"Total input tokens:   {total_in:,}\n")
        f.write(f"Total output tokens (incl. thinking): {total_out:,}\n")
        f.write(f"Model: {decontextualizer.model_name}\n")
        f.write(f"Temperature: {decontextualizer.temperature}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"\nCitation: {citation}\n")
        f.write(f"URL: {url}\n")
        f.write(f"Document name: {name}\n")
        f.write(f"Document length: {len(full_text):,} chars\n")
        f.write(f"Snippet size: {args.snippet_size} chars\n")
        f.write(f"Offset: {args.offset} chars\n")
        f.write(f"Snippets decontextualized: {n_snippets}\n")
        f.write(f"{'=' * 80}\n")

        for r in results:
            idx = r["snippet_idx"]
            failed_tag = " [FAILED — originals kept]" if r["failed"] else ""
            f.write(f"\n{'=' * 80}\n")
            f.write(f"SNIPPET {idx + 1}{failed_tag}\n")
            f.write(f"{'=' * 80}\n\n")

            f.write("--- ORIGINAL TEXT ---\n")
            f.write(f"{r['original_text']}\n\n")

            f.write("--- DECONTEXTUALIZED SENTENCES ---\n")
            for j, (orig, decon) in enumerate(
                zip(r["original_sentences"], r["decontextualized_sentences"])
            ):
                changed = orig.strip() != decon.strip()
                tag = " [CHANGED]" if changed else " [UNCHANGED]"
                f.write(f"\n  Sentence {j + 1}{tag}:\n")
                f.write(f"    Original:          {orig}\n")
                f.write(f"    Decontextualized:  {decon}\n")

            if r.get("raw_responses"):
                for attempt_idx, raw in enumerate(r["raw_responses"], 1):
                    f.write(f"\n--- RAW MODEL RESPONSE (attempt {attempt_idx}) ---\n")
                    f.write(raw)
                    f.write("\n")

            f.write("\n")

    print(f"Results saved to: {output_path}")
    print(f"Total estimated cost: ${total_cost:.6f}")
    decontextualizer.log_usage_summary()


if __name__ == "__main__":
    main()
