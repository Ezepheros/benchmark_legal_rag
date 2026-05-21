"""
Merge Qwen bf16 and 8-bit quantized answer files.

Takes bf16 answers as primary, fills OOM/failed gaps with 8-bit answers.
Output: qwen_answers.jsonl (used by build_eval_ui_data.py)

Usage:
    python scripts/merge_qwen_answers.py
    python scripts/merge_qwen_answers.py --bf16 path/to/bf16.jsonl --eight-bit path/to/8bit.jsonl
"""
import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "runs" / "multi_generator_50" / "results"


def main():
    parser = argparse.ArgumentParser(description="Merge Qwen bf16 + 8-bit answer files.")
    parser.add_argument("--bf16", default=str(RESULTS_DIR / "qwen_1gpu_answers.jsonl"),
                        help="Path to bf16 answers")
    parser.add_argument("--eight-bit", default=str(RESULTS_DIR / "qwen_8bit_answers.jsonl"),
                        help="Path to 8-bit quantized answers")
    parser.add_argument("--output", default=str(RESULTS_DIR / "qwen_answers.jsonl"),
                        help="Output path")
    args = parser.parse_args()

    with open(args.bf16) as f:
        bf16 = {json.loads(l)["query_id"]: json.loads(l) for l in f if l.strip()}
    print(f"Loaded {len(bf16)} bf16 rows ({sum(1 for r in bf16.values() if r.get('answer'))} with answers)")

    with open(args.eight_bit) as f:
        eight_bit = {json.loads(l)["query_id"]: json.loads(l) for l in f if l.strip()}
    print(f"Loaded {len(eight_bit)} 8-bit rows ({sum(1 for r in eight_bit.values() if r.get('answer'))} with answers)")

    merged = []
    for qid in sorted(set(bf16) | set(eight_bit)):
        bf16_row = bf16.get(qid)
        if bf16_row and bf16_row.get("answer"):
            merged.append(bf16_row)
        elif qid in eight_bit and eight_bit[qid].get("answer"):
            row = eight_bit[qid]
            row["note"] = "from_8bit_quantized"
            merged.append(row)
        elif bf16_row:
            merged.append(bf16_row)

    with open(args.output, "w") as f:
        for r in merged:
            f.write(json.dumps(r) + "\n")

    n_bf16 = sum(1 for r in merged if r.get("note") != "from_8bit_quantized" and r.get("answer"))
    n_8bit = sum(1 for r in merged if r.get("note") == "from_8bit_quantized")
    n_failed = sum(1 for r in merged if not r.get("answer"))
    print(f"\nMerged: {len(merged)} total | {n_bf16} bf16 | {n_8bit} 8bit fallback | {n_failed} failed")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
