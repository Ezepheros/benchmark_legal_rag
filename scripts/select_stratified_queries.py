"""
Select 50 stratified queries for the multi-generator comparison experiment.

Stratifies by law area (proportional to actual distribution in usable queries),
then within each law area samples across provinces for diversity.

Output: data/test_dataset/stratified_50_queries.json

Usage:
    python scripts/select_stratified_queries.py
    python scripts/select_stratified_queries.py --n 50 --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT_ROOT = Path(__file__).parent.parent
QUERIES_PATH = PROJECT_ROOT / "data" / "test_dataset" / "queries.json"
BATCHES_PATH = PROJECT_ROOT / "data" / "test_dataset" / "batches_law_areas.csv"


def main():
    parser = argparse.ArgumentParser(description="Select stratified queries for multi-generator experiment.")
    parser.add_argument("--n", type=int, default=50, help="Number of queries to select")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default=None, help="Output path (default: data/test_dataset/stratified_50_queries.json)")
    args = parser.parse_args()

    import pandas as pd

    # Load batch → law area mapping
    batches = pd.read_csv(BATCHES_PATH)
    batch_to_area = dict(zip(batches["batch number"], batches["cleaned_law_area"]))

    # Load queries, filter to usable (have gold citations)
    queries = json.loads(QUERIES_PATH.read_text())
    usable = [q for q in queries if q.get("ground_truth_citations")]

    # Annotate each query with law_area and province
    for q in usable:
        q["_law_area"] = batch_to_area.get(q["batch_id"], "unknown")
        q["_province"] = q.get("province", "Unknown")

    # Count actual distribution
    area_counts = Counter(q["_law_area"] for q in usable)
    total = len(usable)

    # Compute proportional targets
    targets: dict[str, int] = {}
    for area, count in area_counts.most_common():
        targets[area] = max(1, round(count / total * args.n))

    # Adjust to exactly n
    current_sum = sum(targets.values())
    if current_sum > args.n:
        # Trim from largest areas
        for area in sorted(targets, key=lambda a: targets[a], reverse=True):
            if current_sum <= args.n:
                break
            targets[area] -= 1
            current_sum -= 1
    elif current_sum < args.n:
        # Add to largest areas
        for area in sorted(targets, key=lambda a: targets[a], reverse=True):
            if current_sum >= args.n:
                break
            targets[area] += 1
            current_sum += 1

    print(f"Usable queries: {total}")
    print(f"Target: {args.n} queries")
    print(f"\nAllocation by law area:")
    for area in sorted(targets, key=lambda a: targets[a], reverse=True):
        print(f"  {area:30s}: {targets[area]:2d} (from {area_counts[area]:3d}, {100*area_counts[area]/total:.1f}%)")

    # Group queries by law area
    by_area: dict[str, list[dict]] = defaultdict(list)
    for q in usable:
        by_area[q["_law_area"]].append(q)

    # Sample with province diversity
    rng = random.Random(args.seed)
    selected: list[dict] = []

    for area, target_n in targets.items():
        pool = by_area[area]
        if target_n >= len(pool):
            chosen = pool
        else:
            # Sort by province, then shuffle within each province group
            by_province: dict[str, list[dict]] = defaultdict(list)
            for q in pool:
                by_province[q["_province"]].append(q)
            for v in by_province.values():
                rng.shuffle(v)

            # Round-robin across provinces
            provinces = sorted(by_province.keys())
            rng.shuffle(provinces)
            chosen = []
            idx = 0
            while len(chosen) < target_n:
                prov = provinces[idx % len(provinces)]
                if by_province[prov]:
                    chosen.append(by_province[prov].pop(0))
                idx += 1
                if idx > len(provinces) * len(pool):
                    break

        selected.extend(chosen)

    # Remove temp fields
    for q in selected:
        q.pop("_law_area", None)
        q.pop("_province", None)

    rng.shuffle(selected)

    # Summary
    final_areas = Counter(batch_to_area.get(q["batch_id"], "unknown") for q in selected)
    final_provinces = Counter(q.get("province", "Unknown") for q in selected)

    print(f"\nSelected {len(selected)} queries")
    print(f"\nFinal law area distribution:")
    for area, count in final_areas.most_common():
        print(f"  {area:30s}: {count}")
    print(f"\nFinal province distribution:")
    for prov, count in final_provinces.most_common():
        print(f"  {prov:30s}: {count}")

    # Save
    output_path = Path(args.output) if args.output else PROJECT_ROOT / "data" / "test_dataset" / f"stratified_{args.n}_queries.json"
    output_path.write_text(json.dumps(selected, indent=2, ensure_ascii=False))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
