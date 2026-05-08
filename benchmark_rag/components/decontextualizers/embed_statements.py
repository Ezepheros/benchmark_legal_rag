#!/usr/bin/env python
"""
Embed decontextualized statements for faithfulness retrieval.

Splits the dataset into N shards for parallel SLURM jobs, embeds each shard
with Qwen3-Embedding-8B, then stitches all shards into a single FAISS index.

Subcommands:
    prepare  — Split dataset into N shards (default 8).
    embed    — Embed a single shard (run via SLURM array job).
    stitch   — Combine all shard embeddings into one FAISS index.
    query    — Interactive test: find k closest statements to a query.

Usage:
    PROJ=/ubc/cs/research/nlp-raid/students/ethanz01/cs-masters/benchmark_legal_rag
    VENV=$PROJ/.venv/bin/activate

    # 1. Prepare shards
    source $VENV && python -m benchmark_rag.components.decontextualizers.embed_statements prepare

    # 2. Submit SLURM array job (see slurm/decontext_embed.sbatch)
    sbatch slurm/decontext_embed.sbatch

    # 3. Stitch into final index
    source $VENV && python -m benchmark_rag.components.decontextualizers.embed_statements stitch

    # 4. Test queries
    source $VENV && python -m benchmark_rag.components.decontextualizers.embed_statements query "Was the search warrant valid?"
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BATCH_DIR = Path(__file__).resolve().parent / "batch_output"
EMBED_DIR = BATCH_DIR / "embeddings"

DATASET_PATH = BATCH_DIR / "final_decontextualized_dataset.json"
DEFAULT_N_SHARDS = 8
MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
BATCH_SIZE = 16
DEVICE = "cuda:0"


# ---------------------------------------------------------------------------
# prepare — split statements into N shards
# ---------------------------------------------------------------------------

def cmd_prepare(args: argparse.Namespace) -> None:
    with open(DATASET_PATH) as f:
        ds = json.load(f)

    # Build flat list of (doc_idx, stmt_idx, citation, statement_text)
    records: list[dict] = []
    for doc in ds["documents"]:
        citation = doc["citation"]
        for i, stmt in enumerate(doc["decontextualized_statements"]):
            records.append({
                "citation": citation,
                "stmt_idx": i,
                "text": stmt,
            })

    n = args.n_shards
    total = len(records)
    shard_size = (total + n - 1) // n

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    for shard_id in range(n):
        start = shard_id * shard_size
        end = min(start + shard_size, total)
        shard = records[start:end]

        shard_path = EMBED_DIR / f"shard_{shard_id:03d}.json"
        with open(shard_path, "w", encoding="utf-8") as f:
            json.dump(shard, f, ensure_ascii=False)

    meta = {
        "n_shards": n,
        "total_statements": total,
        "shard_size": shard_size,
        "model": MODEL_NAME,
        "dataset": str(DATASET_PATH),
    }
    with open(EMBED_DIR / "embed_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"=== Prepared {n} shards ===")
    print(f"Total statements: {total:,}")
    print(f"Shard size:       ~{shard_size:,}")
    print(f"Output dir:       {EMBED_DIR}")
    for i in range(n):
        p = EMBED_DIR / f"shard_{i:03d}.json"
        cnt = len(json.loads(p.read_text()))
        print(f"  shard_{i:03d}.json: {cnt:,} statements")


# ---------------------------------------------------------------------------
# embed — embed a single shard
# ---------------------------------------------------------------------------

def cmd_embed(args: argparse.Namespace) -> None:
    shard_id = args.shard_id
    shard_path = EMBED_DIR / f"shard_{shard_id:03d}.json"
    if not shard_path.exists():
        print(f"Shard file not found: {shard_path}")
        sys.exit(1)

    with open(shard_path) as f:
        records = json.load(f)

    texts = [r["text"] for r in records]
    print(f"Embedding shard {shard_id}: {len(texts):,} statements with {args.model}...")

    from benchmark_rag.components.embedders.qwen import QwenEmbedder

    embedder = QwenEmbedder(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )

    # Embed in chunks to show progress
    all_embeddings: list[list[float]] = []
    chunk_size = args.batch_size * 10
    for start in range(0, len(texts), chunk_size):
        end = min(start + chunk_size, len(texts))
        batch = texts[start:end]
        embs = embedder.embed(batch)
        all_embeddings.extend(embs)
        print(f"  {end:,}/{len(texts):,} done")

    emb_array = np.array(all_embeddings, dtype=np.float32)

    out_path = EMBED_DIR / f"shard_{shard_id:03d}_embeddings.npy"
    np.save(out_path, emb_array)

    print(f"Saved {emb_array.shape} to {out_path.name}")
    print(f"Embedding dim: {emb_array.shape[1]}")


# ---------------------------------------------------------------------------
# stitch — combine all shards into a single FAISS index
# ---------------------------------------------------------------------------

def cmd_stitch(args: argparse.Namespace) -> None:
    import faiss

    meta_path = EMBED_DIR / "embed_meta.json"
    if not meta_path.exists():
        print("No embed_meta.json found. Run 'prepare' first.")
        sys.exit(1)

    meta = json.loads(meta_path.read_text())
    n_shards = meta["n_shards"]

    # Load all shard embeddings and metadata
    all_embeddings: list[np.ndarray] = []
    all_records: list[dict] = []

    for i in range(n_shards):
        emb_path = EMBED_DIR / f"shard_{i:03d}_embeddings.npy"
        shard_path = EMBED_DIR / f"shard_{i:03d}.json"

        if not emb_path.exists():
            print(f"Missing embeddings for shard {i}: {emb_path}")
            sys.exit(1)

        emb = np.load(emb_path)
        with open(shard_path) as f:
            records = json.load(f)

        if len(emb) != len(records):
            print(f"Shard {i}: embedding count ({len(emb)}) != record count ({len(records)})")
            sys.exit(1)

        all_embeddings.append(emb)
        all_records.extend(records)
        print(f"  Loaded shard {i}: {len(emb):,} embeddings")

    combined = np.vstack(all_embeddings).astype(np.float32)
    dim = combined.shape[1]
    print(f"\nCombined: {combined.shape[0]:,} vectors, dim={dim}")

    # Build FAISS index (Inner Product on L2-normalized vectors = cosine similarity)
    faiss.normalize_L2(combined)
    index = faiss.IndexFlatIP(dim)
    index.add(combined)

    index_path = EMBED_DIR / "decontext_statements.faiss"
    faiss.write_index(index, str(index_path))

    # Save the metadata alongside
    records_path = EMBED_DIR / "decontext_statements_meta.pkl"
    with open(records_path, "wb") as f:
        pickle.dump(all_records, f)

    print(f"\n=== Stitch Complete ===")
    print(f"FAISS index: {index_path} ({index_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Metadata:    {records_path} ({records_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Total vectors: {index.ntotal:,}")
    print(f"Dimension:     {dim}")


# ---------------------------------------------------------------------------
# query — interactive test
# ---------------------------------------------------------------------------

def cmd_query(args: argparse.Namespace) -> None:
    import faiss
    from benchmark_rag.components.embedders.qwen import QwenEmbedder

    index_path = EMBED_DIR / "decontext_statements.faiss"
    records_path = EMBED_DIR / "decontext_statements_meta.pkl"

    if not index_path.exists() or not records_path.exists():
        print("No index found. Run 'stitch' first.")
        sys.exit(1)

    print(f"Loading index ({index_path.stat().st_size / 1e6:.0f} MB)...")
    index = faiss.read_index(str(index_path))
    with open(records_path, "rb") as f:
        records = pickle.load(f)

    print(f"Loading embedder ({args.model})...")
    embedder = QwenEmbedder(
        model_name=args.model,
        device=args.device,
        batch_size=1,
        prompt_name="query",
    )

    query = args.query_text
    k = args.k

    print(f"\nQuery: {query}")
    print(f"Top-{k} results:\n")

    q_emb = np.array(embedder.embed([query]), dtype=np.float32)
    faiss.normalize_L2(q_emb)
    scores, indices = index.search(q_emb, k)

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        rec = records[idx]
        print(f"  [{rank + 1}] score={score:.4f} | {rec['citation']}")
        print(f"      {rec['text'][:150]}")
        print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed decontextualized statements for faithfulness retrieval",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_prep = sub.add_parser("prepare", help="Split dataset into N shards")
    p_prep.add_argument("--n-shards", type=int, default=DEFAULT_N_SHARDS,
                        help=f"Number of shards (default: {DEFAULT_N_SHARDS})")

    p_embed = sub.add_parser("embed", help="Embed a single shard")
    p_embed.add_argument("shard_id", type=int, help="Shard index (0-based)")
    p_embed.add_argument("--model", default=MODEL_NAME, help=f"Model name (default: {MODEL_NAME})")
    p_embed.add_argument("--device", default=DEVICE, help=f"Torch device (default: {DEVICE})")
    p_embed.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                         help=f"Batch size (default: {BATCH_SIZE})")

    p_stitch = sub.add_parser("stitch", help="Combine shard embeddings into FAISS index")

    p_query = sub.add_parser("query", help="Test query against the index")
    p_query.add_argument("query_text", type=str, help="Query text")
    p_query.add_argument("--k", type=int, default=5, help="Number of results (default: 5)")
    p_query.add_argument("--model", default=MODEL_NAME, help=f"Model name (default: {MODEL_NAME})")
    p_query.add_argument("--device", default=DEVICE, help=f"Torch device (default: {DEVICE})")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    {"prepare": cmd_prepare, "embed": cmd_embed,
     "stitch": cmd_stitch, "query": cmd_query}[args.command](args)


if __name__ == "__main__":
    main()
