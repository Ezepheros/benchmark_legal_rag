# benchmark_legal_rag

A modular benchmarking framework for evaluating RAG pipelines on Canadian legal documents. The goal is to systematically compare combinations of chunking strategies, embedding models, and retrieval methods to find the best configuration for legal document retrieval.

---

## Breakpointing / Learning Walkthrough

See **[breakpointing.md](breakpointing.md)** for:
- How to run the step-by-step pipeline walkthrough (`scripts/run_breakpoint_demo.py`)
- The `_bp_utils.py` / `_bp_*` helper / `breakpoint()` pattern used throughout the codebase
- How to add breakpoints when writing new pipeline code

---

## Maintaining This Document

**Update this file whenever you make a significant architectural change.** Adding a new component file (embedder, chunker, reranker, etc.) following the existing registry pattern does not require an update — the pattern is already documented below. Changes that *do* require an update include:

- Adding a new pipeline type (e.g. a new `*_pipeline.py`)
- Adding a new retrieval mode or fusion strategy
- Adding a new stage to the pipeline (e.g. a new pre/post-processing step)
- Changing the data flow between existing stages
- Adding a new required environment variable or external dependency
- Changing the index format or on-disk output structure

When in doubt: if the change affects the architecture diagram, the directory structure listing, or the "Key Design Decisions" section, update those sections too.

---

## What This Project Does

Given a corpus of Canadian legal cases and a set of queries (each with known ground-truth documents), the pipeline:

1. **Indexes** — chunks documents, embeds chunks, builds a FAISS index
2. **Retrieves** — embeds a query, searches the index for top-k relevant chunks
3. **Evaluates** — measures recall@k, precision@k, MRR, nDCG, hit@k against ground truth
4. **(Optionally) Generates** — produces an answer from retrieved context using Gemini
5. **(Optionally) Judges** — scores the generated answer against a reference with an LLM judge

The test dataset (built by `cad_rag/data/build_test_dataset.py`) contains:
- Ground truth documents from human annotations (`legal_data_collection_rag`)
- Stratified random samples from the York canadian-case-law dataset (BCSC, SST, TCC, CHRT)
- Stitched court documents from the Caseway dataset

---

## Architecture

### Two-Stage Pipeline

**Indexing is separated from evaluation** so that multiple experiments sharing the same dataset + chunker + embedder can reuse a single index without re-computing embeddings.

```
Stage 1 (run once): Documents → Chunker → Embedder → index (saved to disk)
Stage 2 (run many): Query → Embedder → Retrieval → [Reranker] → Metrics / Generation
```

Three pipeline types share this two-stage structure:

| Pipeline | Retriever | Reranker | File |
|---|---|---|---|
| `RAGPipeline` | FAISS (dense) | — | `pipeline/rag_pipeline.py` |
| `HybridRAGPipeline` | FAISS + BM25 → union | Kanon2 (or RRF fallback) | `pipeline/hybrid_pipeline.py` |
| `IterRetGenPipeline` | FAISS (dense, iterative) | — | `pipeline/iterretgen_pipeline.py` |

**Hybrid pipeline data flow:**
```
Query
 ├─ Qwen embedding → FAISS top-N candidates ─┐
 └─ BM25 tokenise  → BM25  top-N candidates ─┴─ union/dedup
                                                   │
                                          Kanon2Reranker (cross-encoder)
                                          reads (query, doc) pairs directly
                                                   │
                                                top-k results
```
When no reranker is configured, the hybrid pipeline falls back to Reciprocal Rank Fusion (RRF) over the two ranked lists.

The index ID is a deterministic SHA1 hash of `(dataset_path, max_docs, chunker_type, chunker_params, embedder_type, embedder_model)`. Two experiments that differ only in `k_values` or generator will share the same index.

### Component Registry

All components (chunkers, embedders, retrievers, generators) are referenced by dotted type paths in YAML and instantiated at runtime via `benchmark_rag/registry.py`. This means:
- No hardcoded imports in pipeline code
- New components require zero changes to existing files
- Config is the single source of truth for what runs

```yaml
chunker:
  type: chunkers.recursive.RecursiveChunker
  max_chunk_chars: 1024
  overlap_chars: 128
```

### Config Inheritance

Every experiment YAML inherits from `configs/base.yaml` via deep merge. Only changed keys need to appear in the experiment file. Template strings `{experiment_id}` and `{index_id}` are substituted at load time so output paths are always experiment-scoped.

### Abstract Base Classes

All component types in `benchmark_rag/components/base.py` define the contract via ABC. The data containers (`Document`, `Chunk`, `EmbeddedChunk`, `RetrievedChunk`) are plain dataclasses that flow through the pipeline.

---

## Directory Structure

```
benchmark_legal_rag/
├── benchmark_rag/              # Installable Python package
│   ├── registry.py             # Component factory (type path → instance)
│   ├── logging.py              # Structured dual-output logging
│   ├── components/
│   │   ├── base.py             # ABCs and dataclasses
│   │   ├── chunkers/           # recursive (primary), naive, semantic
│   │   ├── embedders/          # qwen (local), gemini (API), kanon2 (legal API)
│   │   ├── retrievers/         # faiss_retriever, bm25_retriever, hybrid_retriever
│   │   ├── rerankers/          # kanon2 (legal cross-encoder API)
│   │   ├── generators/         # GeminiGenerator + GeminiJudge
│   │   └── splitters/          # sentence (NLTK), paragraph (regex)
│   ├── config/schemas.py       # Pydantic v2 config models
│   ├── evaluation/metrics.py   # recall, precision, MRR, nDCG, hit
│   └── pipeline/
│       ├── indexing_pipeline.py
│       ├── rag_pipeline.py
│       ├── hybrid_pipeline.py  # HybridRAGPipeline (FAISS + BM25 + reranker)
│       └── iterretgen_pipeline.py
├── configs/
│   ├── base.yaml               # Shared defaults for all experiments
│   └── experiments/            # One YAML per experiment
├── data/
│   └── test_dataset/           # Built by cad_rag/data/build_test_dataset.py
│       ├── test_dataset.parquet
│       └── queries.json
├── scripts/
│   ├── run_indexing.py         # Entry point: index documents
│   └── run_benchmark.py        # Entry point: evaluate retrieval/generation
└── runs/                       # Output (gitignored)
    ├── indexes/{index_id}/     # Shared FAISS + BM25 indexes
    └── {experiment_id}/        # Per-experiment logs and results
```

---

## Data Format

### Documents (`test_dataset.parquet`)

| Field | Type | Description |
|---|---|---|
| `citation` | `str` | Primary citation — unique document ID (e.g., `2024 BCSC 123`) |
| `citation2` | `str` | Secondary citation where available |
| `name` | `str` | Style of cause |
| `court` | `str` | Court abbreviation (e.g., `BCSC`, `SST`) |
| `text` | `str` | Full document text |
| `url` | `str` | Source URL |
| `source` | `str` | `"ground_truth"`, `"canadian_case_law"`, or `"caseway"` |
| `is_ground_truth` | `bool` | Whether this document was human-annotated as relevant |
| `ground_truth_query_ids` | `JSON list[str]` | Query IDs this document is a ground-truth answer for |
| `ground_truth_query_texts` | `JSON list[str]` | Corresponding query texts (parallel to `ground_truth_query_ids`) |
| `snippets_by_query` | `JSON dict[str, list[str]]` | Annotator-highlighted snippets keyed by `query_id` — for offline analysis only, not used by the eval pipeline |

> **Parquet encoding note:** `ground_truth_query_ids`, `ground_truth_query_texts`, and `snippets_by_query` are stored as JSON strings. `run_indexing.py` deserialises them back to Python objects when building `Document.metadata`. If you add a new JSON-encoded column, add its name to the `if k in (...)` guard in `scripts/run_indexing.py`.

### Queries (`queries.json`)

```json
{
  "query_id": 12,
  "query_text": "...",
  "user_answer": "...",
  "custom_instruction": "...",
  "batch_id": 3,
  "ground_truth_citations": ["2024 BCSC 123", "2021 SCC 5"],
  "ground_truth_snippets": {
    "2024 BCSC 123": ["snippet text A", "snippet text B"],
    "2021 SCC 5":    ["snippet text C"]
  }
}
```

- `ground_truth_citations` drives retrieval evaluation — metrics check whether these citations appear in the top-k retrieved chunk `doc_id`s.
- `ground_truth_snippets` maps each GT citation to the specific passages the annotator highlighted as relevant to *this* query. It is stored for offline analysis and is not used by `run_benchmark.py`.

---

## Environment Setup

Use the `.venv` virtual environment in the project root:

```bash
source .venv/bin/activate
```

All dependencies are installed there. Always activate this environment before running any scripts or tests.

---

## Running Experiments

```bash
# Index (once per dataset + chunker + embedder combo)
python scripts/run_indexing.py --config configs/experiments/qwen_recursive_1024.yaml

# Evaluate retrieval
python scripts/run_benchmark.py --config configs/experiments/qwen_recursive_1024.yaml

# With answer generation
python scripts/run_benchmark.py --config configs/experiments/qwen_recursive_1024.yaml --generate

# With LLM judge scoring
python scripts/run_benchmark.py --config configs/experiments/qwen_recursive_1024.yaml --generate --judge
```

### Required Environment Variables

| Variable | Required for |
|---|---|
| `GOOGLE_API_KEY` | GeminiEmbedder, GeminiGenerator, GeminiJudge |
| `ISAACUS_API_KEY` | Kanon2Embedder, Kanon2Reranker |

---

## Adding a New Component

### New Embedder

```python
# benchmark_rag/components/embedders/my_embedder.py
from benchmark_rag.components.base import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    @property
    def embedding_dim(self) -> int:
        return 1536

    def _embed(self, texts: list[str]) -> list[list[float]]:
        ...
```

```yaml
embedder:
  type: embedders.my_embedder.MyEmbedder
  model_name: my-model-v1
```

No changes to the pipeline, registry, or scripts are needed — the registry resolves the type path automatically.

The same pattern applies for chunkers (`BaseChunker`), retrievers (`BaseRetriever`), generators (`BaseGenerator`), and rerankers (`BaseReranker`).

### Cost Tracking for API Components

Any component that calls a paid external API **must** implement per-call cost logging and a `log_usage_summary()` method, following the pattern in `Kanon2Embedder` and `Kanon2Reranker`:

- `_COST_PER_1M_TOKENS` class constant with the current pricing
- `_total_prompt_tokens` and `_total_est_cost_usd` accumulators
- `_track_and_log(n_texts, call_prompt_tokens, call_est_tokens)` — logs after every API call with running totals; falls back to `chars/4` token estimation when the API does not report usage
- `max_cost_usd` constructor parameter that raises `RuntimeError` before a batch that would exceed the cap
- `log_usage_summary()` — called by the pipeline at the end of a run to print the aggregate token count and cost

The pipeline scripts call `pipeline.log_usage_summary()` automatically, which must delegate to any component that tracks cost.

---

## Key Design Decisions

**Why a registry pattern instead of hardcoded imports?**
Keeps the pipeline code completely decoupled from concrete implementations. Adding a new embedder is a self-contained change — one new file and a YAML reference.

**Why share indexes across experiments?**
Embedding a large corpus is expensive. The deterministic `index_id` hash means two experiments that differ only in `k` values or generator share the same index, avoiding redundant computation.

**Why Pydantic v2 for config?**
Type validation catches bad YAML at load time rather than mid-run. `model_extra = "allow"` on component configs forwards unknown keys as constructor kwargs, so adding a new param to a component doesn't require a config schema change.

**Why separate indexing and evaluation scripts?**
Indexing is I/O-heavy and slow (embedding the corpus). Evaluation is fast and iterative. Keeping them separate lets you tune `k_values`, add a generator, or change eval metrics without touching the index.

**Why cross-encoder reranking over RRF for hybrid retrieval?**
RRF fuses ranked lists using rank position only — it treats all chunks at rank 3 as equally relevant regardless of their actual content. A cross-encoder (Kanon2Reranker) reads the query and each candidate document together and produces a true relevance score. This is substantially stronger for legal retrieval, where a chunk's rank in either sub-retriever is a weak signal but its semantic match to the query is precise. RRF is kept as a no-cost fallback when `reranker: null` is set in the config.

**Why a two-step retrieval (wide candidates → reranker) instead of just a reranker?**
Cross-encoders are O(n) in the number of candidates — scoring 10k chunks per query would be prohibitively slow and expensive. The FAISS + BM25 first stage cheaply narrows the field to a manageable candidate pool (e.g. 100–200 chunks), and the reranker then does precise scoring only on that pool.

**Why citation as the document ID?**
Canadian neutral citations (`YYYY COURT NNN`) are unique per decision and human-readable. They appear directly in annotator-provided ground truth, making evaluation matching straightforward.

---

## Output Structure

```
runs/
├── indexes/
│   └── qwen__recursive1024__a1b2c3d4e5/
│       ├── index.faiss             # FAISS vectors
│       ├── index.chunks.pkl        # Serialized EmbeddedChunk objects
│       └── chunks_metadata.parquet # Text + metadata for inspection
│
└── qwen_recursive_1024_1k/
    ├── config.json                 # Snapshot of resolved config
    ├── logs/
    │   ├── qwen_recursive_1024_1k.log    # Human-readable
    │   └── qwen_recursive_1024_1k.jsonl  # Structured (one JSON per line)
    └── results/
        ├── metrics.json            # Aggregate scores by metric and k
        └── query_results.jsonl     # Per-query retrieved IDs and answers
```

### Comparing Experiments

```python
import json, pathlib, pandas as pd

rows = []
for f in pathlib.Path("runs").glob("*/results/metrics.json"):
    m = json.loads(f.read_text())
    row = {"experiment": m["experiment_id"]}
    for metric, by_k in m["scores"].items():
        for k, v in by_k.items():
            row[f"{metric}@{k}"] = round(v, 4)
    rows.append(row)

pd.DataFrame(rows).sort_values("recall_at_k@100", ascending=False)
```
