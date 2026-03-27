# benchmark_legal_rag

Benchmarking framework for Retrieval-Augmented Generation on Canadian legal documents.

Designed to compare combinations of:
- **Embedders** — Qwen3-8B, Gemini, Kanon2 (legal-domain)
- **Chunkers** — Recursive character splitting (1024 / 4096 char)
- **Retrievers** — FAISS dense retrieval
- **Generators** — Gemini 2.5 Flash
- **Judge** — Gemini 2.5 Pro (LLM-as-a-judge)

---

## Project layout

```
benchmark_legal_rag/
├── benchmark_rag/                  # Installable Python package
│   ├── components/
│   │   ├── base.py                 # Abstract base classes + dataclasses
│   │   ├── chunkers/
│   │   │   ├── recursive.py        # RecursiveChunker (primary)
│   │   │   ├── naive.py            # Fixed-size baseline
│   │   │   └── semantic.py        # Embedding-guided semantic chunker
│   │   ├── embedders/
│   │   │   ├── qwen.py             # Qwen3-Embedding-8B
│   │   │   ├── gemini.py           # Gemini Embedding
│   │   │   └── kanon2.py          # Isaacus Kanon2 (legal domain)
│   │   ├── generators/
│   │   │   └── gemini.py           # GeminiGenerator (Flash) + GeminiJudge (Pro)
│   │   └── retrievers/
│   │       └── faiss_retriever.py  # FAISS with save/load
│   ├── config/
│   │   └── schemas.py              # Pydantic v2 config schemas
│   ├── evaluation/
│   │   └── metrics.py              # Recall, Precision, Hit, MRR, nDCG@k
│   ├── pipeline/
│   │   ├── indexing_pipeline.py    # chunk → embed → build FAISS index
│   │   └── rag_pipeline.py         # query → retrieve → (generate)
│   ├── logging.py                  # Experiment-scoped logging (.log + .jsonl)
│   └── registry.py                 # Component factory from YAML type strings
├── configs/
│   ├── base.yaml                   # Shared defaults
│   └── experiments/                # One YAML per experiment (inherits base)
│       ├── qwen_recursive_1024.yaml
│       ├── gemini_recursive_4096.yaml
│       └── kanon2_recursive_1024.yaml
├── data/
│   └── test_dataset/               # Symlink → ../../data_collection_site/test_dataset/
│       ├── test_dataset.parquet    #   documents (citation, text, court, source, …)
│       └── queries.json            #   queries + ground_truth_citations
├── runs/                           # Output of experiments (gitignored)
│   └── <experiment_id>/
│       ├── index/                  # FAISS index + chunk metadata
│       ├── logs/                   # .log and .jsonl per experiment
│       └── results/                # metrics.json + query_results.jsonl
├── scripts/
│   ├── run_indexing.py             # Step 1: build index
│   └── run_benchmark.py            # Step 2: evaluate
├── environment.yml
└── pyproject.toml
```

---

## Setup

### 1. Create the virtual environment

```bash
# From the benchmark_legal_rag/ directory
python -m venv .venv

# Activate (Linux / Mac)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate
```

### 2. Install PyTorch with CUDA

Install PyTorch separately first so you can target the right CUDA version.
Check [pytorch.org](https://pytorch.org/get-started/locally/) for your system's command.

```bash
# Example: CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU-only
pip install torch
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the package

```bash
pip install -e .
```

This makes `import benchmark_rag` work from anywhere without `sys.path` hacks.

### 3. Set API keys

```bash
export GOOGLE_API_KEY=...        # Gemini embedder, generator, judge
export ISAACUS_API_KEY=...       # Kanon2 embedder
```

### 4. Build the test dataset

Run `build_test_dataset.py` (one directory up) to produce the document corpus and query file:

```bash
# 1000-doc corpus
python ../build_test_dataset.py \
    --samples_per_court 50 \
    --output_dir data/test_dataset/

# 10000-doc corpus
python ../build_test_dataset.py \
    --samples_per_court 500 \
    --output_dir data/test_dataset_10k/
```

This writes:
- `test_dataset.parquet` — documents with columns: `citation`, `name`, `court`, `text`, `url`, `source`, `is_ground_truth`, `ground_truth_query_ids`, `ground_truth_query_texts`, `snippets`
- `queries.json` — list of queries with `query_id`, `query_text`, `user_answer`, `ground_truth_citations` (list)

---

## Running an experiment

Every experiment is two commands.

### Step 1 — Index

Chunks the corpus, embeds it, and builds a FAISS index.

```bash
python scripts/run_indexing.py --config configs/experiments/qwen_recursive_1024.yaml
```

Output written to `runs/qwen_recursive_1024_1k/`:
```
index/
├── index.faiss             ← FAISS vectors
├── index.chunks.pkl        ← chunk objects (text + metadata + embeddings)
└── chunks_metadata.parquet ← text + metadata only (no embeddings, for inspection)
logs/
├── qwen_recursive_1024_1k.log    ← human-readable
└── qwen_recursive_1024_1k.jsonl  ← one JSON object per log line
```

### Step 2 — Benchmark

Runs retrieval (and optionally generation + judging) and saves metrics.

```bash
# Retrieval only
python scripts/run_benchmark.py --config configs/experiments/qwen_recursive_1024.yaml

# With answer generation (requires generator: in config)
python scripts/run_benchmark.py --config configs/experiments/qwen_recursive_1024.yaml --generate

# With generation + LLM judge
python scripts/run_benchmark.py --config configs/experiments/qwen_recursive_1024.yaml --generate --judge
```

Output written to `runs/qwen_recursive_1024_1k/results/`:
```
metrics.json            ← recall/precision/hit/MRR/nDCG@5,25,100
query_results.jsonl     ← one line per query: query_id, query_text, gold_citations, retrieved_ids, answer
```

---

## Config system

### How inheritance works

Every experiment YAML declares a `base_config` and overrides only what differs:

```yaml
# configs/experiments/my_experiment.yaml
base_config: ../../configs/base.yaml

experiment_id: my_experiment
description: "Kanon2 + 4096 char chunks"

chunker:
  max_chunk_chars: 4096
  overlap_chars: 256

embedder:
  type: embedders.kanon2.Kanon2Embedder
  model_name: kanon-2
```

Keys are deep-merged — you only need to specify what changes. Everything else comes from [configs/base.yaml](configs/base.yaml).

### How the `type` field works

The `type` field in any component config is a dotted path resolved relative to `benchmark_rag/components/`:

```yaml
embedder:
  type: embedders.qwen.QwenEmbedder     # → benchmark_rag/components/embedders/qwen.py :: QwenEmbedder
  model_name: Qwen/Qwen3-Embedding-8B
  device: cuda:0
```

Any additional fields in the block are forwarded as `**kwargs` to the class constructor. This is handled by [benchmark_rag/registry.py](benchmark_rag/registry.py).

### Full config reference

#### `dataset`
| Key | Default | Description |
|---|---|---|
| `name` | `test_dataset` | Identifier used in logs |
| `path` | `data/test_dataset/test_dataset.parquet` | Path to parquet file from `build_test_dataset.py` |
| `max_docs` | `null` | Subsample to N documents; `null` = use all |

#### `chunker`
| Key | Default | Description |
|---|---|---|
| `type` | `chunkers.recursive.RecursiveChunker` | Component class |
| `max_chunk_chars` | `1024` | Hard upper bound on chunk length |
| `overlap_chars` | `128` | Characters of overlap between consecutive chunks |

#### `embedder`
| Key | Default | Description |
|---|---|---|
| `type` | `embedders.qwen.QwenEmbedder` | Component class |
| `model_name` | `Qwen/Qwen3-Embedding-8B` | HuggingFace ID or API model name |
| `device` | `cuda:0` | Torch device (ignored by API-based embedders) |
| `batch_size` | `16` | Texts per forward pass |

#### `retriever`
| Key | Default | Description |
|---|---|---|
| `type` | `retrievers.faiss_retriever.FaissRetriever` | Component class |
| `metric` | `cosine` | `cosine` or `l2` |

#### `generator` (optional)
| Key | Default | Description |
|---|---|---|
| `type` | — | e.g. `generators.gemini.GeminiGenerator` |
| `model_name` | — | e.g. `gemini-2.5-flash` |
| `temperature` | `0.0` | Sampling temperature |
| `max_output_tokens` | `1024` | Max generated tokens |

#### `evaluation`
| Key | Default | Description |
|---|---|---|
| `queries_path` | `data/test_dataset/queries.json` | Path to queries file |
| `k_values` | `[5, 25, 100]` | Cutoffs for retrieval metrics |
| `metrics` | all | `recall_at_k`, `precision_at_k`, `hit_at_k`, `mrr`, `ndcg_at_k` |

#### `indexing`
| Key | Default | Description |
|---|---|---|
| `document_batch_size` | `32` | Documents chunked per batch |
| `embedding_batch_size` | `16` | Chunks embedded per forward pass |
| `save_intermediate_every` | `500` | Save progress every N chunks |
| `output_dir` | `runs/{experiment_id}/index` | `{experiment_id}` is substituted automatically |

#### `logging`
| Key | Default | Description |
|---|---|---|
| `level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `log_dir` | `runs/{experiment_id}/logs` | Where `.log` and `.jsonl` are written |
| `resource_monitor_interval` | `30.0` | Seconds between CPU/RAM/GPU snapshots; `0` = disabled |

---

## Modifying runs

### Change chunker size

```yaml
chunker:
  max_chunk_chars: 4096
  overlap_chars: 256
```

### Switch embedder

```yaml
# Gemini
embedder:
  type: embedders.gemini.GeminiEmbedder
  model_name: models/gemini-embedding-exp-03-07
  task_type: RETRIEVAL_DOCUMENT
  batch_size: 50

# Kanon2
embedder:
  type: embedders.kanon2.Kanon2Embedder
  model_name: kanon-2
  batch_size: 64
```

### Use a subset of documents

```yaml
dataset:
  max_docs: 1000
```

### Enable generation

```yaml
generator:
  type: generators.gemini.GeminiGenerator
  model_name: gemini-2.5-flash
  temperature: 0.0
  max_output_tokens: 1024
```

Then run with `--generate` (and `--judge` to score the outputs).

### Change evaluation cutoffs or metrics

```yaml
evaluation:
  k_values: [1, 5, 10, 50]
  metrics:
    - recall_at_k
    - hit_at_k
    - mrr
```

---

## Extending the code

### Adding a new embedder

1. Create `benchmark_rag/components/embedders/my_embedder.py`:

```python
from benchmark_rag.components.base import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, api_key: str | None = None):
        super().__init__()
        self.model_name = model_name
        # ... initialise client

    @property
    def embedding_dim(self) -> int:
        return 1536  # your model's output dim

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # ... call your API or model
        return embeddings  # list of float lists
```

2. Use it in a config — no other changes needed:

```yaml
embedder:
  type: embedders.my_embedder.MyEmbedder
  model_name: my-model-v1
```

### Adding a new chunker

1. Create `benchmark_rag/components/chunkers/my_chunker.py`:

```python
from benchmark_rag.components.base import BaseChunker, Chunk, Document

class MyChunker(BaseChunker):
    def __init__(self, **kwargs):
        ...

    def chunk(self, document: Document) -> list[Chunk]:
        # Return a list of Chunk objects
        return [Chunk(text=..., doc_id=document.doc_id, chunk_idx=i, metadata=document.metadata)]
```

2. Reference it in a config:

```yaml
chunker:
  type: chunkers.my_chunker.MyChunker
```

### Adding a new generator

1. Create `benchmark_rag/components/generators/my_generator.py`:

```python
from benchmark_rag.components.base import BaseGenerator, RetrievedChunk

class MyGenerator(BaseGenerator):
    def __init__(self, model_name: str, **kwargs):
        ...

    def generate(self, query: str, context_chunks: list[RetrievedChunk]) -> str:
        # Build a prompt from query + chunks, call your LLM, return answer string
        ...
```

2. Reference it in a config:

```yaml
generator:
  type: generators.my_generator.MyGenerator
  model_name: my-llm
```

---

## Debugging

### Inspect logs

Every run writes two log files to `runs/<experiment_id>/logs/`:

- `<experiment_id>.log` — timestamped human-readable lines
- `<experiment_id>.jsonl` — structured, one JSON object per line

Load the JSONL for programmatic inspection:

```python
import pandas as pd
df = pd.read_json("runs/qwen_recursive_1024_1k/logs/qwen_recursive_1024_1k.jsonl", lines=True)
df[df["level"] == "WARNING"]
```

### Inspect indexed chunks

```python
import pandas as pd
df = pd.read_parquet("runs/qwen_recursive_1024_1k/index/chunks_metadata.parquet")

# How many chunks per document?
df.groupby("citation").size().describe()

# Look at chunks for a specific case
df[df["citation"] == "2022 ONCA 100"][["chunk_idx", "text"]].head(10)
```

### Test a single query interactively

```python
from benchmark_rag.config.schemas import ExperimentConfig
from benchmark_rag.pipeline.rag_pipeline import RAGPipeline

cfg = ExperimentConfig.from_yaml("configs/experiments/qwen_recursive_1024.yaml")
pipeline = RAGPipeline.from_config(cfg)  # loads saved index

result = pipeline.query("What constitutes wrongful dismissal in Ontario?", k=10)
for chunk in result.retrieved_chunks:
    print(f"{chunk.score:.3f}  {chunk.doc_id}  {chunk.text[:100]}")
```

### Compare experiments

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

---

## Implemented experiments

| Experiment | Embedder | Chunk size | Corpus |
|---|---|---|---|
| `qwen_recursive_1024_1k` | Qwen3-8B | 1024 char | 1 000 docs |
| `gemini_recursive_4096_1k` | Gemini | 4096 char | 1 000 docs |
| `kanon2_recursive_1024_1k` | Kanon2 | 1024 char | 1 000 docs |
