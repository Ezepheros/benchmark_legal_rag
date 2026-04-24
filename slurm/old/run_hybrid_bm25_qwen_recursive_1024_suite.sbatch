#!/bin/bash
#SBATCH --job-name=HybridBM25Qwen1024
#SBATCH --output=slurm_logs/run_%j_%x.out
#SBATCH --error=slurm_logs/run_%j_%x.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --time=36:00:00
#SBATCH --partition=nlpgpo
set -e

export HF_HOME=/ubc/cs/research/nlp-raid/students/ethanz01/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

cd /ubc/cs/research/nlp-raid/students/ethanz01/cs-masters/benchmark_legal_rag
source .venv/bin/activate
export PYTHONUNBUFFERED=1

# 1. Build the hybrid index (FAISS + BM25).
#    - If the Qwen FAISS index already exists from a prior qwen_recursive_1024
#      run, it is reused and only the BM25 index is rebuilt from the cached
#      chunks — no re-embedding required.
#    - If neither index exists, both are built from scratch (GPU required for
#      Qwen embedding).
python scripts/run_indexing.py \
    --config configs/experiments/hybrid_bm25_qwen_recursive_1024.yaml

# 2. Hybrid retrieval — RRF fusion (no reranker, no API calls).
#    Fuses FAISS and BM25 ranked lists using Reciprocal Rank Fusion.
python scripts/run_benchmark.py \
    --config configs/experiments/hybrid_bm25_qwen_recursive_1024.yaml

# 3. Hybrid retrieval — Kanon2 cross-encoder reranker.
#    Merges FAISS + BM25 candidates then reranks with Kanon2 ($0.35 / 1M tokens).
#    Reuses the index built in step 1.
python scripts/run_benchmark.py \
    --config configs/experiments/hybrid_bm25_qwen_kanon2rerank_recursive_1024.yaml
