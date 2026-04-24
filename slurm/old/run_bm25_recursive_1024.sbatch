#!/bin/bash
#SBATCH --job-name=BM25Rec1024
#SBATCH --output=slurm_logs/run_%j_%x.out
#SBATCH --error=slurm_logs/run_%j_%x.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --time=24:00:00
#SBATCH --partition=nlpgpo
set -e

export HF_HOME=/ubc/cs/research/nlp-raid/students/ethanz01/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

cd /ubc/cs/research/nlp-raid/students/ethanz01/cs-masters/benchmark_legal_rag
source .venv/bin/activate
export PYTHONUNBUFFERED=1

# 1. Build the index (shared with qwen_recursive_1024_1k-docs — skipped if it already exists)
python scripts/run_indexing.py --config configs/experiments/bm25_recursive_1024.yaml

# 2. BM25 benchmark (CPU-only, uses chunks.pkl from the index above)
python scripts/run_bm25_benchmark.py --config configs/experiments/bm25_recursive_1024.yaml
