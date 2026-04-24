#!/bin/bash
#SBATCH --job-name=BM25ParamSuite
#SBATCH --output=slurm_logs/run_%j_%x.out
#SBATCH --error=slurm_logs/run_%j_%x.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --partition=nlpgpo
set -e

export HF_HOME=/ubc/cs/research/nlp-raid/students/ethanz01/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

cd /ubc/cs/research/nlp-raid/students/ethanz01/cs-masters/benchmark_legal_rag
source .venv/bin/activate
export PYTHONUNBUFFERED=1

# Build the shared index once (same dataset + chunker + embedder across all four runs)
python scripts/run_indexing.py --config configs/experiments/bm25_recursive_1024.yaml

# k1=0.5, b=0.1
# python scripts/run_bm25_benchmark.py --config configs/experiments/bm25_recursive_1024_k1lo_blo.yaml

# # k1=0.5, b=0.9
# python scripts/run_bm25_benchmark.py --config configs/experiments/bm25_recursive_1024_k1lo_bhi.yaml

# # k1=2.0, b=0.1
# python scripts/run_bm25_benchmark.py --config configs/experiments/bm25_recursive_1024_k1hi_blo.yaml

# # k1=2.0, b=0.9
# python scripts/run_bm25_benchmark.py --config configs/experiments/bm25_recursive_1024_k1hi_bhi.yaml
