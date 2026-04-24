#!/bin/bash
#SBATCH --job-name=GeminiSuite1024
#SBATCH --output=slurm_logs/run_%j_%x.out
#SBATCH --error=slurm_logs/run_%j_%x.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=nlpgpo
set -e

cd /ubc/cs/research/nlp-raid/students/ethanz01/cs-masters/benchmark_legal_rag
source .venv/bin/activate
export PYTHONUNBUFFERED=1

# 1. Index once (shared by both normal RAG and IterRetGen)
python scripts/run_indexing.py --config configs/experiments/gemini_recursive_1024.yaml

# 2. Normal RAG benchmark
python scripts/run_benchmark.py --config configs/experiments/gemini_recursive_1024.yaml

# 3. IterRetGen benchmark (reuses the index built above)
python scripts/run_benchmark.py \
    --config configs/experiments/gemini_recursive_1024_iterretgen.yaml \
    --iterretgen
