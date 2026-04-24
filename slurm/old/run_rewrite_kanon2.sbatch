#!/bin/bash
#SBATCH --job-name=RewriteQuerySuite
#SBATCH --output=slurm_logs/run_%j_%x.out
#SBATCH --error=slurm_logs/run_%j_%x.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=150G
#SBATCH --time=48:00:00
#SBATCH --partition=nlpgpo
set -e

export HF_HOME=/ubc/cs/research/nlp-raid/students/ethanz01/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

cd /ubc/cs/research/nlp-raid/students/ethanz01/cs-masters/benchmark_legal_rag
source .venv/bin/activate
export PYTHONUNBUFFERED=1

# Indexes are shared with the base experiments — indexing is skipped if already built.

# --- Kanon2 (index shared with kanon2_recursive_1024) ---
python scripts/run_indexing.py --config configs/experiments/kanon2_recursive_1024.yaml
python scripts/run_benchmark_rewrite_query.py \
    --config configs/experiments/kanon2_recursive_1024_rewrite_query.yaml