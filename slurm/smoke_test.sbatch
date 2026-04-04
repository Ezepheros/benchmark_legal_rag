#!/bin/bash
#SBATCH --job-name=SmokeTest
#SBATCH --output=slurm_logs/smoke_%j.out
#SBATCH --error=slurm_logs/smoke_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --partition=nlpgpo
set -e

export HF_HOME=/ubc/cs/research/nlp-raid/students/ethanz01/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

cd /ubc/cs/research/nlp-raid/students/ethanz01/cs-masters/benchmark_legal_rag
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python scripts/run_indexing.py --config configs/experiments/smoke_test.yaml
python scripts/run_benchmark.py --config configs/experiments/smoke_test.yaml
