#!/bin/bash
#SBATCH --job-name=RunKanon21024
#SBATCH --output=slurm_logs/run_%j_%x.out
#SBATCH --error=slurm_logs/run_%j_%x.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --partition=nlpgpo
set -e

cd /ubc/cs/research/nlp-raid/students/ethanz01/cs-masters/benchmark_legal_rag
source .venv/bin/activate
export PYTHONUNBUFFERED=1

python scripts/run_indexing.py --config configs/experiments/kanon2_recursive_1024.yaml
python scripts/run_benchmark.py --config configs/experiments/kanon2_recursive_1024.yaml
