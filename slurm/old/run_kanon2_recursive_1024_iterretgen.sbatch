#!/bin/bash
#SBATCH --job-name=Kanon2IterRet1024
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

# Index is shared with kanon2_recursive_1024_1k-docs — skip re-indexing if it exists.
INDEX_DIR=$(python -c "
from benchmark_rag.config.schemas import ExperimentConfig
cfg = ExperimentConfig.from_yaml('configs/experiments/kanon2_recursive_1024_iterretgen.yaml')
print(cfg.indexing.output_dir)
")

if [ ! -f "${INDEX_DIR}/index.faiss" ]; then
    echo "Index not found at ${INDEX_DIR} — building now."
    python scripts/run_indexing.py --config configs/experiments/kanon2_recursive_1024_iterretgen.yaml
else
    echo "Reusing existing index at ${INDEX_DIR}."
fi

python scripts/run_benchmark.py \
    --config configs/experiments/kanon2_recursive_1024_iterretgen.yaml \
    --iterretgen
