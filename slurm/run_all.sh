#!/bin/bash
# Submit all experiment SLURM jobs in parallel.
# Each job handles its own indexing (skipped if index already exists).
#
# Usage:
#   bash slurm/run_all.sh          # submit everything
#   bash slurm/run_all.sh baseline # submit only baseline jobs
#   bash slurm/run_all.sh reranker # submit only reranker jobs
#   etc.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"  # project root

mkdir -p slurm_logs

GROUP="${1:-all}"

submit_group() {
    local dir="$1"
    local label="$2"
    echo "=== Submitting $label jobs ==="
    for f in "$dir"/*.sbatch; do
        [ -f "$f" ] || continue
        echo "  sbatch $f"
        sbatch "$f"
    done
    echo ""
}

if [ "$GROUP" = "all" ] || [ "$GROUP" = "baseline" ]; then
    submit_group "$SCRIPT_DIR/baseline" "Baseline"
fi

if [ "$GROUP" = "all" ] || [ "$GROUP" = "reranker" ]; then
    submit_group "$SCRIPT_DIR/reranker" "Reranker"
fi

if [ "$GROUP" = "all" ] || [ "$GROUP" = "hybrid" ]; then
    submit_group "$SCRIPT_DIR/hybrid" "Hybrid"
fi

if [ "$GROUP" = "all" ] || [ "$GROUP" = "bm25" ]; then
    submit_group "$SCRIPT_DIR/bm25" "BM25"
fi

if [ "$GROUP" = "all" ] || [ "$GROUP" = "query_rewrite" ]; then
    submit_group "$SCRIPT_DIR/query_rewrite" "Query Rewrite"
fi

if [ "$GROUP" = "all" ] || [ "$GROUP" = "iterretgen" ]; then
    submit_group "$SCRIPT_DIR/iterretgen" "IterRetGen"
fi

echo "All requested jobs submitted. Check status with: squeue -u \$USER"
