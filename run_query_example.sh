#!/bin/bash
#SBATCH --job-name=run_query_example
#SBATCH --nodes=1
#SBATCH --partition=b40x4-long
#SBATCH --mem=4G
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/run_query_example.out
#SBATCH --error=logs/run_query_example.err

timestamp=$(date +"%y%m%d_%H%M%S")

# Activate conda env (module load first in SLURM; then activate by prefix)
if command -v module &>/dev/null; then
  module load miniconda/3
fi
source "$(conda info --base)/bin/activate" /lustre/nvwulf/scratch/solhapark/envs/infinigram

# Load HF_TOKEN for meta-llama/Llama-2-7b-hf (gated model)
if [[ -f /lustre/nvwulf/scratch/solhapark/pretrain-trace/.env ]]; then
  set -a
  source /lustre/nvwulf/scratch/solhapark/pretrain-trace/.env
  set +a
fi

cd /lustre/nvwulf/scratch/solhapark/pretrain-trace
echo "=== Job started at $(date) ==="
START_TIME=$(date +%s)

python query_example.py

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== Job finished at $(date) ==="
printf "=== Elapsed time: %d days %02d:%02d:%02d (total %d seconds)\n" \
  $((ELAPSED/86400)) $(((ELAPSED%86400)/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"
