#!/bin/bash
#SBATCH --job-name=index_step2_3
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -eo pipefail

# Paths (same as run_indexing_step2_2_merge.sh)
HOME="${HOME:-/home/solhapark}"
ENV_DIR="${HOME}/envs/infinigram"
REPO_ROOT="/home/solhapark/pretrain-trace"

# Set locale for Python encoding
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Redirect output and error to timestamped log files
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
mkdir -p logs
exec > >(tee "logs/indexing_step2_3_${TIMESTAMP}.out")
exec 2> >(tee "logs/indexing_step2_3_${TIMESTAMP}.err" >&2)

# Activate conda env (conda already on PATH; no module load)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"

# Load HF_TOKEN
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

cd "${REPO_ROOT}/infini-gram/pkg"

SAVE_DIR="${REPO_ROOT}/index"
TEMP_DIR="${SAVE_DIR}"
CPUS=16
TOKEN_WIDTH=2  # u16 = 2 bytes

echo "=== Step 2.3 (concat) started at $(date) ==="
START_TIME=$(date +%s)

# Check prerequisites
DS_PATH="${SAVE_DIR}/tokenized.0"
MERGED_DIR="${TEMP_DIR}/merged-0"
SA_PATH="${SAVE_DIR}/table.0"

if [[ ! -f "${DS_PATH}" ]]; then
  echo "ERROR: tokenized.0 not found."
  exit 1
fi

if [[ ! -d "${MERGED_DIR}" ]] || [[ -z "$(ls -A "${MERGED_DIR}" 2>/dev/null)" ]]; then
  echo "ERROR: merged-0 not found or empty. Run Step 2.2 (merge) first."
  exit 1
fi

if [[ -f "${SA_PATH}" ]]; then
  echo "WARNING: table.0 already exists. Will be overwritten."
fi

# Calculate parameters
DS_SIZE=$(stat -f%z "${DS_PATH}" 2>/dev/null || stat -c%s "${DS_PATH}")
RATIO=$(python3 -c "import math; print(int(math.ceil(math.log2(${DS_SIZE}) / 8)))")

echo "Ratio: ${RATIO}, Token width: ${TOKEN_WIDTH}"
echo "Merged directory: ${MERGED_DIR}"
echo "Number of merged files: $(find "${MERGED_DIR}" -type f | wc -l)"
echo "Output file: ${SA_PATH}"

# Run concat
"${REPO_ROOT}/infini-gram/pkg/infini_gram/rust_indexing" concat \
  --data-file "${DS_PATH}" \
  --merged-dir "${MERGED_DIR}" \
  --merged-file "${SA_PATH}" \
  --num-threads "${CPUS}" \
  --ratio "${RATIO}" \
  --token-width "${TOKEN_WIDTH}"

if [[ $? -ne 0 ]]; then
  echo "ERROR: concat failed"
  exit 1
fi

# Verify output
if [[ -f "${SA_PATH}" ]]; then
  SA_SIZE=$(stat -f%z "${SA_PATH}" 2>/dev/null || stat -c%s "${SA_PATH}")
  echo "Success! table.0 created: ${SA_SIZE} bytes ($(numfmt --to=iec-i --suffix=B ${SA_SIZE}))"
else
  echo "ERROR: table.0 was not created"
  exit 1
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== Step 2.3 (concat) finished at $(date) ==="
printf "=== Elapsed time: %d days %02d:%02d:%02d (total %d seconds)\n" \
  $((ELAPSED/86400)) $(((ELAPSED%86400)/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"

echo ""
echo "=== Indexing complete! ==="
echo "Final index files:"
ls -lh "${SAVE_DIR}"/*.0
