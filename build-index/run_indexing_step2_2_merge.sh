#!/bin/bash
#SBATCH --job-name=index_step2_2
#SBATCH --nodes=1
# tokenized.0=714GB => merge needs >714GB RAM; only orion has 1.5TB
#SBATCH --nodelist=orion
#SBATCH --cpus-per-task=16
#SBATCH --mem=900G
#SBATCH --time=2-00:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -eo pipefail

# Paths (same style as setup_infinigram_env.sh / setup_infinigram_pkg.sh)
HOME="${HOME:-/home/solhapark}"
ENV_DIR="${HOME}/envs/infinigram"
REPO_ROOT="/home/solhapark/pretrain-trace"

# === Edit here: 1 = phase 1 only, 2 = phase 2 only, empty = full merge (all 16 workers) ===
MERGE_PHASE=""

# Set locale for Python encoding
export LANG=en_US.UTF-8
export LC_ALL=en_US.UTF-8

# Redirect output and error to timestamped log files
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
mkdir -p logs
exec > >(tee "logs/indexing_step2_2_${TIMESTAMP}.out")
exec 2> >(tee "logs/indexing_step2_2_${TIMESTAMP}.err" >&2)

# Activate conda env (conda already on PATH; no module load)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_DIR}"

# Load HF_TOKEN
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

cd /home/solhapark/pretrain-trace/infini-gram/pkg

SAVE_DIR="/home/solhapark/pretrain-trace/index"
TEMP_DIR="${SAVE_DIR}"
CPUS=16
TOKEN_WIDTH=2  # u16 = 2 bytes

echo "=== Step 2.2 (merge) started at $(date) ==="
START_TIME=$(date +%s)

# Check prerequisites
DS_PATH="${SAVE_DIR}/tokenized.0"
PARTS_DIR="${TEMP_DIR}/parts-0"

if [[ ! -f "${DS_PATH}" ]]; then
  echo "ERROR: tokenized.0 not found."
  exit 1
fi

if [[ ! -d "${PARTS_DIR}" ]] || [[ -z "$(ls -A "${PARTS_DIR}" 2>/dev/null)" ]]; then
  echo "ERROR: parts-0 not found or empty. Run Step 2.1 (make-part) first."
  exit 1
fi

# Calculate parameters
DS_SIZE=$(stat -f%z "${DS_PATH}" 2>/dev/null || stat -c%s "${DS_PATH}")
HACK=100000
RATIO=$(python3 -c "import math; print(int(math.ceil(math.log2(${DS_SIZE}) / 8)))")

echo "Ratio: ${RATIO}, Token width: ${TOKEN_WIDTH}"
echo "Parts directory: ${PARTS_DIR}"
echo "Number of part files: $(find "${PARTS_DIR}" -type f | wc -l)"

# Optional: split merge into two 2-day jobs (set MERGE_PHASE above)
# Phase 1: workers 0..8 → merged-0/0000..0007.  Phase 2: workers 8..16 → merged-0/0008..0015.
# Submit: sbatch run_indexing_step2_2_merge.sh

MERGED_DIR="${TEMP_DIR}/merged-0"
if [[ -n "${MERGE_PHASE}" ]]; then
  if [[ "${MERGE_PHASE}" == "1" ]]; then
    rm -rf "${MERGED_DIR}"
    WORKER_RANGE_START=0
    WORKER_RANGE_END=8
    echo "Merge phase 1: workers ${WORKER_RANGE_START}..${WORKER_RANGE_END} (output 0000..0007)"
  elif [[ "${MERGE_PHASE}" == "2" ]]; then
    mkdir -p "${MERGED_DIR}"
    WORKER_RANGE_START=8
    WORKER_RANGE_END=16
    echo "Merge phase 2: workers ${WORKER_RANGE_START}..${WORKER_RANGE_END} (output 0008..0015). Do NOT delete merged-0."
  else
    echo "ERROR: MERGE_PHASE must be 1 or 2 (or unset for full merge)"
    exit 1
  fi
else
  rm -rf "${MERGED_DIR}"
  WORKER_RANGE_START=""
  WORKER_RANGE_END=""
  echo "Merge: full run (all 16 workers)"
fi
mkdir -p "${MERGED_DIR}"

# Avoid "Too many open files": 448 parts + N workers each open 448 files => need 448 + N*448 (e.g. 16 workers = 7616)
MIN_OPEN=8192
ulimit -n 65536 2>/dev/null || true
CURRENT=$(ulimit -n)
if [[ "${CURRENT}" -lt "${MIN_OPEN}" ]]; then
  echo "ERROR: ulimit -n too low (current: ${CURRENT}, need at least ${MIN_OPEN}). Merge will fail with 'Too many open files'."
  exit 1
fi
echo "ulimit -n: ${CURRENT} (OK for merge)"

# Run merge (with optional worker range)
RUST_INDEXING_BIN="${REPO_ROOT}/infini-gram/pkg/infini_gram/rust_indexing"
if [[ ! -f "${RUST_INDEXING_BIN}" ]]; then
  echo "ERROR: rust_indexing not found at ${RUST_INDEXING_BIN}"
  echo "Rebuild after main.rs changes: cd infini-gram/pkg && cargo build --release && cp -f target/release/rust_indexing infini_gram/rust_indexing"
  exit 127
fi
if [[ ! -x "${RUST_INDEXING_BIN}" ]]; then
  echo "ERROR: rust_indexing not executable: ${RUST_INDEXING_BIN}"
  exit 127
fi

MERGE_CMD=(
  "${RUST_INDEXING_BIN}" merge
  --data-file "${DS_PATH}"
  --parts-dir "${PARTS_DIR}"
  --merged-dir "${MERGED_DIR}"
  --num-threads "${CPUS}"
  --hacksize "${HACK}"
  --ratio "${RATIO}"
  --token-width "${TOKEN_WIDTH}"
)
if [[ -n "${WORKER_RANGE_START}" ]]; then
  MERGE_CMD+=(--worker-range-start "${WORKER_RANGE_START}" --worker-range-end "${WORKER_RANGE_END}")
fi
echo "Executing: ${MERGE_CMD[*]}"

# Handle SIGTERM (e.g. from scancel) so merge process exits cleanly and node does not enter DRAIN
cleanup_merge() {
  echo "Received SIGTERM/SIGINT, shutting down merge (PID ${MERGE_PID:-?}) gracefully..."
  [[ -n "${MERGE_PID:-}" ]] && kill -TERM "${MERGE_PID}" 2>/dev/null
  [[ -n "${MERGE_PID:-}" ]] && wait "${MERGE_PID}" 2>/dev/null
  exit 143
}
trap cleanup_merge SIGTERM SIGINT

"${MERGE_CMD[@]}" &
MERGE_PID=$!
wait "${MERGE_PID}"
MERGE_EXIT=$?
trap - SIGTERM SIGINT

if [[ ${MERGE_EXIT} -ne 0 ]]; then
  echo "ERROR: merge failed (exit ${MERGE_EXIT})"
  exit 1
fi

# Clean up parts directory (optional, can keep for debugging)
# rm -rf "${PARTS_DIR}"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "=== Step 2.2 (merge) finished at $(date) ==="
printf "=== Elapsed time: %d days %02d:%02d:%02d (total %d seconds)\n" \
  $((ELAPSED/86400)) $(((ELAPSED%86400)/3600)) $(((ELAPSED%3600)/60)) $((ELAPSED%60)) "$ELAPSED"

echo ""
if [[ "${MERGE_PHASE}" == "1" ]]; then
  echo "=== Next step: Submit again with MERGE_PHASE=2 (same script, 2-day job) ==="
elif [[ "${MERGE_PHASE}" == "2" ]]; then
  echo "=== Next step: Run run_indexing_step2_3_concat.sh ==="
else
  echo "=== Next step: Run run_indexing_step2_3_concat.sh ==="
fi