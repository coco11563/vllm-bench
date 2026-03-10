#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

: "${VENV_DIR:?Set VENV_DIR to the vLLM virtualenv path, e.g. /data/envs/vllm_qwen35/.venv}"
: "${MODEL_ROOT:?Set MODEL_ROOT to the model directory, e.g. /data/model_repo}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the benchmark output directory}"

PYTHON_BIN="${PYTHON_BIN:-${VENV_DIR}/bin/python}"
SESSION_NAME="${SESSION_NAME:-vllm_bench_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/runner.log}"
export PATH="${VENV_DIR}/bin:${PATH}"

command -v tmux >/dev/null 2>&1 || {
  echo "tmux not found" >&2
  exit 1
}

mkdir -p "${OUTPUT_ROOT}"
export PYTHON_BIN MODEL_ROOT OUTPUT_ROOT
export HOST BASE_PORT GPUS CONCURRENCY INPUT_LEN OUTPUT_LEN MIN_PROMPTS
export PROMPTS_PER_CONCURRENCY SERVER_TIMEOUT GPU_MEMORY_UTILIZATION
export MAX_MODEL_LEN DTYPE REPEATS RUNNER PREPARE_MODELS DOWNLOAD_SCRIPT HF_MAX_WORKERS

tmux new-session -d -s "${SESSION_NAME}" \
  "cd \"${SCRIPT_DIR}\" && bash \"${SCRIPT_DIR}/final_run_all_models.sh\" > \"${LOG_FILE}\" 2>&1"

echo "session=${SESSION_NAME}"
echo "log=${LOG_FILE}"
