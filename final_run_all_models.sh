#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/remote_seq_bench.py}"

: "${MODEL_ROOT:?Set MODEL_ROOT to the model directory, e.g. /data/model_repo}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to the benchmark output directory}"

HOST="${HOST:-127.0.0.1}"
BASE_PORT="${BASE_PORT:-8200}"
GPUS="${GPUS:-1,2,4,8}"
CONCURRENCY="${CONCURRENCY:-8}"
INPUT_LEN="${INPUT_LEN:-256}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
MIN_PROMPTS="${MIN_PROMPTS:-32}"
PROMPTS_PER_CONCURRENCY="${PROMPTS_PER_CONCURRENCY:-1}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-600}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-auto}"
REPEATS="${REPEATS:-3}"
PREPARE_MODELS="${PREPARE_MODELS:-0}"
DOWNLOAD_SCRIPT="${DOWNLOAD_SCRIPT:-}"

mkdir -p "${OUTPUT_ROOT}"

if [[ "${PREPARE_MODELS}" == "1" ]]; then
  : "${DOWNLOAD_SCRIPT:?Set DOWNLOAD_SCRIPT when PREPARE_MODELS=1, e.g. /data/model_repo/download_qwen35.py}"
  "${PYTHON_BIN}" "${DOWNLOAD_SCRIPT}"
fi

"${PYTHON_BIN}" "${RUNNER}" \
  --model-root "${MODEL_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --host "${HOST}" \
  --base-port "${BASE_PORT}" \
  --gpus "${GPUS}" \
  --concurrency "${CONCURRENCY}" \
  --input-len "${INPUT_LEN}" \
  --output-len "${OUTPUT_LEN}" \
  --min-prompts "${MIN_PROMPTS}" \
  --prompts-per-concurrency "${PROMPTS_PER_CONCURRENCY}" \
  --server-timeout "${SERVER_TIMEOUT}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --dtype "${DTYPE}" \
  --repeats "${REPEATS}" \
  --resume-any \
  "$@"
