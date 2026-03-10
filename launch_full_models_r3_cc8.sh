#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/remote_seq_bench.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/benchmarks/vllm_qwen35_full_r3_cc8}"
REPEATS="${REPEATS:-3}"

"${PYTHON_BIN}" "${RUNNER}" \
  --output-root "${OUTPUT_ROOT}" \
  --host 127.0.0.1 \
  --gpus 1,2,4,8 \
  --concurrency 8 \
  --input-len 256 \
  --output-len 128 \
  --min-prompts 32 \
  --prompts-per-concurrency 1 \
  --server-timeout 600 \
  --repeats "${REPEATS}" \
  --resume-ok
