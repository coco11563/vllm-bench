#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNNER="${RUNNER:-${SCRIPT_DIR}/remote_seq_bench.py}"

COMMON_ARGS=(
  --output-root /data/benchmarks/vllm_qwen35_seq
  --host 127.0.0.1
  --concurrency 1,8,32
  --input-len 256
  --output-len 128
  --min-prompts 32
  --prompts-per-concurrency 2
  --server-timeout 600
  --resume-ok
)

run_batch() {
  echo "==== $*"
  "${PYTHON_BIN}" "${RUNNER}" "${COMMON_ARGS[@]}" "$@"
}

# 1) 所有模型至少跑一次，优先选择更可能成功且节省时间的卡数。
run_batch --gpus 1 \
  --model Qwen3.5-0.8B \
  --model Qwen3.5-0.8B-Base \
  --model Qwen3.5-2B \
  --model Qwen3.5-2B-Base \
  --model Qwen3.5-4B \
  --model Qwen3.5-4B-Base \
  --model Qwen3.5-9B \
  --model Qwen3.5-9B-Base \
  --model Qwen3.5-27B-FP8

# 先提前跑完 30B 左右中等规模模型的 8 卡结果。
run_batch --gpus 8 \
  --model Qwen3.5-27B \
  --model Qwen3.5-27B-FP8 \
  --model Qwen3.5-35B-A3B \
  --model Qwen3.5-35B-A3B-Base \
  --model Qwen3.5-35B-A3B-FP8

run_batch --gpus 2 \
  --model Qwen3.5-27B \
  --model Qwen3.5-35B-A3B \
  --model Qwen3.5-35B-A3B-Base \
  --model Qwen3.5-35B-A3B-FP8

run_batch --gpus 4 \
  --model Qwen3.5-122B-A10B \
  --model Qwen3.5-122B-A10B-FP8

run_batch --gpus 8 \
  --model Qwen3.5-397B-A17B

# 2) 代表模型扩展测试，覆盖小/中/大模型的卡数扩展趋势。
run_batch --concurrency 8 --min-prompts 32 --prompts-per-concurrency 1 --gpus 1,2,4,8 --model Qwen3.5-9B
run_batch --concurrency 8 --min-prompts 32 --prompts-per-concurrency 1 --gpus 1,2,4,8 --model Qwen3.5-27B
run_batch --concurrency 8 --min-prompts 32 --prompts-per-concurrency 1 --gpus 1,2,4,8 --model Qwen3.5-122B-A10B-FP8
