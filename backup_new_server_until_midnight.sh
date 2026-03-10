#!/bin/zsh
set -euo pipefail

REMOTE_HOST="111.127.53.159"
REMOTE_PORT="2201"
REMOTE_USER="root"
REMOTE_DIR="/ssd/benchmarks/vllm_full_r1_cc8"
SSH_KEY="/Users/mengxiao/.ssh/sc_2"
SSH_OPTS=(
  -i "${SSH_KEY}"
  -o StrictHostKeyChecking=no
  -o ControlMaster=no
  -o ControlPath=none
  -o ControlPersist=no
  -p "${REMOTE_PORT}"
)

LOCAL_ROOT="/Volumes/980Pro/Tabular-Condension/vllm-bench/results/backups/111.127.53.159_2201"
LOCAL_PARENT="${LOCAL_ROOT}"
LOG_FILE="${LOCAL_ROOT}/backup_live.log"
STOP_AT="2026-03-08 23:55:00"
SLEEP_SECONDS="600"

mkdir -p "${LOCAL_PARENT}"

backup_once() {
  local ts
  ts="$(date '+%F %T')"
  echo "[${ts}] start backup" >> "${LOG_FILE}"

  ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
    "test -d '${REMOTE_DIR}' && tar cf - -C /ssd/benchmarks vllm_full_r1_cc8" \
    | tar xf - -C "${LOCAL_PARENT}"

  ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
    "test -f '${REMOTE_DIR}/runner.log' && tail -n 40 '${REMOTE_DIR}/runner.log' || true" \
    > "${LOCAL_ROOT}/runner_tail.latest.log"

  echo "[${ts}] backup done" >> "${LOG_FILE}"
}

while true; do
  now_epoch="$(date '+%s')"
  stop_epoch="$(date -j -f '%Y-%m-%d %H:%M:%S' "${STOP_AT}" '+%s')"
  if [[ "${now_epoch}" -ge "${stop_epoch}" ]]; then
    break
  fi

  backup_once || true

  now_epoch="$(date '+%s')"
  remaining="$((stop_epoch - now_epoch))"
  if [[ "${remaining}" -le 0 ]]; then
    break
  fi

  if [[ "${remaining}" -lt "${SLEEP_SECONDS}" ]]; then
    sleep "${remaining}"
  else
    sleep "${SLEEP_SECONDS}"
  fi
done

backup_once || true
