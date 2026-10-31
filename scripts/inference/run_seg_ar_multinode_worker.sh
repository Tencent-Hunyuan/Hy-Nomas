#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

source scripts/inference/seg_ar_demo_config.sh

RUNTIME_DIR="${RUNTIME_DIR:-${PWD}/.runtime/seg_ar_demo}"
JOB_DB="${JOB_DB:-/tmp/nomas_seg_ar_jobs.json}"
WORKER_HOSTS_RAW="${NOMAS_WORKER_HOSTS:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_PORT="${MASTER_PORT:-29500}"
RDZV_ID="${RDZV_ID:-nomas-seg-ar-${MASTER_PORT}}"
GENERATION_TIMEOUT_SECONDS="${NOMAS_GENERATION_TIMEOUT_SECONDS:-900}"
SSH_OPTIONS=(
  -o BatchMode=yes
  -o StrictHostKeyChecking=no
  -o ConnectTimeout=10
)

if [[ -z "${WORKER_HOSTS_RAW}" ]]; then
  echo "NOMAS_WORKER_HOSTS is required, for example host-a,host-b" >&2
  exit 2
fi

mapfile -t WORKER_HOSTS < <(
  printf '%s\n' "${WORKER_HOSTS_RAW}" \
    | tr ',;' '\n' \
    | awk '{$1=$1}; NF && !seen[$0]++'
)
NNODES="${#WORKER_HOSTS[@]}"
if (( NNODES < 1 )); then
  echo "NOMAS_WORKER_HOSTS did not contain a usable host" >&2
  exit 2
fi

if (( NNODES * NPROC_PER_NODE != MODEL_PARALLEL_SIZE )); then
  echo "SP_SIZE * TP_SIZE must equal nodes * GPUs per node; got ${MODEL_PARALLEL_SIZE} != ${NNODES} * ${NPROC_PER_NODE}" >&2
  exit 2
fi

MASTER_ADDR="${MASTER_ADDR:-${WORKER_HOSTS[0]}}"
REPO_ROOT="${NOMAS_PROJECT_ROOT:-${PWD}}"
WORKER_SLOT="${WORKER_SLOT:-0}"
WORKER_ID="${NOMAS_WORKER_ID:-$(hostname):seg-ar-sp${SP_SIZE}-slot-${WORKER_SLOT}}"
VISIBLE_GPUS="${CUDA_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC_PER_NODE - 1)))}"

mkdir -p "${RUNTIME_DIR}"

shell_quote() {
  printf '%q' "$1"
}

declare -a FORWARDED_ENV=()
while IFS='=' read -r name value; do
  case "${name}" in
    NOMAS_*|NCCL_*|NVSHMEM_*|UCX_*|PYTORCH_*|TORCH_*|MODEL_ID|DIT_WEIGHT|TRAIN_OUTPUT|EXPERIMENT|DRY_RUN_SOURCE|SAVE_PATH|COLLECTIVE_KEEPALIVE_INTERVAL|MAX_CONCURRENT_JOBS|MIN_FREE_GPU_MEMORY_GIB|ADMISSION_SETTLE_SECONDS|SESSION_IDLE_TIMEOUT|LONG_SESSION_THRESHOLD_SECONDS|MAX_LONG_SESSIONS|EXCLUSIVE_LONG_SESSION|INTERACTIVE_NOISE_MAX_SECONDS|INTERACTIVE_NOISE_HARD_MAX_SECONDS|INTERACTIVE_NOISE_GROWTH_SECONDS|KV_LAYERWISE_CPU_OFFLOAD|KV_CPU_MODE|KV_RESIDENT_BLOCKS|KV_INITIAL_GPU_TOKENS|KV_PREALLOC_SECONDS|KV_MIN_SEGMENT_FRAMES|KV_GROWTH_FACTOR|KV_STAGING_SLOTS)
      FORWARDED_ENV+=("${name}=${value}")
      ;;
  esac
done < <(env)

remote_command() {
  local rank="$1"
  local rank_runtime="${RUNTIME_DIR}/rank_${rank}"
  local rank_pid="${rank_runtime}/worker.pid"
  local rank_log="${rank_runtime}/worker.log"
  local command="set -euo pipefail; "
  command+="cd $(shell_quote "${REPO_ROOT}"); "
  command+="mkdir -p $(shell_quote "${rank_runtime}"); "
  command+="if test -s $(shell_quote "${rank_pid}"); then "
  command+="old_pid=\$(cat $(shell_quote "${rank_pid}")); "
  command+="if kill -0 \"\${old_pid}\" 2>/dev/null; then "
  command+="echo \"rank ${rank} already running as pid \${old_pid}\" >&2; exit 3; "
  command+="fi; fi; "
  command+="nohup setsid env "
  local item
  for item in "${FORWARDED_ENV[@]}"; do
    command+="$(shell_quote "${item}") "
  done
  command+="$(shell_quote "RUNTIME_DIR=${rank_runtime}") "
  command+="$(shell_quote "JOB_DB=${JOB_DB}") "
  command+="$(shell_quote "WORKER_SLOT=${WORKER_SLOT}") "
  command+="$(shell_quote "WORKER_LOCK=/tmp/nomas_seg_ar_sp${SP_SIZE}_${MASTER_PORT}_rank${rank}.lock") "
  command+="$(shell_quote "NOMAS_WORKER_ID=${WORKER_ID}") "
  command+="$(shell_quote "NOMAS_WORKER_REPLICAS=1") "
  command+="$(shell_quote "WORKER_REPLICAS=1") "
  command+="$(shell_quote "SP_SIZE=${SP_SIZE}") "
  command+="$(shell_quote "TP_SIZE=${TP_SIZE}") "
  command+="$(shell_quote "NNODES=${NNODES}") "
  command+="$(shell_quote "NODE_RANK=${rank}") "
  command+="$(shell_quote "NPROC_PER_NODE=${NPROC_PER_NODE}") "
  command+="$(shell_quote "MASTER_ADDR=${MASTER_ADDR}") "
  command+="$(shell_quote "MASTER_PORT=${MASTER_PORT}") "
  command+="$(shell_quote "RDZV_ID=${RDZV_ID}") "
  command+="$(shell_quote "CUDA_VISIBLE_DEVICES=${VISIBLE_GPUS}") "
  command+="bash scripts/inference/run_seg_ar_gpu_worker.sh "
  for item in "${DEMO_INFERENCE_ARGS[@]}"; do
    command+="$(shell_quote "${item}") "
  done
  command+="> $(shell_quote "${rank_log}") 2>&1 < /dev/null & "
  command+="echo \$! > $(shell_quote "${rank_pid}")"
  printf '%s' "${command}"
}

rank_is_alive() {
  local host="$1"
  local rank="$2"
  local rank_pid="${RUNTIME_DIR}/rank_${rank}/worker.pid"
  ssh "${SSH_OPTIONS[@]}" "${host}" \
    "pid=\$(cat $(shell_quote "${rank_pid}") 2>/dev/null || true); test -n \"\${pid}\" && kill -0 \"\${pid}\" 2>/dev/null"
}

stop_rank() {
  local host="$1"
  local rank="$2"
  local rank_pid="${RUNTIME_DIR}/rank_${rank}/worker.pid"
  ssh "${SSH_OPTIONS[@]}" "${host}" bash -s -- "${rank_pid}" <<'REMOTE' || true
set -u
pid_file="$1"
pid="$(cat "${pid_file}" 2>/dev/null || true)"
if [[ -z "${pid}" ]] || ! kill -0 "${pid}" 2>/dev/null; then
  rm -f "${pid_file}"
  exit 0
fi
command="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
case "${command}" in
  *run_seg_ar_gpu_worker.sh*|*torchrun*|*seg_ar_web_server.py*) ;;
  *)
    echo "Refusing to stop unexpected rank process pid=${pid}: ${command}" >&2
    exit 1
    ;;
esac
pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
if [[ -n "${pgid}" ]]; then
  kill -TERM -- "-${pgid}" 2>/dev/null || true
else
  kill -TERM "${pid}" 2>/dev/null || true
fi
for _ in $(seq 1 60); do
  kill -0 "${pid}" 2>/dev/null || break
  sleep 0.5
done
if kill -0 "${pid}" 2>/dev/null; then
  if [[ -n "${pgid}" ]]; then
    kill -KILL -- "-${pgid}" 2>/dev/null || true
  else
    kill -KILL "${pid}" 2>/dev/null || true
  fi
fi
rm -f "${pid_file}"
REMOTE
}

cleanup() {
  trap - TERM INT EXIT
  for rank in "${!WORKER_HOSTS[@]}"; do
    stop_rank "${WORKER_HOSTS[$rank]}" "${rank}" &
  done
  wait || true
}
trap cleanup TERM INT EXIT

start_group() {
  echo "Starting one SP${SP_SIZE}/TP${TP_SIZE} worker across ${NNODES} node(s):"
  for rank in "${!WORKER_HOSTS[@]}"; do
    echo "  rank ${rank}: ${WORKER_HOSTS[$rank]} (${NPROC_PER_NODE} GPUs)"
  done
  echo "  rendezvous: ${MASTER_ADDR}:${MASTER_PORT}"
  echo "  queue:      ${JOB_DB}"
  echo "  runtime:    ${RUNTIME_DIR}"

  local started_ranks=0
  for rank in "${!WORKER_HOSTS[@]}"; do
    host="${WORKER_HOSTS[$rank]}"
    if ! ssh "${SSH_OPTIONS[@]}" "${host}" "$(remote_command "${rank}")"; then
      echo "Failed to launch distributed rank ${rank} on ${host}" >&2
      return 1
    fi
    started_ranks=$((started_ranks + 1))
  done
  if (( started_ranks != NNODES )); then
    echo "Started ${started_ranks}/${NNODES} distributed ranks" >&2
    return 1
  fi
}

reset_worker_after_failure() {
  local reason="$1"
  python3 - "${JOB_DB}" "${WORKER_ID}" "${reason}" <<'PY' || true
import sys
from apps.seg_ar_job_store import JobStore

store = JobStore(sys.argv[1])
worker_id = sys.argv[2]
reason = sys.argv[3]
store.reset_worker(worker_id)
store.record_error(
    code="WORKER_CRASH",
    message=reason,
    worker_id=worker_id,
    recoverable=True,
)
PY
}

restart_count=0
while true; do
  if ! start_group; then
    cleanup
    restart_count=$((restart_count + 1))
    delay=$((restart_count * 5))
    (( delay > 60 )) && delay=60
    reset_worker_after_failure "SP${SP_SIZE}/TP${TP_SIZE} worker failed during startup; restarting in ${delay}s"
    sleep "${delay}"
    trap cleanup TERM INT EXIT
    continue
  fi

  failure_reason=""
  while [[ -z "${failure_reason}" ]]; do
    for rank in "${!WORKER_HOSTS[@]}"; do
      host="${WORKER_HOSTS[$rank]}"
      if ! rank_is_alive "${host}" "${rank}"; then
        failure_reason="SP${SP_SIZE}/TP${TP_SIZE} worker rank ${rank} on ${host} exited"
        break
      fi
    done
    if [[ -z "${failure_reason}" ]] && (( GENERATION_TIMEOUT_SECONDS > 0 )); then
      timed_out="$(
        python3 - "${JOB_DB}" "${GENERATION_TIMEOUT_SECONDS}" "${WORKER_ID}" <<'PY' || true
import sys
from apps.seg_ar_job_store import JobStore

result = JobStore(sys.argv[1]).timeout_running_jobs(float(sys.argv[2]))
if sys.argv[3] in result["worker_ids"]:
    print("yes")
PY
      )"
      if [[ "${timed_out}" == "yes" ]]; then
        failure_reason="SP${SP_SIZE}/TP${TP_SIZE} worker exceeded the ${GENERATION_TIMEOUT_SECONDS}s generation timeout"
      fi
    fi
    [[ -n "${failure_reason}" ]] || sleep 2
  done

  echo "${failure_reason}; restarting the complete distributed group." >&2
  cleanup
  trap cleanup TERM INT EXIT
  reset_worker_after_failure "${failure_reason}"
  restart_count=$((restart_count + 1))
  delay=$((restart_count * 5))
  (( delay > 60 )) && delay=60
  sleep "${delay}"
done
