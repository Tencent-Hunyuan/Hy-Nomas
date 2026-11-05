#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

source scripts/inference/seg_ar_demo_config.sh

RUNTIME_DIR="${RUNTIME_DIR:-${PWD}/.runtime/seg_ar_demo}"
WORKER_REPLICAS="${WORKER_REPLICAS:-1}"
GPU_COUNT="${GPU_COUNT:-$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)}"
GPU_GROUPS="${WORKER_GPU_GROUPS:-}"
ENABLE_CUDA_MPS="${NOMAS_ENABLE_CUDA_MPS:-false}"
GENERATION_TIMEOUT_SECONDS="${NOMAS_GENERATION_TIMEOUT_SECONDS:-900}"
MPS_PIPE_DIR="${CUDA_MPS_PIPE_DIRECTORY:-${RUNTIME_DIR}/mps-pipe}"
MPS_LOG_DIR="${CUDA_MPS_LOG_DIRECTORY:-${RUNTIME_DIR}/mps-log}"
mkdir -p "${RUNTIME_DIR}"

mps_responds() {
  [[ -S "${MPS_PIPE_DIR}/control" ]] || return 1
  timeout 5s env \
    CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
    CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
    bash -c 'printf "get_server_list\n" | nvidia-cuda-mps-control' \
    >/dev/null 2>&1
}

stop_private_mps() {
  if mps_responds; then
    timeout 10s env \
      CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}" \
      CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}" \
      bash -c 'printf "quit\n" | nvidia-cuda-mps-control' \
      >/dev/null 2>&1 || true
  fi
  for _ in $(seq 1 40); do
    [[ ! -S "${MPS_PIPE_DIR}/control" ]] && break
    sleep 0.25
  done
  rm -rf "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
}

if (( WORKER_REPLICAS < 1 )); then
  echo "WORKER_REPLICAS must be >= 1" >&2
  exit 2
fi

declare -a groups=()
if [[ -n "${GPU_GROUPS}" ]]; then
  IFS=';' read -r -a groups <<< "${GPU_GROUPS}"
  if (( ${#groups[@]} != WORKER_REPLICAS )); then
    echo "WORKER_GPU_GROUPS must contain ${WORKER_REPLICAS} semicolon-separated groups" >&2
    exit 2
  fi
else
  if (( WORKER_REPLICAS * MODEL_PARALLEL_SIZE > GPU_COUNT )); then
    echo "Need $((WORKER_REPLICAS * MODEL_PARALLEL_SIZE)) distinct GPUs for ${WORKER_REPLICAS}x(SP${SP_SIZE}/TP${TP_SIZE}), only ${GPU_COUNT} visible." >&2
    echo "Set WORKER_GPU_GROUPS explicitly to opt into shared-GPU replicas, e.g. 0,1,...,7;0,1,...,7." >&2
    exit 2
  fi
  for (( slot=0; slot<WORKER_REPLICAS; slot++ )); do
    first_gpu=$((slot * MODEL_PARALLEL_SIZE))
    last_gpu=$((first_gpu + MODEL_PARALLEL_SIZE - 1))
    groups+=("$(seq -s, "${first_gpu}" "${last_gpu}")")
  done
fi

export NOMAS_WORKER_REPLICAS="${WORKER_REPLICAS}"
export NOMAS_MAX_CONCURRENT_JOBS="${NOMAS_MAX_CONCURRENT_JOBS:-1}"

started_mps=0
if [[ "${ENABLE_CUDA_MPS,,}" =~ ^(1|true|yes|on)$ ]]; then
  if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
    echo "NOMAS_ENABLE_CUDA_MPS=true but nvidia-cuda-mps-control is missing" >&2
    exit 2
  fi
  export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
  export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
  if ! mps_responds; then
    # A killed daemon can leave its Unix socket behind. Never mistake that
    # stale file for a usable MPS service.
    rm -rf "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
    mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
    nvidia-cuda-mps-control -d
    started_mps=1
    echo "Started private CUDA MPS daemon in ${MPS_PIPE_DIR}"
  else
    echo "Using existing private CUDA MPS daemon in ${MPS_PIPE_DIR}"
  fi
fi

declare -a pids=()
declare -a restart_counts=()
shutdown_pool() {
  trap - TERM INT EXIT
  for pid in "${pids[@]:-}"; do
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
    if [[ -n "${pgid}" ]]; then
      kill -TERM -- "-${pgid}" 2>/dev/null || true
    else
      kill -TERM "${pid}" 2>/dev/null || true
    fi
  done
  wait || true
  if (( started_mps )); then
    stop_private_mps
  fi
}
trap shutdown_pool TERM INT EXIT

start_replica() {
  local slot="$1"
  local gpu_list="${groups[$slot]}"
  local slot_runtime="${RUNTIME_DIR}/slot_${slot}"
  mkdir -p "${slot_runtime}"
  setsid env \
    RUNTIME_DIR="${slot_runtime}" \
    JOB_DB="${JOB_DB:-/tmp/nomas_seg_ar_jobs.json}" \
    WORKER_SLOT="${slot}" \
    SP_SIZE="${SP_SIZE}" \
    TP_SIZE="${TP_SIZE}" \
    CUDA_VISIBLE_DEVICES="${gpu_list}" \
    NOMAS_WORKER_ID="$(hostname):seg-ar-slot-${slot}" \
    bash scripts/inference/run_seg_ar_gpu_worker.sh "${DEMO_INFERENCE_ARGS[@]}" \
    >> "${RUNTIME_DIR}/worker_${slot}.log" 2>&1 &
  pids[$slot]=$!
  echo "Started worker replica ${slot} pid ${pids[$slot]} on CUDA_VISIBLE_DEVICES=${gpu_list}"
}

: > "${RUNTIME_DIR}/worker_replicas.pids"
for (( slot=0; slot<WORKER_REPLICAS; slot++ )); do
  : > "${RUNTIME_DIR}/worker_${slot}.log"
  restart_counts[$slot]=0
  start_replica "${slot}"
done

# A failed replica is restarted without taking the Gateway or healthy replicas
# down. Its fixed worker id lets JobStore fail only the lost interactive state
# and safely requeue standalone jobs.
while true; do
  timed_out_slots="$(
    python - "${JOB_DB:-/tmp/nomas_seg_ar_jobs.json}" \
      "${GENERATION_TIMEOUT_SECONDS}" <<'PY' || true
import sys
from apps.seg_ar_job_store import JobStore

timeout_seconds = max(0.0, float(sys.argv[2]))
if timeout_seconds > 0:
    result = JobStore(sys.argv[1]).timeout_running_jobs(timeout_seconds)
    for worker_id in result["worker_ids"]:
        marker = ":seg-ar-slot-"
        if marker in worker_id:
            print(worker_id.rsplit(marker, 1)[1])
PY
  )"
  for slot in ${timed_out_slots}; do
    pid="${pids[$slot]:-}"
    [[ -n "${pid}" ]] || continue
    if kill -0 "${pid}" 2>/dev/null; then
      pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
      echo "Generation watchdog timed out worker replica ${slot}; restarting its process group" >&2
      if [[ -n "${pgid}" ]]; then
        kill -TERM -- "-${pgid}" 2>/dev/null || true
      else
        kill -TERM "${pid}" 2>/dev/null || true
      fi
      for _ in $(seq 1 30); do
        kill -0 "${pid}" 2>/dev/null || break
        sleep 1
      done
      if kill -0 "${pid}" 2>/dev/null; then
        echo "Timed-out worker replica ${slot} ignored SIGTERM; sending SIGKILL" >&2
        if [[ -n "${pgid}" ]]; then
          kill -KILL -- "-${pgid}" 2>/dev/null || true
        else
          kill -KILL "${pid}" 2>/dev/null || true
        fi
      fi
    fi
  done
  : > "${RUNTIME_DIR}/worker_replicas.pids"
  for index in "${!pids[@]}"; do
    pid="${pids[$index]}"
    echo "${pid}" >> "${RUNTIME_DIR}/worker_replicas.pids"
    if ! kill -0 "${pid}" 2>/dev/null; then
      set +e
      wait "${pid}"
      rc=$?
      set -e
      restart_counts[$index]=$((restart_counts[$index] + 1))
      python - "${JOB_DB:-/tmp/nomas_seg_ar_jobs.json}" "${index}" "${rc}" <<'PY' || true
import socket
import sys
from apps.seg_ar_job_store import JobStore

store = JobStore(sys.argv[1])
slot = sys.argv[2]
return_code = sys.argv[3]
worker_id = f"{socket.gethostname()}:seg-ar-slot-{slot}"
store.reset_worker(worker_id)
store.record_error(
    code="WORKER_CRASH",
    message=f"GPU worker slot {slot} exited with code {return_code}; restarting",
    worker_id=worker_id,
    recoverable=True,
)
PY
      delay=$((restart_counts[$index] * 5))
      (( delay > 60 )) && delay=60
      echo "Worker replica ${index} exited with code ${rc}; restarting in ${delay}s" >&2
      sleep "${delay}"
      start_replica "${index}"
    fi
  done
  sleep 2
done
