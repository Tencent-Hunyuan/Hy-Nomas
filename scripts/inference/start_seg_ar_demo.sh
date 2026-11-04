#!/bin/bash
# --single-gpu-low-mem-mode selects configs/experiment/single_gpu_low_mem_inference.yaml.
# The shared parser forwards the selected preset to the gateway and GPU workers.
set -euo pipefail

cd "$(dirname "$0")/../.."

source scripts/inference/seg_ar_demo_config.sh

RUNTIME_DIR="${RUNTIME_DIR:-${PWD}/.runtime/seg_ar_demo}"
JOB_DB="${JOB_DB:-/tmp/nomas_seg_ar_jobs.json}"
WORKER_REPLICAS="${WORKER_REPLICAS:-1}"
mkdir -p "${RUNTIME_DIR}"

is_enabled() {
  [[ "${1,,}" =~ ^(1|true|yes|on)$ ]]
}

# Multi-node workers are enabled only when their hosts are supplied explicitly.
# SP_SIZE and TP_SIZE come from the inference config or launch overrides.
if [[ -n "${NOMAS_WORKER_HOSTS:-}" ]] \
    && [[ -z "${NOMAS_WORKER_SCRIPT:-}" ]]; then
  export NOMAS_WORKER_SCRIPT="${PWD}/scripts/inference/run_seg_ar_multinode_worker.sh"
fi
if [[ -n "${NOMAS_WORKER_HOSTS:-}" ]]; then
  mapfile -t configured_worker_hosts < <(
    printf '%s\n' "${NOMAS_WORKER_HOSTS}" \
      | tr ',;' '\n' \
      | awk '{$1=$1}; NF && !seen[$0]++'
  )
  if (( ${#configured_worker_hosts[@]} < 1 )); then
    echo "NOMAS_WORKER_HOSTS did not contain a usable host" >&2
    exit 2
  fi
  export NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
  export MASTER_ADDR="${MASTER_ADDR:-${configured_worker_hosts[0]}}"
  export MASTER_PORT="${MASTER_PORT:-29500}"
  export RDZV_ID="${RDZV_ID:-nomas-seg-ar-sp${SP_SIZE}-${MASTER_PORT}}"
fi

WORKER_SCRIPT="${NOMAS_WORKER_SCRIPT:-${PWD}/scripts/inference/run_seg_ar_gpu_pool.sh}"

if (( WORKER_REPLICAS < 1 )); then
  echo "WORKER_REPLICAS must be >= 1" >&2
  exit 2
fi
if [[ ! -f "${WORKER_SCRIPT}" ]]; then
  echo "Worker launch script does not exist: ${WORKER_SCRIPT}" >&2
  exit 2
fi

# Inspect the selected topology without opening ports or starting GPU processes.
if is_enabled "${NOMAS_PRINT_TOPOLOGY_ONLY:-false}"; then
  printf 'DEMO_CONFIG_FILE=%s\n' "${DEMO_CONFIG_FILE}"
  printf 'WORKER_SCRIPT=%s\n' "${WORKER_SCRIPT}"
  printf 'NOMAS_WORKER_HOSTS=%s\n' "${NOMAS_WORKER_HOSTS:-}"
  printf 'MASTER_ADDR=%s\n' "${MASTER_ADDR:-}"
  printf 'MASTER_PORT=%s\n' "${MASTER_PORT:-}"
  printf 'NPROC_PER_NODE=%s\n' "${NPROC_PER_NODE:-}"
  printf 'SP_SIZE=%s\n' "${SP_SIZE:-}"
  printf 'TP_SIZE=%s\n' "${TP_SIZE:-}"
  printf 'MODEL_PARALLEL_SIZE=%s\n' "${MODEL_PARALLEL_SIZE:-}"
  printf 'RDZV_ID=%s\n' "${RDZV_ID:-}"
  printf 'PORT=%s\n' "${PORT:-}"
  printf 'NOMAS_DISPATCHER_ENABLED=%s\n' "${NOMAS_DISPATCHER_ENABLED:-}"
  printf 'NOMAS_DISPATCHER_PORT=%s\n' "${NOMAS_DISPATCHER_PORT:-}"
  printf 'NOMAS_GALLERY_SOURCE_DIR=%s\n' "${NOMAS_GALLERY_SOURCE_DIR:-}"
  exit 0
fi

if [[ -z "${DIT_WEIGHT:-}" || ! -s "${DIT_WEIGHT}" ]]; then
  echo "Set dit_weight in ${DEMO_CONFIG_FILE} or pass --dit-weight to an existing checkpoint" >&2
  exit 2
fi
if [[ -n "${DRY_RUN_SOURCE}" && ! -f "${DRY_RUN_SOURCE}" ]]; then
  echo "inference.prompt/DRY_RUN_SOURCE must be an existing JSON file when dry-run is enabled" >&2
  exit 2
fi

listener_pid_for_port() {
  local port="$1"
  local pid=""
  if command -v ss >/dev/null 2>&1; then
    pid="$(
      ss -lntp 2>/dev/null \
        | awk -v needle=":${port}" '
            $4 ~ needle "$" && match($0, /pid=[0-9]+/) {
              print substr($0, RSTART + 4, RLENGTH - 4)
              exit
            }'
    )"
    if [[ -n "${pid}" ]]; then
      printf '%s\n' "${pid}"
      return 0
    fi
  fi
  if command -v netstat >/dev/null 2>&1; then
    pid="$(
      netstat -lntp 2>/dev/null \
        | awk -v needle=":${port}" '
            $4 ~ needle "$" {
              split($7, proc, "/")
              if (proc[1] ~ /^[0-9]+$/) {
                print proc[1]
                exit
              }
            }'
    )"
    if [[ -n "${pid}" ]]; then
      printf '%s\n' "${pid}"
      return 0
    fi
  fi
  return 1
}

http_proxy_status_code() {
  curl -sS -m 2 -o /dev/null -w "%{http_code}" \
    "http://127.0.0.1:${NOMAS_HTTP_PROXY_PORT}/" 2>/dev/null || true
}

http_proxy_health_status_code() {
  curl -sS -m 2 -u "${NOMAS_DEMO_USERNAME}:${NOMAS_DEMO_PASSWORD}" \
    -o /dev/null -w "%{http_code}" \
    "http://127.0.0.1:${NOMAS_HTTP_PROXY_PORT}/health" 2>/dev/null || true
}

export NOMAS_WORKER_REPLICAS="${WORKER_REPLICAS}"
export NOMAS_MAX_CONCURRENT_JOBS="${NOMAS_MAX_CONCURRENT_JOBS:-1}"

AUTH_PROXY_ENABLED=0
if [[ -n "${NOMAS_AUTH_PROXY_PORT:-}" ]]; then
  if [[ -z "${NOMAS_DEMO_CREDENTIAL_SHA256:-}" ]]; then
    echo "NOMAS_AUTH_PROXY_PORT requires NOMAS_DEMO_CREDENTIAL_SHA256" >&2
    exit 2
  fi
  AUTH_PROXY_ENABLED=1
  export NOMAS_GATEWAY_HOST="${NOMAS_GATEWAY_HOST:-127.0.0.1}"
fi

HTTP_SERVER_PROXY_ENABLED=0
if [[ -n "${NOMAS_HTTP_PROXY_PORT:-}" ]]; then
  HTTP_SERVER_PROXY_ENABLED=1
  if [[ -z "${PORT:-}" && "${NOMAS_HTTP_PROXY_PORT}" == "8081" ]]; then
    export PORT=18084
  fi
  export NOMAS_GATEWAY_HOST="${NOMAS_GATEWAY_HOST:-127.0.0.1}"
  export NOMAS_HTTP_PROXY_BACKEND="${NOMAS_HTTP_PROXY_BACKEND:-http://127.0.0.1:${PORT:-8081}}"
  if [[ "${NOMAS_HTTP_PROXY_BACKEND}" == "http://127.0.0.1:${NOMAS_HTTP_PROXY_PORT}" ]] \
      || [[ "${NOMAS_HTTP_PROXY_BACKEND}" == "http://localhost:${NOMAS_HTTP_PROXY_PORT}" ]]; then
    echo "NOMAS_HTTP_PROXY_BACKEND must not point back to NOMAS_HTTP_PROXY_PORT=${NOMAS_HTTP_PROXY_PORT}" >&2
    exit 2
  fi
  credential_file="${NOMAS_HTTP_PROXY_CREDENTIAL_FILE:-${RUNTIME_DIR}/http_server_8081.env}"
  if [[ -z "${NOMAS_DEMO_PASSWORD:-}" ]]; then
    if [[ -f "${credential_file}" ]]; then
      chmod 600 "${credential_file}" 2>/dev/null || true
      set -a
      # This file is generated below with mode 0600 and shell-escaped values.
      # shellcheck disable=SC1090
      source "${credential_file}"
      set +a
    else
      export NOMAS_DEMO_USERNAME="${NOMAS_DEMO_USERNAME:-hy}"
      export NOMAS_DEMO_PASSWORD="$(
        python3 -c 'import secrets; print("NOMAS-" + secrets.token_urlsafe(12))'
      )"
      old_umask="$(umask)"
      umask 077
      {
        printf 'NOMAS_DEMO_USERNAME=%q\n' "${NOMAS_DEMO_USERNAME}"
        printf 'NOMAS_DEMO_PASSWORD=%q\n' "${NOMAS_DEMO_PASSWORD}"
      } > "${credential_file}"
      umask "${old_umask}"
      chmod 600 "${credential_file}"
    fi
  fi
  if [[ -z "${NOMAS_DEMO_PASSWORD:-}" ]]; then
    echo "NOMAS_HTTP_PROXY_PORT requires NOMAS_DEMO_PASSWORD" >&2
    exit 2
  fi
fi

DISPATCHER_ENABLED=0
if [[ "${NOMAS_DISPATCHER_ENABLED:-false}" =~ ^(1|true|yes|on)$ ]]; then
  DISPATCHER_ENABLED=1
  NOMAS_DISPATCHER_PORT="${NOMAS_DISPATCHER_PORT:-18085}"
  NOMAS_DISPATCHER_LEASE_SECONDS="${NOMAS_DISPATCHER_LEASE_SECONDS:-900}"
  NOMAS_DISPATCHER_STATE_DB="${NOMAS_DISPATCHER_STATE_DB:-${RUNTIME_DIR}/dispatcher_resources.json}"
  if [[ -z "${NOMAS_DISPATCHER_HOSTS_FILE:-}" ]]; then
    NOMAS_DISPATCHER_HOSTS_FILE="${RUNTIME_DIR}/dispatcher_hosts.json"
    cat > "${NOMAS_DISPATCHER_HOSTS_FILE}" <<EOF
{"hosts":[{"id":"$(hostname)-sp${SP_SIZE:-8}","url":"http://127.0.0.1:${PORT:-8081}"}]}
EOF
  elif [[ ! -f "${NOMAS_DISPATCHER_HOSTS_FILE}" ]]; then
    echo "Dispatcher hosts file does not exist: ${NOMAS_DISPATCHER_HOSTS_FILE}" >&2
    exit 2
  fi
  if (( HTTP_SERVER_PROXY_ENABLED )); then
    export NOMAS_HTTP_PROXY_BACKEND="http://127.0.0.1:${NOMAS_DISPATCHER_PORT}"
  fi
fi

existing_gateway_pid="$(listener_pid_for_port "${PORT:-8081}" || true)"
if [[ -n "${existing_gateway_pid}" ]]; then
  gateway_status="$(
    curl -sS -m 2 -o /dev/null -w "%{http_code}" \
      "http://127.0.0.1:${PORT:-8081}/health" 2>/dev/null || true
  )"
  if [[ "${gateway_status}" != "200" ]]; then
    echo "Port ${PORT:-8081} is already in use, but /health returned HTTP ${gateway_status}." >&2
    exit 1
  fi
  printf '%s\n' "${existing_gateway_pid}" > "${RUNTIME_DIR}/gateway.pid"
  echo "Gateway already running (pid ${existing_gateway_pid})"
else
  nohup setsid env RUNTIME_DIR="${RUNTIME_DIR}" JOB_DB="${JOB_DB}" \
    WORKER_REPLICAS="${WORKER_REPLICAS}" \
    NOMAS_WORKER_REPLICAS="${WORKER_REPLICAS}" \
    NOMAS_MAX_CONCURRENT_JOBS="${NOMAS_MAX_CONCURRENT_JOBS}" \
    bash scripts/inference/run_seg_ar_gateway.sh \
    "${DEMO_INFERENCE_ARGS[@]}" \
    > "${RUNTIME_DIR}/gateway.log" 2>&1 < /dev/null &
  gateway_launcher_pid="$!"
  printf '%s\n' "${gateway_launcher_pid}" > "${RUNTIME_DIR}/gateway.launcher.pid"
  printf '%s\n' "${gateway_launcher_pid}" > "${RUNTIME_DIR}/gateway.pid"
  gateway_ready=0
  for _ in $(seq 1 360); do
    gateway_pid="$(listener_pid_for_port "${PORT:-8081}" || true)"
    if [[ -n "${gateway_pid}" ]]; then
      printf '%s\n' "${gateway_pid}" > "${RUNTIME_DIR}/gateway.pid"
    fi
    gateway_status="$(
      curl -sS -m 2 -o /dev/null -w "%{http_code}" \
        "http://127.0.0.1:${PORT:-8081}/health" 2>/dev/null || true
    )"
    if [[ "${gateway_status}" == "200" ]]; then
      gateway_ready=1
      break
    fi
    sleep 0.25
  done
  if (( ! gateway_ready )); then
    echo "Gateway failed to become ready" >&2
    tail -100 "${RUNTIME_DIR}/gateway.log" >&2 || true
    kill "${gateway_launcher_pid}" 2>/dev/null || true
    gateway_pid="$(listener_pid_for_port "${PORT:-8081}" || true)"
    if [[ -n "${gateway_pid}" ]]; then
      kill "${gateway_pid}" 2>/dev/null || true
    fi
    exit 1
  fi
  rm -f "${RUNTIME_DIR}/gateway.launcher.pid"
  echo "Started gateway pid $(cat "${RUNTIME_DIR}/gateway.pid")"
fi

if (( DISPATCHER_ENABLED )); then
  existing_dispatcher_pid="$(
    listener_pid_for_port "${NOMAS_DISPATCHER_PORT}" || true
  )"
  if [[ -n "${existing_dispatcher_pid}" ]]; then
    dispatcher_status="$(
      curl -sS -m 2 "http://127.0.0.1:${NOMAS_DISPATCHER_PORT}/health" \
        2>/dev/null \
        | python3 -c 'import json,sys; print(json.load(sys.stdin).get("architecture",""))' \
        2>/dev/null || true
    )"
    if [[ "${dispatcher_status}" != "multi_host_dispatcher" ]]; then
      echo "Port ${NOMAS_DISPATCHER_PORT} is occupied by a non-dispatcher process." >&2
      exit 1
    fi
    printf '%s\n' "${existing_dispatcher_pid}" > "${RUNTIME_DIR}/dispatcher.pid"
    echo "Dispatcher already running (pid ${existing_dispatcher_pid})"
  else
    nohup setsid env \
      HOSTS_FILE="${NOMAS_DISPATCHER_HOSTS_FILE}" \
      RESOURCE_DB="${NOMAS_DISPATCHER_STATE_DB}" \
      PORT="${NOMAS_DISPATCHER_PORT}" \
      LEASE_SECONDS="${NOMAS_DISPATCHER_LEASE_SECONDS}" \
      bash scripts/inference/run_seg_ar_dispatcher.sh \
      > "${RUNTIME_DIR}/dispatcher.log" 2>&1 < /dev/null &
    dispatcher_launcher_pid="$!"
    printf '%s\n' "${dispatcher_launcher_pid}" > "${RUNTIME_DIR}/dispatcher.pid"
    dispatcher_ready=0
    for _ in $(seq 1 360); do
      dispatcher_pid="$(
        listener_pid_for_port "${NOMAS_DISPATCHER_PORT}" || true
      )"
      if [[ -n "${dispatcher_pid}" ]]; then
        printf '%s\n' "${dispatcher_pid}" > "${RUNTIME_DIR}/dispatcher.pid"
      fi
      dispatcher_status="$(
        curl -sS -m 2 "http://127.0.0.1:${NOMAS_DISPATCHER_PORT}/health" \
          2>/dev/null \
          | python3 -c 'import json,sys; print(json.load(sys.stdin).get("architecture",""))' \
          2>/dev/null || true
      )"
      if [[ "${dispatcher_status}" == "multi_host_dispatcher" ]]; then
        dispatcher_ready=1
        break
      fi
      sleep 0.25
    done
    if (( ! dispatcher_ready )); then
      echo "Dispatcher failed to become ready" >&2
      tail -100 "${RUNTIME_DIR}/dispatcher.log" >&2 || true
      kill "${dispatcher_launcher_pid}" 2>/dev/null || true
      exit 1
    fi
    echo "Started dispatcher pid $(cat "${RUNTIME_DIR}/dispatcher.pid")"
  fi
fi

if (( AUTH_PROXY_ENABLED )); then
  if [[ -f "${RUNTIME_DIR}/auth_proxy.pid" ]] \
      && kill -0 "$(cat "${RUNTIME_DIR}/auth_proxy.pid")" 2>/dev/null; then
    echo "Auth proxy already running (pid $(cat "${RUNTIME_DIR}/auth_proxy.pid"))"
  else
    nohup setsid env \
      NOMAS_PROXY_BACKEND="http://127.0.0.1:${PORT:-8081}" \
      NOMAS_PROXY_HOST="${NOMAS_PROXY_HOST:-0.0.0.0}" \
      NOMAS_PROXY_PORT="${NOMAS_AUTH_PROXY_PORT}" \
      NOMAS_DEMO_USERNAME="${NOMAS_DEMO_USERNAME:-demo}" \
      NOMAS_DEMO_CREDENTIAL_SHA256="${NOMAS_DEMO_CREDENTIAL_SHA256}" \
      python3 scripts/inference/run_nomas_auth_proxy.py \
      > "${RUNTIME_DIR}/auth_proxy.log" 2>&1 < /dev/null &
    echo $! > "${RUNTIME_DIR}/auth_proxy.pid"
    echo "Started auth proxy pid $(cat "${RUNTIME_DIR}/auth_proxy.pid")"
  fi
fi

if (( HTTP_SERVER_PROXY_ENABLED )); then
  existing_proxy_pid="$(listener_pid_for_port "${NOMAS_HTTP_PROXY_PORT}" || true)"
  if [[ -n "${existing_proxy_pid}" ]]; then
    status="$(http_proxy_status_code)"
    health_status="$(http_proxy_health_status_code)"
    if [[ "${status}" != "401" || "${health_status}" != "200" ]]; then
      echo "Port ${NOMAS_HTTP_PROXY_PORT} is already in use, but it did not match the expected NOMAS http-server proxy (unauth HTTP ${status}, auth /health HTTP ${health_status})." >&2
      exit 1
    fi
    printf '%s\n' "${existing_proxy_pid}" > "${RUNTIME_DIR}/http_server_8081.pid"
    echo "HTTP server proxy already running (pid ${existing_proxy_pid})"
  else
    nohup setsid env \
      NOMAS_HTTP_PROXY_HOST="${NOMAS_HTTP_PROXY_HOST:-0.0.0.0}" \
      NOMAS_HTTP_PROXY_PORT="${NOMAS_HTTP_PROXY_PORT}" \
      NOMAS_HTTP_PROXY_BACKEND="${NOMAS_HTTP_PROXY_BACKEND}" \
      NOMAS_DEMO_USERNAME="${NOMAS_DEMO_USERNAME:-hy}" \
      NOMAS_DEMO_PASSWORD="${NOMAS_DEMO_PASSWORD}" \
      NODE_BIN="${NODE_BIN:-}" \
      bash scripts/inference/run_nomas_http_proxy.sh \
      > "${RUNTIME_DIR}/http_server_8081.log" 2>&1 < /dev/null &
    launcher_pid="$!"
    printf '%s\n' "${launcher_pid}" > "${RUNTIME_DIR}/http_server_8081.launcher.pid"
    printf '%s\n' "${launcher_pid}" > "${RUNTIME_DIR}/http_server_8081.pid"
    proxy_ready=0
    for _ in $(seq 1 240); do
      proxy_pid="$(listener_pid_for_port "${NOMAS_HTTP_PROXY_PORT}" || true)"
      if [[ -n "${proxy_pid}" ]]; then
        printf '%s\n' "${proxy_pid}" > "${RUNTIME_DIR}/http_server_8081.pid"
      fi
      status="$(http_proxy_status_code)"
      health_status="$(http_proxy_health_status_code)"
      if [[ "${status}" == "401" && "${health_status}" == "200" ]]; then
        proxy_ready=1
        break
      fi
      sleep 0.25
    done
    if (( ! proxy_ready )); then
      echo "HTTP server proxy failed to become ready" >&2
      tail -50 "${RUNTIME_DIR}/http_server_8081.log" >&2 || true
      kill "${launcher_pid}" 2>/dev/null || true
      proxy_pid="$(listener_pid_for_port "${NOMAS_HTTP_PROXY_PORT}" || true)"
      if [[ -n "${proxy_pid}" ]]; then
        kill "${proxy_pid}" 2>/dev/null || true
      fi
      exit 1
    fi
    rm -f "${RUNTIME_DIR}/http_server_8081.launcher.pid"
    echo "Started HTTP server proxy pid $(
      cat "${RUNTIME_DIR}/http_server_8081.pid"
    )"
    echo "HTTP server credentials: ${credential_file}"
  fi
fi

if [[ -f "${RUNTIME_DIR}/worker.pid" ]] \
    && kill -0 "$(cat "${RUNTIME_DIR}/worker.pid")" 2>/dev/null; then
  echo "GPU worker pool already running (pid $(cat "${RUNTIME_DIR}/worker.pid"))"
else
  python - "${JOB_DB}" <<'PY'
import sys
from apps.seg_ar_job_store import JobStore

JobStore(sys.argv[1]).clear_worker()
PY
  nohup setsid env RUNTIME_DIR="${RUNTIME_DIR}" JOB_DB="${JOB_DB}" \
    WORKER_REPLICAS="${WORKER_REPLICAS}" \
    bash "${WORKER_SCRIPT}" "${DEMO_INFERENCE_ARGS[@]}" \
    > "${RUNTIME_DIR}/worker.log" 2>&1 < /dev/null &
  echo $! > "${RUNTIME_DIR}/worker.pid"
  echo "Started GPU worker pool pid $(cat "${RUNTIME_DIR}/worker.pid")"
fi

echo "Gateway log: ${RUNTIME_DIR}/gateway.log"
echo "Worker pool log: ${RUNTIME_DIR}/worker.log"
echo "Replica logs:     ${RUNTIME_DIR}/worker_<slot>.log"
if [[ -f "${RUNTIME_DIR}/dispatcher.pid" ]]; then
  echo "Dispatcher log:  ${RUNTIME_DIR}/dispatcher.log"
fi
if [[ -f "${RUNTIME_DIR}/auth_proxy.pid" ]]; then
  echo "Auth proxy log:   ${RUNTIME_DIR}/auth_proxy.log"
fi
if [[ -f "${RUNTIME_DIR}/http_server_8081.pid" ]]; then
  echo "HTTP proxy log:   ${RUNTIME_DIR}/http_server_8081.log"
fi
