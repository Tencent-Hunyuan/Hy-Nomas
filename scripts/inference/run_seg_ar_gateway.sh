#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

source scripts/inference/seg_ar_demo_config.sh

JOB_DB="${JOB_DB:-/tmp/nomas_seg_ar_jobs.json}"
RUNTIME_DIR="${RUNTIME_DIR:-${PWD}/.runtime/seg_ar_demo}"
PORT="${PORT:-8081}"
HOST="${NOMAS_GATEWAY_HOST:-0.0.0.0}"

# Keep API credentials outside the repository while allowing Gateway-only
# restarts to retain online prompt enhancement.
PE_CREDENTIAL_FILE="${PE_CREDENTIAL_FILE:-${RUNTIME_DIR}/pe_gateway.env}"
if [[ -f "${PE_CREDENTIAL_FILE}" ]]; then
  chmod 600 "${PE_CREDENTIAL_FILE}" 2>/dev/null || true
  set -a
  # shellcheck disable=SC1090
  source "${PE_CREDENTIAL_FILE}"
  set +a
fi

export PE_API_URL="${PE_API_URL:-http://127.0.0.1:30000/v1/chat/completions}"
export PE_MODEL="${PE_MODEL:-Nomas-PE}"
# Optional K3 credentials for text/camera prompt enhancement. Image captions
# use Nomas-PE through PE_API_URL; credentials never reach the frontend.
K3_CREDENTIAL_FILE="${K3_CREDENTIAL_FILE:-${RUNTIME_DIR}/k3_gateway.env}"
if [[ -f "${K3_CREDENTIAL_FILE}" ]]; then
  set -a
  source "${K3_CREDENTIAL_FILE}"
  set +a
fi
if [[ "${NOMAS_SKIP_FRONTEND_BUILD:-false}" =~ ^(1|true|yes|on)$ ]]; then
  if [[ ! -f apps/frontend/dist/index.html ]]; then
    echo "Prebuilt frontend is missing: apps/frontend/dist/index.html" >&2
    exit 1
  fi
  echo "Using prebuilt frontend in apps/frontend/dist"
elif command -v npm >/dev/null 2>&1; then
  if [[ ! -x apps/frontend/node_modules/.bin/vite || ! -x apps/frontend/node_modules/.bin/tsc ]]; then
    if command -v prepare-nomas-frontend >/dev/null 2>&1; then
      prepare-nomas-frontend apps/frontend
    else
      npm --prefix apps/frontend ci --include=dev
    fi
  fi
  npm --prefix apps/frontend run build
elif [[ -f apps/frontend/dist/index.html ]]; then
  echo "npm is unavailable; using the prebuilt frontend in apps/frontend/dist"
else
  echo "npm is unavailable and apps/frontend/dist/index.html is missing" >&2
  exit 1
fi

GALLERY_SOURCE_DIR="${NOMAS_GALLERY_SOURCE_DIR:-}"
if [[ -n "${GALLERY_SOURCE_DIR}" ]]; then
  if [[ ! -d "${GALLERY_SOURCE_DIR}" ]]; then
    echo "Hy-Nomas gallery source does not exist: ${GALLERY_SOURCE_DIR}" >&2
    exit 1
  fi
fi

GALLERY_ARGS=()
if [[ -n "${GALLERY_SOURCE_DIR}" ]]; then
  GALLERY_ARGS=(--gallery-dir "${GALLERY_SOURCE_DIR}")
fi

exec python apps/seg_ar_gateway.py \
  --host "${HOST}" \
  --port "${PORT}" \
  --job-db "${JOB_DB}" \
  --output-dir "${SAVE_PATH}" \
  "${GALLERY_ARGS[@]}" \
  --frontend-dir apps/frontend/dist \
  "${DEMO_INFERENCE_ARGS[@]}"
