#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

source scripts/inference/seg_ar_demo_config.sh

RUNTIME_DIR="${RUNTIME_DIR:-${PWD}/.runtime/seg_ar_demo}"
JOB_DB="${JOB_DB:-/tmp/nomas_seg_ar_jobs.json}"
WORKER_SLOT="${WORKER_SLOT:-0}"
WORKER_LOCK="${WORKER_LOCK:-/tmp/nomas_seg_ar_gpu_worker_${WORKER_SLOT}.lock}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${MODEL_PARALLEL_SIZE}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
RDZV_ID="${RDZV_ID:-nomas-seg-ar-${MASTER_PORT}}"
COLLECTIVE_KEEPALIVE_INTERVAL="${COLLECTIVE_KEEPALIVE_INTERVAL:-300}"
MAX_CONCURRENT_JOBS="${MAX_CONCURRENT_JOBS:-${NOMAS_MAX_CONCURRENT_JOBS:-1}}"
MIN_FREE_GPU_MEMORY_GIB="${MIN_FREE_GPU_MEMORY_GIB:-${NOMAS_MIN_FREE_GPU_MEMORY_GIB:-4}}"
ADMISSION_SETTLE_SECONDS="${ADMISSION_SETTLE_SECONDS:-${NOMAS_ADMISSION_SETTLE_SECONDS:-2}}"
SESSION_IDLE_TIMEOUT="${SESSION_IDLE_TIMEOUT:-${NOMAS_SESSION_IDLE_TIMEOUT:-900}}"
LONG_SESSION_THRESHOLD_SECONDS="${LONG_SESSION_THRESHOLD_SECONDS:-${NOMAS_LONG_SESSION_THRESHOLD_SECONDS:-0}}"
MAX_LONG_SESSIONS="${MAX_LONG_SESSIONS:-${NOMAS_MAX_LONG_SESSIONS:-0}}"
EXCLUSIVE_LONG_SESSION="${EXCLUSIVE_LONG_SESSION:-${NOMAS_EXCLUSIVE_LONG_SESSION:-false}}"
export NOMAS_WORKER_SLOT="${NOMAS_WORKER_SLOT:-${WORKER_SLOT}}"
export NOMAS_WORKER_ID="${NOMAS_WORKER_ID:-$(hostname):seg-ar-slot-${WORKER_SLOT}}"
if [[ "${EXCLUSIVE_LONG_SESSION,,}" =~ ^(1|true|yes|on)$ ]]; then
  EXCLUSIVE_LONG_SESSION_FLAG="--worker-exclusive-long-session"
else
  EXCLUSIVE_LONG_SESSION_FLAG="--no-worker-exclusive-long-session"
fi

# Let NCCL, UCX, and NVSHMEM detect the network unless the caller sets them.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_INDUCTOR_FX_GRAPH_CACHE="${TORCH_INDUCTOR_FX_GRAPH_CACHE:-1}"

if [[ ! -s "${DIT_WEIGHT}" ]]; then
  echo "Missing distilled checkpoint: ${DIT_WEIGHT}" >&2
  exit 1
fi
if [[ -n "${DRY_RUN_SOURCE}" && ! -f "${DRY_RUN_SOURCE}" ]]; then
  echo "inference.prompt/DRY_RUN_SOURCE must be an existing JSON file when dry-run is enabled" >&2
  exit 1
fi
if (( NNODES < 1 || NODE_RANK < 0 || NODE_RANK >= NNODES )); then
  echo "Invalid distributed topology: NNODES=${NNODES}, NODE_RANK=${NODE_RANK}" >&2
  exit 2
fi
if (( NPROC_PER_NODE < 1 || NNODES * NPROC_PER_NODE != MODEL_PARALLEL_SIZE )); then
  echo "SP_SIZE * TP_SIZE must equal NNODES * NPROC_PER_NODE; got ${MODEL_PARALLEL_SIZE} != ${NNODES} * ${NPROC_PER_NODE}" >&2
  exit 2
fi

echo "Starting Hy-Nomas GPU queue worker with:"
echo "  location:   ${MACHINE_LOCATION:-default}"
echo "  config:     ${DEMO_CONFIG_FILE} + launch overrides"
echo "  checkpoint: ${DIT_WEIGHT}"
echo "  benchmark:  ${DRY_RUN_SOURCE}"
echo "  queue:      ${JOB_DB}"
echo "  output:     ${SAVE_PATH}"
echo "  worker id:  ${NOMAS_WORKER_ID}"
echo "  slot:       ${WORKER_SLOT}"
echo "  CUDA GPUs:  ${CUDA_VISIBLE_DEVICES:-all}"
echo "  topology:   ${NNODES} node(s) x ${NPROC_PER_NODE} GPU(s) = SP${SP_SIZE} x TP${TP_SIZE}"
if (( NNODES > 1 )); then
  echo "  rendezvous: ${MASTER_ADDR}:${MASTER_PORT} rank ${NODE_RANK}/${NNODES}"
fi
echo "  global concurrency: ${MAX_CONCURRENT_JOBS}"
echo "  min free GPU memory: ${MIN_FREE_GPU_MEMORY_GIB} GiB"
echo "  session idle timeout: ${SESSION_IDLE_TIMEOUT}s"

if (( NNODES > 1 )); then
  TORCHRUN_ARGS=(
    --nnodes="${NNODES}"
    --nproc_per_node="${NPROC_PER_NODE}"
    --node_rank="${NODE_RANK}"
    --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}"
    --rdzv_id="${RDZV_ID}"
  )
else
  TORCHRUN_ARGS=(
    --standalone
    --nproc_per_node="${NPROC_PER_NODE}"
  )
fi

# Online-only FP8/decode/noise/KV settings are fixed after launch overrides.
# Keep the shared inference preset on BF16 for offline runs.
exec flock -n "${WORKER_LOCK}" \
  python -m torch.distributed.run "${TORCHRUN_ARGS[@]}" \
  apps/seg_ar_web_server.py \
  --worker-only \
  --job-db "${JOB_DB}" \
  --worker-collective-keepalive-interval "${COLLECTIVE_KEEPALIVE_INTERVAL}" \
  --worker-max-concurrent-jobs "${MAX_CONCURRENT_JOBS}" \
  --worker-min-free-gpu-memory-gib "${MIN_FREE_GPU_MEMORY_GIB}" \
  --worker-admission-settle-seconds "${ADMISSION_SETTLE_SECONDS}" \
  --worker-session-idle-timeout "${SESSION_IDLE_TIMEOUT}" \
  --worker-long-session-threshold-seconds "${LONG_SESSION_THRESHOLD_SECONDS}" \
  --worker-max-long-sessions "${MAX_LONG_SESSIONS}" \
  "${EXCLUSIVE_LONG_SESSION_FLAG}" \
  "${DEMO_INFERENCE_ARGS[@]}" \
  ++model.vae.deepgemm_fp8=true \
  ++inference.seg_ar_segment_vae_decode=true \
  ++model.vae.stable_prefix_streaming=true \
  ++inference.interactive_noise_max_seconds=120 \
  ++inference.interactive_noise_hard_max_seconds=180 \
  ++inference.interactive_noise_growth_seconds=30 \
  ++inference.interactive_kv_cache_layerwise_cpu_offload=false \
  ++inference.interactive_kv_cache_cpu_mode=phase_swap \
  ++inference.interactive_kv_cache_resident_blocks=16 \
  ++inference.interactive_kv_cache_initial_gpu_tokens=65536 \
  ++inference.interactive_kv_cache_preallocate_seconds=0 \
  ++inference.interactive_kv_cache_min_segment_frames=80 \
  ++inference.interactive_kv_cache_growth_factor=1.5 \
  ++inference.interactive_kv_cache_staging_slots=4
