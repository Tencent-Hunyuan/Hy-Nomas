#!/usr/bin/env bash
# Run SP8/TP8 traces or the B300 SP4/TP4 benchmarks with memory profiling disabled.
# Required model locations: replace the placeholders with your local paths.
# DIT_WEIGHT=/path/to/dit/model.safetensors \
# PIXELUMM_CHECKPOINT=/path/to/pixelumm/model.safetensors \
# TEXT_ENCODER_PATH=/path/to/Qwen2.5-VL-7B-Instruct \
# bash perf_bench/run_sp8_tp8.sh both
# For Hugging Face loading, use TEXT_ENCODER_PATH=Qwen/Qwen2.5-VL-7B-Instruct
# in the same command instead of the local text encoder directory.
set -euo pipefail
unset BASH_ENV ENV

usage() {
    cat <<'HELP'
Usage: bash perf_bench/run_sp8_tp8.sh [both|sp8|tp8|b300_sp4|b300_tp4] [LOCAL_ROOT]

Defaults to running SP8 (sp_size=8, tp_size=1), then TP8 (sp_size=1,
tp_size=8), sequentially on one eight-GPU machine. Stops on failure.
Use b300_sp4 (SP=4, TP=1) or b300_tp4 (SP=1, TP=4) on four B300 GPUs.
Both use scripts/inference/inference_b300_sp4.py for one full warmup
followed by one measured request. SP4 and TP4 keep DiT/VAE torch_compile enabled;
TP4 retains explicit all-gather/reduce-scatter between compiled local regions.

Required environment variables (no defaults):
  DIT_WEIGHT            DiT checkpoint (local file).
  PIXELUMM_CHECKPOINT    PixelUMM checkpoint (local file).
  TEXT_ENCODER_PATH     Text encoder local directory or Hugging Face model ID.
                        Example ID: Qwen/Qwen2.5-VL-7B-Instruct.
                        Use ./ or an absolute path for explicit local paths.

Optional environment overrides:
  PYTHON_BIN            Python executable; defaults to this worktree's .venv,
                        then ../Hy-Nomas/.venv, then python3 on PATH.
  PROMPT_FILE           Defaults to REPO_ROOT/perf_bench/test_prompt.json.
                        B300 modes accept pipeline or annotation JSON.
  LOCAL_ROOT            New output directory at any writable location.
                        Defaults to a unique directory under /tmp.
                        The second positional argument takes precedence.
  CUDA_VISIBLE_DEVICES  Eight GPUs for sp8/tp8; four distinct numeric indices for
                        b300_sp4/b300_tp4 (default: 0,1,2,3).
  HY_PROFILE_SEGMENTS    Segment indices to trace; default: 0,6.
  HY_PROFILE_RANKS       Ranks to trace; default: 0.
  SP8_MASTER_PORT        Default: 29618.
  TP8_MASTER_PORT        Default: 29628.
  B300_PROFILE          1 (default) traces the measured run; 0 disables traces.
  B300_SEGMENTS         Optional positive limit on the number of input segments.
  B300_ENTRY_INDEX      Zero-based sample index; required for multi-entry JSON lists.
  CUDA_HOME             B300 default: /usr/local/cuda; point to your toolkit.
  HF_HUB_OFFLINE        B300 default: 0 (allows downloading a model ID).
  OMP_NUM_THREADS       B300 default: 8.

B300 modes use NCCL_SOCKET_IFNAME=lo, NCCL_IB_DISABLE=1, NCCL_NET=Socket and
NCCL_NET_PLUGIN=none by default; explicit environment settings take precedence.
They retain the Python entry point's Magi/CuTe/DeepGEMM and VAE compile settings.

Examples (replace the placeholder paths):
  export DIT_WEIGHT=/path/to/dit/model.safetensors
  export PIXELUMM_CHECKPOINT=/path/to/pixelumm/model.safetensors
  export TEXT_ENCODER_PATH=/path/to/Qwen2.5-VL-7B-Instruct
  bash perf_bench/run_sp8_tp8.sh
  bash perf_bench/run_sp8_tp8.sh tp8 ./perf_bench/run_tp8
  HY_PROFILE_SEGMENTS=6 bash perf_bench/run_sp8_tp8.sh both
  bash perf_bench/run_sp8_tp8.sh b300_sp4 ./perf_bench/run_b300_sp4
  bash perf_bench/run_sp8_tp8.sh b300_tp4 ./perf_bench/run_b300_tp4
  B300_PROFILE=0 bash perf_bench/run_sp8_tp8.sh b300_sp4 ./perf_bench/run_b300_timing
  TEXT_ENCODER_PATH=Qwen/Qwen2.5-VL-7B-Instruct bash perf_bench/run_sp8_tp8.sh both

Outputs: LOCAL_ROOT/{sp8,tp8}/{traces,metrics,video,cache}, run.log,
command.sh and exit_code.txt in each case directory. Existing run directories
are rejected. Logs, traces and JIT caches are stored under LOCAL_ROOT.
B300 modes write results.json, run{0,1}/stages.rank*.jsonl and run1/video.mp4
under LOCAL_ROOT/<case>; traces are under <case>/traces (measured run only).
The last traced segment includes the final VAE decode if that segment is selected.
Profiling adds overhead to reported timing; use B300_PROFILE=0 for timing runs.

Both inference.profile_memory=false and HY_PROFILE_MEMORY=0 are forced.
Trace shapes are recorded; stacks are not. SP8/TP8 segment 0 includes cold
compilation; B300 modes record only after warmup.
Synchronous trace export can inflate the following segment's wall time.
HELP
}

fail() { printf 'Error: %s\n' "$*" >&2; exit 2; }

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi
(( $# <= 2 )) || fail "Expected at most a case and a new output directory."
CASE="${1:-both}"
case "$CASE" in
    both) CASES=(sp8 tp8) ;;
    sp8|tp8|b300_sp4|b300_tp4) CASES=("$CASE") ;;
    *) fail "Unknown case '$CASE'; use both, sp8, tp8, b300_sp4 or b300_tp4." ;;
esac

for required_var in DIT_WEIGHT PIXELUMM_CHECKPOINT TEXT_ENCODER_PATH; do
    [[ -n "${!required_var:-}" ]] || fail "$required_var is required; set it as an environment variable (see --help)."
done

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
    elif [[ -x "$REPO_ROOT/../Hy-Nomas/.venv/bin/python" ]]; then
        PYTHON_BIN="$REPO_ROOT/../Hy-Nomas/.venv/bin/python"
    else
        PYTHON_BIN="$(command -v python3)" || fail "Set PYTHON_BIN to the inference Python."
    fi
fi
PYTHON_BIN="$(command -v -- "$PYTHON_BIN")" || fail "Python executable not found."
[[ -x "$PYTHON_BIN" ]] || fail "Python is not executable: $PYTHON_BIN"
# Preserve the venv executable path: resolving its symlink selects system Python.
PYTHON_BIN="$(cd -- "$(dirname -- "$PYTHON_BIN")" && pwd)/$(basename -- "$PYTHON_BIN")"

PROMPT_FILE="${PROMPT_FILE:-$REPO_ROOT/perf_bench/test_prompt.json}"
for input_file in "$PROMPT_FILE" "$DIT_WEIGHT" "$PIXELUMM_CHECKPOINT"; do
    [[ -f "$input_file" && -r "$input_file" ]] || fail "Input file is not readable: $input_file"
done
# Resolve local directories before changing cwd; pass model IDs unchanged.
if [[ -d "$TEXT_ENCODER_PATH" ]]; then
    [[ -r "$TEXT_ENCODER_PATH" && -x "$TEXT_ENCODER_PATH" ]] || fail "Text encoder directory is not readable: $TEXT_ENCODER_PATH"
    TEXT_ENCODER_PATH="$(realpath -- "$TEXT_ENCODER_PATH")"
elif [[ -e "$TEXT_ENCODER_PATH" || -L "$TEXT_ENCODER_PATH" || "$TEXT_ENCODER_PATH" == /* || "$TEXT_ENCODER_PATH" == ./* || "$TEXT_ENCODER_PATH" == ../* ]]; then
    fail "Text encoder local path is not a readable directory: $TEXT_ENCODER_PATH"
fi
# Transformers validates and resolves any remaining Hugging Face model ID.
[[ -f "$REPO_ROOT/hymm/utils/torch_profile_windows.py" ]] || fail "This worktree lacks torch_profile_windows.py."
# Resolve user-supplied relative paths before entering the repository.
PROMPT_FILE="$(realpath -- "$PROMPT_FILE")"
DIT_WEIGHT="$(realpath -- "$DIT_WEIGHT")"
PIXELUMM_CHECKPOINT="$(realpath -- "$PIXELUMM_CHECKPOINT")"

EXPECTED_GPUS=8
DEFAULT_GPUS=0,1,2,3,4,5,6,7
if [[ "$CASE" == b300_sp4 || "$CASE" == b300_tp4 ]]; then
    EXPECTED_GPUS=4
    DEFAULT_GPUS=0,1,2,3
fi
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEFAULT_GPUS}"
IFS=',' read -r -a GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#GPU_IDS[@]} == "$EXPECTED_GPUS" && "$CUDA_VISIBLE_DEVICES" != *, ]] || fail "CUDA_VISIBLE_DEVICES must list exactly $EXPECTED_GPUS GPUs for $CASE."
for gpu_id in "${GPU_IDS[@]}"; do
    [[ -n "$gpu_id" && "$gpu_id" != *[[:space:]]* ]] || fail "Invalid GPU list: $CUDA_VISIBLE_DEVICES"
done
SP8_MASTER_PORT="${SP8_MASTER_PORT:-29618}"
TP8_MASTER_PORT="${TP8_MASTER_PORT:-29628}"
for port in "$SP8_MASTER_PORT" "$TP8_MASTER_PORT"; do
    [[ "$port" =~ ^[1-9][0-9]{0,4}$ ]] && (( port <= 65535 )) || fail "Invalid master port: $port"
done

B300_ARGS=()
if [[ "$CASE" == b300_sp4 || "$CASE" == b300_tp4 ]]; then
    B300_PROFILE="${B300_PROFILE:-1}"
    [[ "$B300_PROFILE" == 0 || "$B300_PROFILE" == 1 ]] || fail "B300_PROFILE must be 0 or 1."
    if [[ -n "${B300_SEGMENTS:-}" ]]; then
        [[ "$B300_SEGMENTS" =~ ^[1-9][0-9]*$ ]] || fail "B300_SEGMENTS must be a positive integer."
        B300_ARGS+=(--segments "$B300_SEGMENTS")
    fi
    if [[ -n "${B300_ENTRY_INDEX:-}" ]]; then
        [[ "$B300_ENTRY_INDEX" =~ ^[0-9]+$ ]] || fail "B300_ENTRY_INDEX must be a nonnegative integer."
        B300_ARGS+=(--entry-index "$B300_ENTRY_INDEX")
    fi
    # Preserve camera sidecar paths when the prompt is copied into LOCAL_ROOT.
    B300_ARGS+=(--input-base-dir "$(dirname -- "$PROMPT_FILE")")
    # Validate/convert JSON before allocating a run directory or GPU workers.
    # Annotation conversion imports the regular parser, but loads no models.
    "$PYTHON_BIN" "$REPO_ROOT/scripts/inference/inference_b300_sp4.py" \
        "$PROMPT_FILE" --gpus "$CUDA_VISIBLE_DEVICES" \
        "${B300_ARGS[@]}" --validate-only
fi

LOCAL_ROOT="${2:-${LOCAL_ROOT:-}}"
if [[ -n "$LOCAL_ROOT" ]]; then
    LOCAL_ROOT="$(realpath -m -- "$LOCAL_ROOT")"
    mkdir -- "$LOCAL_ROOT" || fail "Cannot create LOCAL_ROOT (it must not already exist): $LOCAL_ROOT"
else
    LOCAL_ROOT="$(mktemp -d "/tmp/hynomas_$(date +%Y%m%d_%H%M%S)_sp8_tp8.XXXXXX")"
fi
cp -- "$PROMPT_FILE" "$LOCAL_ROOT/test_prompt.json"
mkdir -- "$LOCAL_ROOT/bin"

# The venv may not provide a torchrun entrypoint. Keep both YAML parsing and
# distributed launch on the selected interpreter without modifying the venv.
cat > "$LOCAL_ROOT/bin/torchrun" <<'SH'
#!/usr/bin/env bash
exec "$HY_BENCH_PYTHON" -m torch.distributed.run "$@"
SH
cat > "$LOCAL_ROOT/bin/python3" <<'SH'
#!/usr/bin/env bash
exec "$HY_BENCH_PYTHON" "$@"
SH
cp -- "$LOCAL_ROOT/bin/python3" "$LOCAL_ROOT/bin/python"
chmod +x "$LOCAL_ROOT/bin/torchrun" "$LOCAL_ROOT/bin/python3" "$LOCAL_ROOT/bin/python"
cd -- "$REPO_ROOT"
{
    printf 'created_at=%s\nrepository=%s\npython=%s\nprompt_source=%s\n' \
        "$(date -Is)" "$REPO_ROOT" "$PYTHON_BIN" "$PROMPT_FILE"
    printf 'git_head=%s\n' "$(git rev-parse HEAD 2>/dev/null || printf unknown)"
    sha256sum "$LOCAL_ROOT/test_prompt.json"
} > "$LOCAL_ROOT/run_info.txt"
printf 'Run directory: %s\n' "$LOCAL_ROOT"

run_case() {
    local name="$1" sp tp port case_dir exit_code
    local -a command pipeline_status
    case "$name" in
        sp8) sp=8; tp=1; port="$SP8_MASTER_PORT" ;;
        tp8) sp=1; tp=8; port="$TP8_MASTER_PORT" ;;
        b300_sp4) sp=4; tp=1; port="" ;;
        b300_tp4) sp=1; tp=4; port="" ;;
    esac
    case_dir="$LOCAL_ROOT/$name"
    mkdir -p -- "$case_dir"/{metrics,traces,video,cache}
    command=(
        env -u BASH_ENV -u ENV
        "PATH=$LOCAL_ROOT/bin:$(dirname -- "$PYTHON_BIN"):$PATH"
        "HY_BENCH_PYTHON=$PYTHON_BIN"
        "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
        OMP_NUM_THREADS=1 GLOBAL_NUM_NODES=1 GLOBAL_NODE_RANK=0
        "DG_JIT_CACHE_DIR=$case_dir/cache/deep_gemm"
        "MAGI_ATTENTION_WORKSPACE_BASE=$case_dir/cache/magi"
        "TORCH_EXTENSIONS_DIR=$case_dir/cache/extensions"
        "HF_HOME=$case_dir/cache/huggingface"
        "CUDA_CACHE_PATH=$case_dir/cache/cuda"
        "XDG_CACHE_HOME=$case_dir/cache"
        "TORCHINDUCTOR_CACHE_DIR=$case_dir/cache/inductor"
        "TRITON_CACHE_DIR=$case_dir/cache/triton"
        "PYTORCH_KERNEL_CACHE_PATH=$case_dir/cache/kernels"
        "INFERENCE_METRICS_DIR=$case_dir/metrics"
        "HY_PROFILE_DIR=$case_dir/traces"
        "HY_PROFILE_SEGMENTS=${HY_PROFILE_SEGMENTS:-0,6}"
        "HY_PROFILE_RANKS=${HY_PROFILE_RANKS:-0}"
        HY_PROFILE_SHAPES=1 HY_PROFILE_STACK=0 HY_PROFILE_MEMORY=0
        HY_PROFILE_TABLE_ROWS=120
    )
    if [[ "$name" == b300_sp4 || "$name" == b300_tp4 ]]; then
        command+=(
            "OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}"
            "CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}"
            "HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}"
            "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}"
            "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}"
            "NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}"
            "NCCL_NET=${NCCL_NET:-Socket}"
            "NCCL_DEBUG=${NCCL_DEBUG:-WARN}"
        )
        if [[ "$B300_PROFILE" == 0 ]]; then command+=(HY_PROFILE_DIR=); fi
        # This Python entry point launches its own torchrun process.
        command+=(
            "$PYTHON_BIN" scripts/inference/inference_b300_sp4.py
            "$LOCAL_ROOT/test_prompt.json"
            --gpus "$CUDA_VISIBLE_DEVICES" --output "$case_dir"
            "${B300_ARGS[@]}"
            "dit_weight=$DIT_WEIGHT"
            "model.vae.checkpoint=$PIXELUMM_CHECKPOINT"
            "model.text_encoder.model_name=$TEXT_ENCODER_PATH"
            "parallelism.sp_size=$sp" "parallelism.tp_size=$tp"
            inference.profile_memory=false
            inference.seg_ar_segment_vae_decode=false
            model.vae.stable_prefix_streaming=false
        )
    else
        command+=(
            bash scripts/inference/inference_from_config.sh inference
            --nnodes 1 --node-rank 0 --master-addr 127.0.0.1 --master-port "$port"
            --sp-size "$sp" --tp-size "$tp"
            --dit-weight "$DIT_WEIGHT" --save-path "$case_dir/video"
            "model.vae.checkpoint=$PIXELUMM_CHECKPOINT"
            "model.text_encoder.model_name=$TEXT_ENCODER_PATH"
            "inference.prompt=$LOCAL_ROOT/test_prompt.json"
            inference.profile_memory=false
            model.vae.deepgemm_fp8=true
            inference.seg_ar_segment_vae_decode=true
            model.vae.stable_prefix_streaming=true
        )
    fi
    {
        printf '#!/usr/bin/env bash\nset -euo pipefail\nunset BASH_ENV ENV\n'
        printf 'cd -- %q\n' "$REPO_ROOT"
        printf '%q ' "${command[@]}"
        printf '\n'
    } > "$case_dir/command.sh"
    printf 'Starting %s (SP=%s, TP=%s); log: %s/run.log\n' "$name" "$sp" "$tp" "$case_dir"
    if "${command[@]}" 2>&1 | tee "$case_dir/run.log"; then
        exit_code=0
    else
        pipeline_status=("${PIPESTATUS[@]}")
        exit_code="${pipeline_status[0]}"
        if (( exit_code == 0 )); then exit_code="${pipeline_status[1]}"; fi
    fi
    printf '%s\n' "$exit_code" > "$case_dir/exit_code.txt"
    if (( exit_code != 0 )); then
        printf '%s failed (exit %s). See %s/run.log\n' "$name" "$exit_code" "$case_dir" >&2
        return "$exit_code"
    fi
    if [[ "$name" == b300_sp4 || "$name" == b300_tp4 ]]; then
        printf '%s complete; results: %s/results.json; video: %s/run1/video.mp4\n' "$name" "$case_dir" "$case_dir"
        if [[ "$B300_PROFILE" == 1 ]]; then printf 'Traces: %s/traces\n' "$case_dir"; fi
    else
        printf '%s complete; traces: %s/traces\n' "$name" "$case_dir"
    fi
}

for case_name in "${CASES[@]}"; do
    run_case "$case_name"
done
printf 'All requested cases completed: %s\n' "$LOCAL_ROOT"

