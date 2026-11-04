#!/bin/bash
# Public inference launcher. The inference preset uses segment-AR with SP=1, TP=8.
#
# Usage:
#   bash scripts/inference/inference_from_config.sh inference \
#     --dit-weight /path/to/distilled_model.safetensors \
#     --save-path ./outputs/distill -p "A moving camera tracks the subject"
#
set -e

# ================= Environment Setup =================
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-bond1}
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=1
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-bond1}
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=INIT,NET
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_INDUCTOR_FX_GRAPH_CACHE=1
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/cusparselt/lib:${LD_LIBRARY_PATH:-}

# ================= Parse positional args =================
if [ $# -lt 1 ]; then
    echo "Usage: bash $0 <experiment> [--dit-weight ...] [--save-path ...] [-p ...] [other overrides]"
    echo ""
    echo "Example:"
    echo "  bash $0 inference \\"
    echo "    --dit-weight /path/to/ckpt.safetensors --save-path ./outputs/test -p \"prompt1|||prompt2\""
    exit 1
fi

EXPERIMENT="$1"
shift 1

# ---- Normalize experiment path ----
# Users may pass full paths like "configs/experiment/inference.yaml"
# but Hydra expects just "inference" (relative to configs/experiment/).
# Strip the prefix and suffix automatically.
EXPERIMENT="${EXPERIMENT%.yaml}"
EXPERIMENT="${EXPERIMENT#configs/experiment/}"

# ================= Auto-detect GPUs =================
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a _VISIBLE_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
    NUM_GPUS=${#_VISIBLE_DEVICES[@]}
elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
else
    NUM_GPUS=1
fi
if [ "${NUM_GPUS:-0}" -le 0 ] 2>/dev/null; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' \
            | grep -c '[^[:space:]]' || echo 1)
    else
        NUM_GPUS=1
    fi
fi

# ================= Extract shell-level args =================
SP_SIZE=""
TP_SIZE=""
ROLLOUT_TYPE=""
_SP_SIZE_FROM_CLI=0
_TP_SIZE_FROM_CLI=0
_ROLLOUT_TYPE_FROM_CLI=0
# Preserve platform-provided topology before resolving the inference group.
_ENV_NODE_RANK="${NODE_RANK:-}"
_ENV_MASTER_ADDR="${MASTER_ADDR:-}"
_ENV_MASTER_PORT="${MASTER_PORT:-}"
_CLI_NNODES=""
_CLI_NODE_RANK=""
_CLI_MASTER_ADDR=""
_CLI_MASTER_PORT=""
INDEX_INCREMENT=${INDEX_INCREMENT:-0}

PASSTHROUGH_ARGS=()
DEMO_REPLAY_MANIFEST=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --demo-replay)
            DEMO_REPLAY_MANIFEST="$2"
            shift 2
            ;;
        --sp-size)
            SP_SIZE="$2"
            _SP_SIZE_FROM_CLI=1
            shift 2
            ;;
        --tp-size)
            TP_SIZE="$2"
            _TP_SIZE_FROM_CLI=1
            shift 2
            ;;
        --rollout-type)
            ROLLOUT_TYPE="$2"
            _ROLLOUT_TYPE_FROM_CLI=1
            shift 2
            ;;
        --output-folder|--save-path)
            PASSTHROUGH_ARGS+=("inference.save_path=$2")
            shift 2
            ;;
        --dit-weight)
            PASSTHROUGH_ARGS+=("dit_weight=$2")
            shift 2
            ;;
        --nnodes)
            _CLI_NNODES="$2"
            shift 2
            ;;
        --node-rank)
            _CLI_NODE_RANK="$2"
            shift 2
            ;;
        --master-addr)
            _CLI_MASTER_ADDR="$2"
            shift 2
            ;;
        --master-port)
            _CLI_MASTER_PORT="$2"
            shift 2
            ;;
        --)
            # Conventional option separator. Remaining tokens are still Hydra
            # overrides, but the literal "--" must not be forwarded.
            shift
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

# ================= Read parallel degrees and rollout from the experiment =================
_EXPERIMENT_FILE="configs/experiment/${EXPERIMENT}.yaml"
if [ -f "$_EXPERIMENT_FILE" ]; then
_yaml_values=$(python3 -c "
import yaml, sys, os, copy

def deep_merge(base, override):
    \"\"\"Recursively merge override into base (override wins).\"\"\"
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result

def resolve_yaml(filepath, configs_root='configs', visited=None):
    \"\"\"Load a YAML file and recursively resolve its Hydra 'defaults' list.\"\"\"
    if visited is None:
        visited = set()
    filepath = os.path.abspath(filepath)
    if filepath in visited:
        return {}
    visited.add(filepath)

    with open(filepath) as f:
        data = yaml.safe_load(f) or {}

    defaults = data.pop('defaults', [])
    merged = {}
    for entry in defaults:
        # Each entry is either a string or a single-key dict
        if isinstance(entry, str):
            ref = entry
        elif isinstance(entry, dict):
            ref = list(entry.values())[0]
        else:
            continue
        # Resolve Hydra path: '/experiment/foo/bar' -> 'configs/experiment/foo/bar.yaml'
        ref_path = ref.lstrip('/')
        parent_file = os.path.join(configs_root, ref_path + '.yaml')
        # Skip Hydra internal overrides (e.g. 'override hydra/...')
        if isinstance(entry, dict):
            key = list(entry.keys())[0]
            if str(key).startswith('override '):
                continue
        # Skip _self_
        if ref == '_self_':
            continue
        if not os.path.isfile(parent_file):
            raise FileNotFoundError(
                f\"Default config file not found!\\n\"
                f\"  Source file : {filepath}\\n\"
                f\"  Missing ref : '{ref}'\\n\"
                f\"  Expected at : {os.path.abspath(parent_file)}\\n\"
                f\"  Please check that the referenced config file exists.\"
            )
        parent_data = resolve_yaml(parent_file, configs_root, visited)
        merged = deep_merge(merged, parent_data)

    # Current file's values override inherited ones
    merged = deep_merge(merged, data)
    return merged

data = resolve_yaml('${_EXPERIMENT_FILE}')
sp = data.get('sp_size') or (data.get('parallelism') or {}).get('sp_size') or 1
tp = data.get('tp_size') or (data.get('parallelism') or {}).get('tp_size') or 1
rt = data.get('rollout_type') or (data.get('inference') or {}).get('rollout_type') or 'one_shot'
print(f'{sp} {tp} {rt}')
" 2>/dev/null || echo "1 1 one_shot")
else
    _yaml_values="1 1 one_shot"
fi

_yaml_sp=$(echo "$_yaml_values" | awk '{print $1}')
_yaml_tp=$(echo "$_yaml_values" | awk '{print $2}')
_yaml_rt=$(echo "$_yaml_values" | awk '{print $3}')

if [ -z "$SP_SIZE" ]; then
    SP_SIZE="$_yaml_sp"
fi
if [ -z "$TP_SIZE" ]; then
    TP_SIZE="$_yaml_tp"
fi
if [ -z "$ROLLOUT_TYPE" ]; then
    ROLLOUT_TYPE="$_yaml_rt"
fi

# Keep launcher-level and Hydra-level overrides consistent. This allows either
# ``--sp-size 16`` or ``parallelism.sp_size=16`` to control both sides.
_HYDRA_SP_SIZE=""
_HYDRA_TP_SIZE=""
_HYDRA_ROLLOUT_TYPE=""
for _arg in "${PASSTHROUGH_ARGS[@]}"; do
    case "$_arg" in
        parallelism.sp_size=*|+parallelism.sp_size=*|++parallelism.sp_size=*)
            _HYDRA_SP_SIZE="${_arg#*=}"
            ;;
        parallelism.tp_size=*|+parallelism.tp_size=*|++parallelism.tp_size=*)
            _HYDRA_TP_SIZE="${_arg#*=}"
            ;;
        inference.rollout_type=*|+inference.rollout_type=*)
            _HYDRA_ROLLOUT_TYPE="${_arg#*=}"
            ;;
    esac
done
if [ -n "$_HYDRA_SP_SIZE" ]; then
    if [ "$_SP_SIZE_FROM_CLI" -eq 1 ] && [ "$SP_SIZE" != "$_HYDRA_SP_SIZE" ]; then
        echo "Error: --sp-size ($SP_SIZE) conflicts with "
        echo "       parallelism.sp_size ($_HYDRA_SP_SIZE)."
        exit 1
    fi
    SP_SIZE="$_HYDRA_SP_SIZE"
elif [ -n "$SP_SIZE" ] && [ "$SP_SIZE" != "$_yaml_sp" ]; then
    PASSTHROUGH_ARGS+=("parallelism.sp_size=$SP_SIZE")
fi
if [ -n "$_HYDRA_TP_SIZE" ]; then
    if [ "$_TP_SIZE_FROM_CLI" -eq 1 ] && [ "$TP_SIZE" != "$_HYDRA_TP_SIZE" ]; then
        echo "Error: --tp-size ($TP_SIZE) conflicts with parallelism.tp_size ($_HYDRA_TP_SIZE)." >&2
        exit 1
    fi
    TP_SIZE="$_HYDRA_TP_SIZE"
elif [ -n "$TP_SIZE" ] && [ "$TP_SIZE" != "$_yaml_tp" ]; then
    PASSTHROUGH_ARGS+=("parallelism.tp_size=$TP_SIZE")
fi
if [[ ! "$SP_SIZE" =~ ^[1-9][0-9]*$ || ! "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Error: sp_size and tp_size must be positive integers." >&2
    exit 1
fi
MODEL_PARALLEL_SIZE=$((SP_SIZE * TP_SIZE))
if [ -n "$_HYDRA_ROLLOUT_TYPE" ]; then
    if [ "$_ROLLOUT_TYPE_FROM_CLI" -eq 1 ] \
            && [ "$ROLLOUT_TYPE" != "$_HYDRA_ROLLOUT_TYPE" ]; then
        echo "Error: --rollout-type ($ROLLOUT_TYPE) conflicts with "
        echo "       inference.rollout_type ($_HYDRA_ROLLOUT_TYPE)."
        exit 1
    fi
    ROLLOUT_TYPE="$_HYDRA_ROLLOUT_TYPE"
elif [ -n "$ROLLOUT_TYPE" ] && [ "$ROLLOUT_TYPE" != "$_yaml_rt" ]; then
    PASSTHROUGH_ARGS+=("inference.rollout_type=$ROLLOUT_TYPE")
fi

# ================= Resolve distributed inference topology =================
# HOST_NUM maps directly to torchrun --nnodes and defaults to one node.
NNODES="${HOST_NUM:-1}"

_PLATFORM_INDEX="${INDEX:-${_ENV_NODE_RANK:-}}"
if [ -z "$_PLATFORM_INDEX" ] && [ -n "${NODE_IP_LIST:-}" ]; then
    _myips=$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -v '^127\.')
    _idx=0
    while IFS= read -r _node_entry; do
        _node_ip=$(echo "$_node_entry" | sed 's/:.*$//' | tr -d '[:space:]')
        [ -n "$_node_ip" ] || continue
        for _myip in $_myips; do
            if [ "$_node_ip" = "$_myip" ]; then
                _PLATFORM_INDEX="$_idx"
                break 2
            fi
        done
        _idx=$((_idx + 1))
    done < <(echo "${NODE_IP_LIST}" | tr ',' '\n')
fi
_PLATFORM_INDEX="${_PLATFORM_INDEX:-0}"
_PLATFORM_INDEX=$((_PLATFORM_INDEX + INDEX_INCREMENT))

if [ -n "$_CLI_NNODES" ]; then
    NNODES="$_CLI_NNODES"
fi

NODE_RANK="${_CLI_NODE_RANK:-$_PLATFORM_INDEX}"
MASTER_ADDR="${_CLI_MASTER_ADDR:-${_ENV_MASTER_ADDR:-${CHIEF_IP:-}}}"
if [ -z "$MASTER_ADDR" ] && [ -n "${NODE_IP_LIST:-}" ]; then
    MASTER_ADDR=$(echo "${NODE_IP_LIST}" | tr ',' '\n' \
        | sed -n '1p' | sed 's/:.*$//' | tr -d '[:space:]')
fi
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${_CLI_MASTER_PORT:-${_ENV_MASTER_PORT:-29504}}"
_TOPOLOGY_MODE="${NNODES}-node"

RDZV_PORT="$MASTER_PORT"
TOTAL_GPUS=$(( NUM_GPUS * NNODES ))

export GLOBAL_NUM_NODES="${GLOBAL_NUM_NODES:-1}"
export GLOBAL_NODE_RANK="${GLOBAL_NODE_RANK:-0}"
export NNODES NODE_RANK MASTER_ADDR MASTER_PORT

# ================= Select Python script based on rollout_type =================
case "$ROLLOUT_TYPE" in
    one_shot|seg_ar|chunk_ar)
        PYTHON_SCRIPT="hymm/inference.py"
        ;;
    *)
        echo "Error: Unknown rollout_type '${ROLLOUT_TYPE}'. Must be one of: one_shot, seg_ar, chunk_ar"
        exit 1
        ;;
esac

echo "=========================================="
echo "Unified Inference Launcher"
echo "=========================================="
echo "Experiment:    ${EXPERIMENT}"
echo "Rollout type:  ${ROLLOUT_TYPE}"
echo "Python script: ${PYTHON_SCRIPT}"
echo "GPUs per node: $NUM_GPUS"
echo "Total GPUs:    $TOTAL_GPUS"
echo "SP Size:       $SP_SIZE"
echo "TP Size:       $TP_SIZE"
echo "Topology:      ${_TOPOLOGY_MODE}"
echo "Platform:      node ${_PLATFORM_INDEX}/${NNODES}"
echo "Torchrun:      node ${NODE_RANK}/${NNODES}"
echo "Rendezvous:    ${MASTER_ADDR}:${RDZV_PORT}"
echo "Prompt shard:  ${GLOBAL_NODE_RANK:-0}/${GLOBAL_NUM_NODES:-1}"
echo "Extra CLI args: ${PASSTHROUGH_ARGS[*]}"
echo "=========================================="

# ================= Build launcher =================
if [ $(( TOTAL_GPUS % MODEL_PARALLEL_SIZE )) -ne 0 ]; then
    echo "Error: Total GPUs ($TOTAL_GPUS) must be a multiple of SP_SIZE * TP_SIZE ($MODEL_PARALLEL_SIZE)"
    exit 1
fi

if [ "$TOTAL_GPUS" -gt 1 ] 2>/dev/null; then
    # Use torchrun for multi-GPU: supports SP, TP, and independent DP replicas
LAUNCHER="torchrun --nnodes=$NNODES --nproc_per_node=$NUM_GPUS --node_rank=$NODE_RANK --rdzv_endpoint=${MASTER_ADDR}:${RDZV_PORT} --rdzv_id=$RDZV_PORT"
else
    LAUNCHER="python"
fi

# ================= Run inference =================
if [ -n "$DEMO_REPLAY_MANIFEST" ]; then
    if [ "$ROLLOUT_TYPE" != "seg_ar" ] || [ "$TOTAL_GPUS" -ne "$MODEL_PARALLEL_SIZE" ]; then
        echo "Demo replay requires seg_ar and exactly one SP/TP model replica per torchrun world." >&2
        exit 1
    fi
    _REPLAY_OUTPUT=""
    for _arg in "${PASSTHROUGH_ARGS[@]}"; do
        case "$_arg" in
            inference.save_path=*) _REPLAY_OUTPUT="${_arg#*=}" ;;
            inference.prompt=*|inference.max_iters=*|inference.max_segments=*)
                echo "Demo replay uses recorded turns; remove $_arg" >&2
                exit 1 ;;
        esac
    done
    if [ -z "$_REPLAY_OUTPUT" ]; then
        echo "Demo replay requires an explicit inference.save_path=<new-output-directory>." >&2
        exit 1
    fi
    $LAUNCHER apps/seg_ar_web_server.py \
        --replay-manifest "$DEMO_REPLAY_MANIFEST" \
        --output-dir "$_REPLAY_OUTPUT" \
        +experiment="${EXPERIMENT}" \
        "${PASSTHROUGH_ARGS[@]}"
    exit $?
fi
$LAUNCHER ${PYTHON_SCRIPT} \
    +experiment="${EXPERIMENT}" \
    "${PASSTHROUGH_ARGS[@]}"
