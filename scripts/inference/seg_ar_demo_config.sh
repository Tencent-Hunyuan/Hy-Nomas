#!/bin/bash
# Source from the repository root to resolve YAML defaults and launch overrides.
export PYTHONPATH="${PWD}${PYTHONPATH:+:${PYTHONPATH}}"
_demo_config="$(python apps/seg_ar_demo_config.py "$@")"
eval "${_demo_config}"
unset _demo_config

if [[ ! "$SP_SIZE" =~ ^[1-9][0-9]*$ || ! "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "SP_SIZE and TP_SIZE must be positive integers" >&2
  return 2 2>/dev/null || exit 2
fi
export MODEL_PARALLEL_SIZE=$((SP_SIZE * TP_SIZE))
