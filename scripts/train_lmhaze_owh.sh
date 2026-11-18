#!/usr/bin/env bash
set -euo pipefail

GPUS="${1:-2}"
if [[ $# -gt 0 ]]; then
    shift
fi
if ! [[ "${GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: bash scripts/train_lmhaze_owh.sh [num_gpus] [--auto_resume] [--debug]" >&2
    exit 1
fi

ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_PATH}"
if ! python -c 'from mamba_ssm.ops.selective_scan_interface import selective_scan_fn' 2>/dev/null; then
    echo "LMHaze training requires mamba-ssm and causal-conv1d; install the versions in requirements.txt." >&2
    exit 1
fi
CONFIG="options/train/LMHaze/train_LMHaze_owh_full.yml"
PORT="${PORT:-29511}" bash scripts/dist_train.sh "${GPUS}" "${CONFIG}" "$@"
