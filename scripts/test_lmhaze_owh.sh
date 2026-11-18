#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-0}"
if [[ $# -gt 0 ]]; then
    shift
fi
ROOT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_PATH}"
export CUDA_VISIBLE_DEVICES="${GPU}"
python basicsr/test.py -opt options/test/LMHaze/test_LMHaze_owh.yml "$@"
