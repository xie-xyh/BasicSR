#!/usr/bin/env bash
set -euo pipefail

protocol=$1
long_run=$2
tools=research/011_wavelet_unet/tools
export PATH="$HOME/miniconda3/envs/basicsr/bin:$PATH"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
deadline=$((SECONDS + 21600))

wait_file() {
    until [[ -s "$1" ]]; do
        (( SECONDS < deadline )) || return 1
        sleep 30
    done
}

wait_file "$long_run/batch_retry.exit"
[[ -s "$long_run/official_audit.json" ]]
free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
(( free >= 47000 ))
round="$protocol/backbone_011k"
mkdir "$round"
batch=0
for size in 20 16 12; do
    summary="$round/dea_batch_$size.json"
    if CUDA_VISIBLE_DEVICES=0 python -u "$tools/benchmark_batches.py" "$protocol/011k_dea.yml" \
        "$size" "$summary" --warmup 4 --steps 6 >"$round/dea_batch_$size.log" 2>&1; then
        if jq -e '.peak_reserved_gib <= 43' "$summary" >/dev/null; then
            batch=$size
            break
        fi
    elif ! rg -qi 'out of memory|CUDNN_STATUS_ALLOC_FAILED' "$round/dea_batch_$size.log"; then
        cat "$round/dea_batch_$size.log" >&2
        exit 1
    fi
done
(( batch > 0 ))
CUDA_VISIBLE_DEVICES=0 python -u "$tools/benchmark_batches.py" "$protocol/011k_wavelet.yml" \
    "$batch" "$round/wavelet_batch_$batch.json" --warmup 4 --steps 6 >"$round/wavelet_batch_$batch.log" 2>&1
python - "$protocol" "$round" "$batch" <<'PY'
from pathlib import Path
import sys

import yaml

protocol, output = map(Path, sys.argv[1:3])
batch = int(sys.argv[3])
configs = {arm: yaml.safe_load((protocol / f'011k_{arm}.yml').read_text()) for arm in ('wavelet', 'dea')}
recipes = [{key: value for key, value in config.items() if key not in ('name', 'network_g')}
           for config in configs.values()]
assert recipes[0] == recipes[1]
for arm, config in configs.items():
    config['datasets']['train']['batch_size_per_gpu'] = batch
    config['name'] = f'probe_011k_matched_{arm}_hust_v2_8000iter_b{batch}'
    (output / f'{arm}.yml').write_text(yaml.safe_dump(config, sort_keys=False))
PY
for arm in dea wavelet; do
    run="$round/run_$arm"
    bash "$tools/launch_probe.sh" 0 "$round/$arm.yml" "$run"
    wait_file "$run/exit_code"
    [[ $(cat "$run/exit_code") == 0 ]]
done
python "$tools/watch_training.py" "$round"/run_{dea,wavelet} --csv "$round/curves.csv"
CUDA_VISIBLE_DEVICES=0 python -u "$tools/paired_eval.py" \
    "$round/wavelet.yml" "experiments/probe_011k_matched_wavelet_hust_v2_8000iter_b$batch/models/net_g_latest.pth" \
    "$round/dea.yml" "experiments/probe_011k_matched_dea_hust_v2_8000iter_b$batch/models/net_g_latest.pth" \
    "$round/paired_8000" --correction-check --mid-profile
