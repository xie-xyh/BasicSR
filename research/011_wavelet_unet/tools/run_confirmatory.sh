#!/usr/bin/env bash
set -euo pipefail

config_dir=$(realpath "$1")
snapshot=$(realpath "$2")
mode=${3:---check}
root=$PWD
export PATH="$HOME/miniconda3/envs/basicsr/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

(cd "$snapshot"; sha256sum -c --quiet source.sha256)
python - "$config_dir" "$root" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import yaml

directory, root = map(Path, sys.argv[1:])
fingerprints = json.loads((directory / 'sha256.json').read_text())
for path in (*directory.glob('*.yml'), *directory.glob('*_manifest.txt')):
    if hashlib.sha256(path.read_bytes()).hexdigest() != fingerprints[path.name]:
        raise ValueError(f'Changed confirmatory input: {path}')
paths = [directory / f'{arm}_train.yml' for arm in ('control', 'candidate')]
train = [yaml.safe_load(path.read_text()) for path in paths]
if train[0]['name'] == train[1]['name']:
    raise ValueError('Training names must be distinct.')
for config in train:
    if config['train']['total_iter'] != 60000 or config['train']['scheduler']['periods'] != [60000]:
        raise ValueError('Expected matching 60k budget and schedule.')
    for options in config['datasets'].values():
        if not (root / options['meta_info_file']).is_file() or '/test/' in options['dataroot_gt']:
            raise ValueError('Training must use frozen training and internal validation manifests.')
print('Verified snapshot, manifests and training budget.')
PY
[[ "$mode" == --check ]] && exit 0
[[ "$mode" == --run || "$mode" == --staggered ]]

read_name() {
    python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["name"])' "$1"
}

initial_gpus=(0)
[[ "$mode" != --run ]] || initial_gpus+=(1)
for gpu in "${initial_gpus[@]}"; do
    free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    (( free >= 24000 )) || { printf 'GPU %s only has %s MiB free.\n' "$gpu" "$free" >&2; exit 1; }
done

control_config="$config_dir/control_train.yml"
candidate_config="$config_dir/candidate_train.yml"
control_name=$(read_name "$control_config")
candidate_name=$(read_name "$candidate_config")
[[ ! -e "$snapshot/experiments/$control_name" && ! -e "$snapshot/experiments/$candidate_name" ]]
mkdir "$config_dir/run"
export PYTHONPATH="$snapshot"
run_arm() {
    local arm=$1 gpu=$2 config=$3
    local status=0
    bash "$(dirname "$0")/start_swanlab_bridge.sh" "$config_dir/${arm}_train.yml" \
        "$snapshot/tb_logger" "$config_dir/run/$arm.exit" "$config_dir/run/${arm}_swanlab"
    printf '%s GPU%s\n' "$(date -Is)" "$gpu" >"$config_dir/run/$arm.start"
    CUDA_VISIBLE_DEVICES="$gpu" python -u "$snapshot/basicsr/train.py" -opt "$config" \
        >"$config_dir/run/$arm.log" 2>&1 || status=$?
    printf '%s\n' "$status" >"$config_dir/run/$arm.exit"
    return "$status"
}

run_arm candidate 0 "$candidate_config" &
candidate_pid=$!
run_control() {
    if [[ "$mode" == --staggered ]]; then
        if [[ -n "${OWH_CONTROL_WAIT_EXIT:-}" ]]; then
            local deadline=$((SECONDS + 14400))
            printf '%s Waiting for previous GPU1 experiment to exit: %s\n' "$(date -Is)" "$OWH_CONTROL_WAIT_EXIT"
            until [[ -s "$OWH_CONTROL_WAIT_EXIT" ]]; do
                (( SECONDS < deadline )) || return 1
                sleep 30
            done
            [[ $(cat "$OWH_CONTROL_WAIT_EXIT") == 0 ]] || return 1
        fi
        printf '%s Waiting for GPU1 with at least 45000 MiB free.\n' "$(date -Is)"
        while :; do
            local free
            free=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
            if (( free >= 45000 )); then
                break
            fi
            if [[ -s "$config_dir/run/candidate.exit" ]]; then
                [[ $(cat "$config_dir/run/candidate.exit") == 0 ]] || return 1
                free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
                if (( free >= 45000 )); then
                    run_arm control 0 "$control_config"
                    return $?
                fi
            fi
            sleep 60
        done
    fi
    run_arm control 1 "$control_config"
}
run_control &
control_pid=$!
control_status=0
candidate_status=0
wait "$control_pid" || control_status=$?
wait "$candidate_pid" || candidate_status=$?
(( control_status == 0 && candidate_status == 0 ))
for arm in control candidate; do
    rg -q 'iter:[[:space:]]*60,000, lr:' "$config_dir/run/$arm.log"
done

CUDA_VISIBLE_DEVICES=0 python -u "$snapshot/research/011_wavelet_unet/tools/paired_eval.py" \
    "$config_dir/control_official.yml" "$snapshot/experiments/$control_name/models/net_g_latest.pth" \
    "$config_dir/candidate_official.yml" "$snapshot/experiments/$candidate_name/models/net_g_latest.pth" \
    "$config_dir/official_60000" --correction-check
