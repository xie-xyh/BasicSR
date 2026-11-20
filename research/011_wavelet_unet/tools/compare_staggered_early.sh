#!/usr/bin/env bash
set -euo pipefail

run=$(realpath "$1")
snapshot=$(realpath "$2")
tools=$(realpath "$(dirname "$0")")
export PATH="$HOME/miniconda3/envs/basicsr/bin:$PATH"
export PYTHONPATH="$snapshot" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

weights="$run/retained_8000"
mkdir "$weights"
retain_arm() {
    local arm=$1 name file deadline
    name=$(sed -n 's/^name: //p' "$run/${arm}_train.yml")
    file="$snapshot/experiments/$name/models/net_g_8000.pth"
    deadline=$((SECONDS+86400))
    until [[ -s "$file" ]] && rg -q 'iter: *8,100, lr:' "$run/run/$arm.log"; do
        if (( SECONDS >= deadline )) || [[ -s "$run/run/$arm.exit" ]]; then
            printf '%s: no complete 8k checkpoint for %s\n' "$(date -Is)" "$arm" >&2
            return 1
        fi
        sleep 30
    done
    cp "$file" "$weights/$arm.pth.tmp"
    cmp "$file" "$weights/$arm.pth.tmp"
    mv "$weights/$arm.pth.tmp" "$weights/$arm.pth"
    printf '%s: retained %s at 8k\n' "$(date -Is)" "$arm"
}

retain_arm candidate &
candidate_pid=$!
retain_arm control &
control_pid=$!
wait "$candidate_pid"
wait "$control_pid"
sha256sum "$weights"/{control,candidate}.pth >"$weights/sha256.txt"
nice -n 10 python -u "$tools/paired_eval.py" \
    "$run/control_train.yml" "$weights/control.pth" "$run/candidate_train.yml" "$weights/candidate.pth" \
    "$run/paired_internal_8000_cpu" --device cpu --mid-profile --correction-check
