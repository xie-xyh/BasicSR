#!/usr/bin/env bash
set -euo pipefail

run=$(realpath "$1")
snapshot=$(realpath "$2")
shift 2
tools=$(realpath "$(dirname "$0")")
export PATH="$HOME/miniconda3/envs/basicsr/bin:$PATH"
export PYTHONPATH="$snapshot" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
deadline=$((SECONDS + 21600))

for step in "$@"; do
    weights="$run/retained_$step"
    until [[ -s "$weights/control.pth" && -s "$weights/candidate.pth" ]] &&
        rg -q "iter: *$(printf '%s,%03d' "$((step / 1000))" 100), lr:" "$run/run/candidate.log"; do
        if (( SECONDS >= deadline )) || [[ -s "$run/run/candidate.exit" ]]; then
            printf '%s 同步权重或训练日志未就绪\n' "$step" >&2
            exit 1
        fi
        sleep 30
    done
    printf '%s：开始 %s CPU 配对\n' "$(date -Is)" "$step"
    nice -n 10 python -u "$tools/paired_eval.py" "$run/control_train.yml" "$weights/control.pth" \
        "$run/candidate_train.yml" "$weights/candidate.pth" "$run/paired_internal_${step}_cpu" \
        --device cpu --mid-profile --correction-check --spatial-check
    python "$tools/watch_training.py" "$run"/run/*.log --csv "$run/internal_curves.csv"
done
