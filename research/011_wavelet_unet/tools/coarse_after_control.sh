#!/usr/bin/env bash
set -euo pipefail

protocol=$1
long_run=$2
tools=research/011_wavelet_unet/tools
export PATH="$HOME/miniconda3/envs/basicsr/bin:$PATH"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

wait_success() {
    local status=$1 deadline=$((SECONDS + 21600))
    until [[ -s "$status" ]]; do
        (( SECONDS < deadline )) || { printf '等待超时：%s\n' "$status" >&2; return 1; }
        sleep 30
    done
    [[ $(cat "$status") == 0 ]]
}

wait_success "$long_run/run/control.exit"
for arm in candidate control; do
    free=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
    (( free >= 45000 ))
    run="$protocol/run_011i_$arm"
    printf '%s 启动低频残差 %s 组\n' "$(date -Is)" "$arm"
    bash "$tools/launch_probe.sh" 1 "$protocol/011i_$arm.yml" "$run"
    wait_success "$run/exit_code"
done

python "$tools/watch_training.py" "$protocol"/run_011i_{control,candidate} --csv "$protocol/coarse_curves.csv"
CUDA_VISIBLE_DEVICES=1 python -u "$tools/paired_eval.py" \
    "$protocol/011i_control.yml" experiments/probe_011i_coarse_control_hust_v2_8000iter/models/net_g_latest.pth \
    "$protocol/011i_candidate.yml" experiments/probe_011i_coarse_candidate_hust_v2_8000iter/models/net_g_latest.pth \
    "$protocol/paired_coarse_8000" --correction-check --spatial-check
