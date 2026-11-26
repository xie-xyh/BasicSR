#!/usr/bin/env bash
set -euo pipefail

protocol=$1
long_run=$2
tools=research/011_wavelet_unet/tools
export PATH="$HOME/miniconda3/envs/basicsr/bin:$PATH"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
deadline=$((SECONDS + 28800))

wait_file() {
    until [[ -s "$1" ]]; do
        (( SECONDS < deadline )) || { printf '等待超时：%s\n' "$1" >&2; return 1; }
        sleep 30
    done
}

wait_file "$protocol/coarse_driver.exit"
[[ $(cat "$protocol/coarse_driver.exit") == 0 ]]
wait_file "$long_run/batch_retry.exit"
[[ -s "$long_run/official_60000/summary.json" ]]
sha256sum -c --quiet "$protocol/run_011i_control/source.sha256"
sha256sum -c --quiet "$protocol/011j_recipe.sha256"
free=$(nvidia-smi -i 1 --query-gpu=memory.free --format=csv,noheader,nounits)
(( free >= 45000 ))

run="$protocol/run_011j_mid_band"
printf '%s 启动 011j 中尺度残差\n' "$(date -Is)"
bash "$tools/launch_probe.sh" 1 "$protocol/011j_mid_band.yml" "$run"
wait_file "$run/exit_code"
[[ $(cat "$run/exit_code") == 0 ]]
python "$tools/watch_training.py" "$protocol/run_011i_control" "$run" --csv "$protocol/mid_curves.csv"
CUDA_VISIBLE_DEVICES=1 python -u "$tools/paired_eval.py" \
    "$protocol/011i_control.yml" experiments/probes/probe_011i_coarse_control_hust_v2_8000iter/models/net_g_latest.pth \
    "$protocol/011j_mid_band.yml" experiments/probes/probe_011j_mid_band_hust_v2_8000iter/models/net_g_latest.pth \
    "$protocol/paired_mid_8000" --mid-profile --correction-check --spatial-check
