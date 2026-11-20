#!/usr/bin/env bash
set -euo pipefail

protocol=$1
mode=${2:-dual}
out="$protocol/supervision_011qs"
tools=research/011_wavelet_unet/tools
export PATH="$HOME/miniconda3/envs/basicsr/bin:$PATH"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
deadline=$((SECONDS+172800))

wait_exit() {
    until [[ -s "$1" ]]; do
        (( SECONDS < deadline )) || { printf '等待超时：%s\n' "$1" >&2; return 1; }
        sleep 60
    done
    [[ $(cat "$1") == 0 ]]
}

case "$mode" in
    dual) wait_exit "$protocol/confirm_011p_queue.exit"; evaluation_gpu=0 ;;
    gpu1) wait_exit "$protocol/confirm_011p_backbone_draft/run/control.exit"; evaluation_gpu=1 ;;
    *) exit 2 ;;
esac
sha256sum -c --quiet "$out/recipe.sha256"

train_arm() {
    local arm=$1 gpu=$2 free
    while :; do
        free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
        (( free < 45000 )) || break
        (( SECONDS < deadline )) || return 1
        sleep 60
    done
    bash "$tools/launch_probe.sh" "$gpu" "$out/$arm.yml" "$out/run_$arm"
    wait_exit "$out/run_$arm/exit_code"
}

if [[ "$mode" == gpu1 ]]; then
    for arm in control smooth band; do
        train_arm "$arm" 1
    done
else
    train_arm control 0 &
    control_pid=$!
    train_arm smooth 1 &
    smooth_pid=$!
    wait "$control_pid"
    train_arm band 0 &
    band_pid=$!
    wait "$smooth_pid"
    wait "$band_pid"
fi
python "$tools/watch_training.py" "$out"/run_{control,smooth,band} --csv "$out/curves.csv"
for arm in smooth band; do
    control_name=$(sed -n 's/^name: //p' "$out/control.yml")
    candidate_name=$(sed -n 's/^name: //p' "$out/$arm.yml")
    CUDA_VISIBLE_DEVICES="$evaluation_gpu" python -u "$tools/paired_eval.py" \
        "$out/control.yml" "experiments/$control_name/models/net_g_latest.pth" \
        "$out/$arm.yml" "experiments/$candidate_name/models/net_g_latest.pth" \
        "$out/paired_$arm" --correction-check --mid-profile
done
python "$tools/compare_paired_tables.py" "$out/paired_band/pairs.csv" "$out/paired_smooth/pairs.csv" \
    "$out/smooth_vs_band.json" --control-arm candidate
