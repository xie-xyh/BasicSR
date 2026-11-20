#!/usr/bin/env bash
set -eu

current=$1
next=$2
tools=research/011_wavelet_unet/tools
export PATH="$HOME/miniconda3/envs/basicsr/bin:$PATH"
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

wait_success() {
    local run=$1
    local deadline=$((SECONDS + 3600))
    until [[ -s "$run/exit_code" ]]; do
        if (( SECONDS >= deadline )); then
            printf '未收到 %s 的退出记录\n' "$run" >&2
            return 1
        fi
        sleep 30
    done
    local result
    result=$(cat "$run/exit_code")
    if [[ "$result" != 0 ]]; then
        printf '%s 退出码 %s，请先查看训练日志\n' "$run" "$result" >&2
        return 1
    fi
}

wait_success "$current/run_011b"
wait_success "$current/run_011c"
python "$tools/watch_training.py" "$current/run_011b" "$current/run_011c" --csv "$current/curves.csv"

printf '%s：启动 v2 金字塔双对照\n' "$(date -Is)"
bash "$tools/launch_probe.sh" 0 "$next/011d_single_scale.yml" "$next/run_011d_single_scale" &
control_launch=$!
bash "$tools/launch_probe.sh" 1 "$next/011d_multiscale.yml" "$next/run_011d_multiscale" &
candidate_launch=$!
wait "$control_launch"
wait "$candidate_launch"

printf '%s：评估 v1 的 8k EMA\n' "$(date -Is)"
CUDA_VISIBLE_DEVICES=1 nice -n 5 python -u "$tools/paired_eval.py" \
    "$current/011b_control.yml" experiments/probe_011b_control_owh_internal_v1_8000iter/models/net_g_latest.pth \
    "$current/011c_detail_stem.yml" experiments/probe_011c_detail_stem_owh_internal_v1_8000iter/models/net_g_latest.pth \
    "$current/paired_8000" --phase-check

wait_success "$next/run_011d_single_scale"
wait_success "$next/run_011d_multiscale"
python "$tools/watch_training.py" "$next/run_011d_single_scale" "$next/run_011d_multiscale" --csv "$next/curves.csv"

printf '%s：评估 v2 的 8k EMA\n' "$(date -Is)"
CUDA_VISIBLE_DEVICES=0 nice -n 5 python -u "$tools/paired_eval.py" \
    "$next/011d_single_scale.yml" experiments/probe_011d_single_scale_owh_internal_v2_8000iter/models/net_g_latest.pth \
    "$next/011d_multiscale.yml" experiments/probe_011d_multiscale_owh_internal_v2_8000iter/models/net_g_latest.pth \
    "$next/paired_8000" --phase-check
