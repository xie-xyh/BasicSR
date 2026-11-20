#!/usr/bin/env bash
set -euo pipefail

protocol=$1
experiment="$protocol/context_011o"
tools=research/011_wavelet_unet/tools
export PATH="$HOME/miniconda3/envs/basicsr/bin:$PATH"
export PYTHONPATH="$PWD" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
deadline=$((SECONDS+43200))

wait_file() {
    until [[ -s "$1" ]]; do
        (( SECONDS < deadline )) || { printf '等待超时：%s\n' "$1" >&2; return 1; }
        sleep 30
    done
}

wait_file "$protocol/confirm_011n_queue.exit"
[[ $(cat "$protocol/confirm_011n_queue.exit") == 0 ]]
sha256sum -c --quiet "$experiment/recipe.sha256"
gpu=1
while :; do
    free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    (( free < 45000 )) || break
    (( SECONDS < deadline )) || { printf 'GPU%s 显存等待超时\n' "$gpu" >&2; exit 1; }
    sleep 60
done

for arm in rectangle square; do
    CUDA_VISIBLE_DEVICES="$gpu" python -u "$tools/benchmark_batches.py" "$experiment/$arm.yml" 20 \
        "$experiment/$arm.benchmark.json" --warmup 10 --steps 30 >"$experiment/$arm.benchmark.log" 2>&1
    jq -e '.peak_reserved_gib < 43' "$experiment/$arm.benchmark.json" >/dev/null
done
for arm in rectangle square; do
    run="$experiment/run_$arm"
    printf '%s: 启动 %s @ GPU%s\n' "$(date -Is)" "$arm" "$gpu"
    bash "$tools/launch_probe.sh" "$gpu" "$experiment/$arm.yml" "$run"
    wait_file "$run/exit_code"
    [[ $(cat "$run/exit_code") == 0 ]]
done

python "$tools/watch_training.py" "$experiment"/run_{square,rectangle} --csv "$experiment/curves.csv"
CUDA_VISIBLE_DEVICES="$gpu" python -u "$tools/paired_eval.py" \
    "$experiment/square.yml" experiments/probe_011o_context_square_hust_v2_8000iter/models/net_g_latest.pth \
    "$experiment/rectangle.yml" experiments/probe_011o_context_rectangle_hust_v2_8000iter/models/net_g_latest.pth \
    "$experiment/paired_8000" --mid-profile --correction-check --spatial-check
