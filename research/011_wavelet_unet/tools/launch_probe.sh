#!/usr/bin/env bash
set -eu

gpu=$1
config=$2
output=$3
gpu_free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
if (( gpu_free < 24000 )); then
    printf 'GPU %s 剩余 %s MiB，暂缓启动\n' "$gpu" "$gpu_free" >&2
    exit 1
fi
mkdir "$output"
sha256sum "$config" basicsr/archs/wavelet_unet_arch.py > "$output/source.sha256"
revision=$(git rev-parse HEAD)
printf '%s\n' "$revision" > "$output/revision.txt"
bash "$(dirname "$0")/start_swanlab_bridge.sh" "$config" "$PWD/tb_logger" "$output/exit_code" "$output/swanlab"
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu"
setsid -f bash -c '
    config=$1
    output=$2
    exec >>"$output/console.log" 2>&1
    printf "开始 %s，GPU %s\n" "$(date -Is)" "$CUDA_VISIBLE_DEVICES"
    "$HOME/miniconda3/envs/basicsr/bin/python" -u basicsr/train.py -opt "$config" &
    echo $! > "$output/pid"
    wait "$(cat "$output/pid")"
    status=$?
    printf "%s\n" "$status" > "$output/exit_code"
    printf "结束 %s，退出码 %s\n" "$(date -Is)" "$status"
    exit "$status"
' probe "$config" "$output" </dev/null >/dev/null 2>&1
