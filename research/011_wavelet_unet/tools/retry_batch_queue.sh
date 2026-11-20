#!/usr/bin/env bash
set -euo pipefail

run=$(realpath "$1")
snapshot=$(realpath "$2")
coarse_exit=$(realpath -m "$3")
tools=$(realpath "$(dirname "$0")")
deadline=$((SECONDS + 43200))

wait_file() {
    while [[ ! -s "$1" ]]; do
        (( SECONDS < deadline )) || return 1
        sleep 30
    done
}

wait_file "$run/batch_driver.exit"
if [[ -d "$run/batch_benchmark" ]]; then
    printf '已有吞吐输出目录，保留现有结果。\n'
    exit 0
fi
[[ $(cat "$run/driver.exit") == 0 && -s "$run/official_60000/summary.json" ]]
wait_file "$coarse_exit"

while :; do
    available=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    if awk '$1 >= 45000 {free++} END {exit free != 2}' <<<"$available"; then
        break
    fi
    (( SECONDS < deadline )) || exit 1
    sleep 30
done

printf '%s 两张卡空闲，补跑吞吐测试\n' "$(date -Is)"
bash "$tools/benchmark_after_training.sh" "$run" "$snapshot"
