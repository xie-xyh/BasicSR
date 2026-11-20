#!/usr/bin/env bash
set -euo pipefail

run=$(realpath "$1")
snapshot=$(realpath "$2")
tool=$(realpath "$(dirname "$0")/benchmark_batches.py")
export PATH="$HOME/miniconda3/envs/basicsr/bin:$PATH"
export PYTHONPATH="$snapshot" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

deadline=$((SECONDS + 43200))
until [[ -s "$run/driver.exit" ]]; do
    (( SECONDS < deadline )) || exit 1
    sleep 60
done
[[ $(cat "$run/driver.exit") == 0 && -s "$run/official_60000/summary.json" ]]
for gpu in 0 1; do
    free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    (( free >= 45000 )) || { printf 'GPU %s only has %s MiB free.\n' "$gpu" "$free" >&2; exit 1; }
done

mkdir "$run/batch_benchmark"
benchmark_arm() {
    local arm=$1 gpu=$2
    local batch
    for batch in 20 32 40; do
        local prefix="$run/batch_benchmark/${arm}_${batch}"
        local code=0
        CUDA_VISIBLE_DEVICES="$gpu" python -u "$tool" "$run/${arm}_train.yml" "$batch" "$prefix.json" \
            >"$prefix.log" 2>&1 || code=$?
        printf '%s\n' "$code" >"$prefix.exit"
        if (( code != 0 )); then
            break
        fi
    done
}

benchmark_arm control 1 &
control_pid=$!
benchmark_arm candidate 0 &
candidate_pid=$!
wait "$control_pid"
wait "$candidate_pid"
