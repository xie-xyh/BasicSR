#!/usr/bin/env bash
set -euo pipefail

protocol=$(realpath "$1")
tools=$(realpath "$(dirname "$0")")
export PATH="$HOME/miniconda3/envs/basicsr/bin:$PATH"
run="$protocol/confirm_011p_backbone_draft"
snapshot="$protocol/confirm_code_011n"
deadline=$((SECONDS + 21600))
until [[ -s "$protocol/confirm_011n_queue.exit" ]]; do
    (( SECONDS < deadline )) || exit 1
    sleep 30
done
[[ $(cat "$protocol/confirm_011n_queue.exit") == 0 ]]
printf '%s Waiting for GPU0 for the DEA-Net arm.\n' "$(date -Is)"
while :; do
    free=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
    (( free < 47000 )) || break
    (( SECONDS < deadline )) || exit 1
    sleep 60
done
export OWH_CONTROL_WAIT_EXIT="$protocol/context_011o/driver.exit"
bash "$tools/run_confirmatory.sh" "$run" "$snapshot" --staggered
python "$tools/audit_final.py" "$run"
