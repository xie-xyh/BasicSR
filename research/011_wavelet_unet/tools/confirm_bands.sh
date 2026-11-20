#!/usr/bin/env bash
set -euo pipefail

protocol=$1
tools=research/011_wavelet_unet/tools
deadline=$((SECONDS + 21600))
until [[ -s "$protocol/combined_vs_plain_8000.json" && -s "$protocol/backbone_driver.exit" ]]; do
    (( SECONDS < deadline )) || exit 1
    if [[ -s "$protocol/backbone_driver.exit" && $(cat "$protocol/backbone_driver.exit") != 0 ]]; then
        exit 1
    fi
    sleep 30
done
[[ $(cat "$protocol/backbone_driver.exit") == 0 ]]
candidate=mid
if jq -e '
    [.["OWH-Dense-internal-v2"], .["OWH-Mist-internal-v2"]]
    | all(.psnr.delta > 0 and .ssim.delta >= 0)
' "$protocol/paired_combined_vs_mid_8000/summary.json" >/dev/null &&
   jq -e '
    [.["OWH-Dense-internal-v2"], .["OWH-Mist-internal-v2"]]
    | all(.psnr.ci95[0] > 0 and .ssim.delta >= 0)
' "$protocol/combined_vs_plain_8000.json" >/dev/null; then
    candidate=joint
fi
printf '%s Waiting for GPU0 to have at least 45000 MiB free.\n' "$(date -Is)"
while :; do
    available=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
    if (( available >= 45000 )); then
        break
    fi
    if (( SECONDS >= deadline )); then
        printf '%s GPU wait timed out; free memory (MiB): %s\n' "$(date -Is)" "$available" >&2
        exit 1
    fi
    sleep 60
done
printf '%s 011n candidate: %s\n' "$(date -Is)" "$candidate"
run="$protocol/confirm_011n_$candidate"
bash "$tools/run_confirmatory.sh" "$run" "$protocol/confirm_code_011n" --staggered
"$HOME/miniconda3/envs/basicsr/bin/python" "$tools/audit_final.py" "$run"
