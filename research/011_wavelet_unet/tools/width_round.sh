#!/usr/bin/env bash
set -euo pipefail

protocol=$1
tools=research/011_wavelet_unet/tools
export PATH="$HOME/miniconda3/envs/basicsr/bin:$PATH"
export PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

wait_success() {
    local run=$1
    local deadline=$((SECONDS + 5400))
    until [[ -s "$run/exit_code" ]]; do
        if (( SECONDS >= deadline )); then
            printf '等待超时：%s\n' "$run" >&2
            return 1
        fi
        sleep 30
    done
    [[ $(cat "$run/exit_code") == 0 ]]
}

until [[ -s "$protocol/paired_8000/summary.json" ]]; do
    if rg -q '^驱动结束 ' "$protocol/round_driver.log"; then
        printf '011d 驱动结束，配对报告未生成。\n' >&2
        exit 1
    fi
    sleep 30
done
wait_success "$protocol/run_011d_single_scale"
wait_success "$protocol/run_011d_multiscale"

python - "$protocol" <<'PY'
from pathlib import Path
import sys

import yaml

protocol = Path(sys.argv[1])
config = yaml.safe_load((protocol / '011b_control.yml').read_text())
config['name'] = 'probe_011b_width_control_hust_v2_8000iter'
(protocol / '011b_width_control.yml').write_text(yaml.safe_dump(config, sort_keys=False))
config['name'] = 'probe_011f_width_hust_v2_8000iter'
config['network_g']['widths'] = [64, 96, 160, 256]
(protocol / '011f_width.yml').write_text(yaml.safe_dump(config, sort_keys=False))
PY

printf '%s：启动容量对照\n' "$(date -Is)"
bash "$tools/launch_probe.sh" 0 "$protocol/011f_width.yml" "$protocol/run_011f_width"
bash "$tools/launch_probe.sh" 1 "$protocol/011b_width_control.yml" "$protocol/run_011b_width_control"
wait_success "$protocol/run_011f_width"
wait_success "$protocol/run_011b_width_control"
python "$tools/watch_training.py" "$protocol/run_011b_width_control" "$protocol/run_011f_width" \
    --csv "$protocol/width_curves.csv"
CUDA_VISIBLE_DEVICES=0 python -u "$tools/paired_eval.py" \
    "$protocol/011b_width_control.yml" experiments/probe_011b_width_control_hust_v2_8000iter/models/net_g_latest.pth \
    "$protocol/011f_width.yml" experiments/probe_011f_width_hust_v2_8000iter/models/net_g_latest.pth \
    "$protocol/paired_width_8000" --phase-check
