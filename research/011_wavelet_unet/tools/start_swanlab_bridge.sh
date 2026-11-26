#!/usr/bin/env bash
set -euo pipefail

config=$1
tensorboard=$2
terminal=$3
output=$4
python="$HOME/miniconda3/envs/basicsr/bin/python"
tool="$(dirname "$0")/swanlab_bridge.py"

status=0
"$python" - "$config" <<'PY' || status=$?
from pathlib import Path
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
logger = config['logger']
if config['name'].startswith('probe_') and (logger.get('swanlab') or {}).get('project') != 'dehaze-probes':
    raise SystemExit('Probe runs must use the dehaze-probes SwanLab project.')
if not logger.get('use_tb_logger') or 'debug' in config['name']:
    raise SystemExit('TensorBoard logging is required for SwanLab tracking.')
if logger.get('use_swanlab'):
    if (logger.get('swanlab') or {}).get('mode') != 'online':
        raise SystemExit('Native SwanLab must use online mode.')
    raise SystemExit(10)
import swanlab
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
PY
(( status != 10 )) || exit 0
(( status == 0 )) || exit "$status"
mkdir "$output"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1
setsid -f bash -c '
    echo $$ >"$6/pid"
    "$1" -u "$2" "$6" "$4" "$6/data" candidate --config "$3" --exit-file "$5"
    status=$?
    printf "%s\n" "$status" >"$6/exit_code"
    exit "$status"
' swan "$python" "$tool" "$config" "$tensorboard" "$terminal" "$output" \
    </dev/null >"$output/console.log" 2>&1
