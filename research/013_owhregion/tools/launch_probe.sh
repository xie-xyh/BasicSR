#!/usr/bin/env bash
set -euo pipefail

gpu=$1
config=$2
output=$3
minimum_free=${4:-24000}
python="$HOME/miniconda3/envs/basicsr/bin/python"

"$python" - "$config" <<'PY'
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
assert config['name'].startswith('probe_013')
assert config['path'].get('experiments_root') == 'experiments/probes'
assert config['logger'].get('tb_logdir') == 'tb_logger/probes'
assert config['logger'].get('swanlab', {}).get('project') == 'dehaze-probes'
assert config['logger'].get('use_tb_logger')
assert not config['logger'].get('use_swanlab')
assert config['path'].get('resume_state') is None
experiment = Path(config['path']['experiments_root']) / config['name']
assert not experiment.exists(), f'Experiment already exists: {experiment}'
PY

if [[ -e "$output" ]]; then
    printf '输出目录已存在，拒绝重复启动：%s\n' "$output" >&2
    exit 1
fi
free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
if (( free < minimum_free )); then
    printf 'GPU %s 剩余 %s MiB，低于启动门槛 %s MiB\n' "$gpu" "$free" "$minimum_free" >&2
    exit 1
fi

mkdir "$output"
sha256sum "$config" basicsr/archs/owh_lfm_unet_arch.py > "$output/source.sha256"
git rev-parse HEAD > "$output/revision.txt"
bash research/011_wavelet_unet/tools/start_swanlab_bridge.sh \
    "$config" "$PWD/tb_logger/probes" "$output/exit_code" "$output/swanlab"
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
printf '已启动 %s，GPU %s，日志 %s/console.log\n' "$config" "$gpu" "$output"
