#!/usr/bin/env bash
set -euo pipefail

# 014 探针启动脚本（防重复启动 + 强制检查断点保存配置 + SwanLab 同步）
# 用法：
#   bash research/014_maritime_sfnet/tools/launch_probe.sh <gpu> <config> <output_dir> [min_free_mib] [--auto_resume]
# 说明：
#   - 首次启动拒绝覆盖已有 experiments/probes/<name> 与 output_dir；
#   - 异常中断续训时传入第 5 参数 --auto_resume（需使用新的 output_dir 记录本次日志），
#     自动加载 experiments/probes/<name>/training_states/*.state 继续训练。

gpu=$1
config=$2
output=$3
minimum_free=${4:-38000}
resume_flag=${5:-}
python="$HOME/miniconda3/envs/basicsr/bin/python"

if [[ -n "$resume_flag" && "$resume_flag" != "--auto_resume" ]]; then
    printf '第 5 参数仅支持留空或 --auto_resume，收到：%s\n' "$resume_flag" >&2
    exit 1
fi

"$python" - "$config" "$resume_flag" <<'PY'
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text())
resume_mode = sys.argv[2] == '--auto_resume'
assert config['name'].startswith('probe_014'), 'Probe config name must start with probe_014'
assert config['path'].get('experiments_root') == 'experiments/probes'
assert config['logger'].get('tb_logdir') == 'tb_logger/probes'
assert config['logger'].get('swanlab', {}).get('project') == 'dehaze-probes'
assert config['logger'].get('use_tb_logger')
assert not config['logger'].get('use_swanlab')
assert config['logger'].get('save_checkpoint_freq', 0) > 0, 'save_checkpoint_freq must be enabled for resume safety'
assert config['path'].get('resume_state') is None
experiment = Path(config['path']['experiments_root']) / config['name']
if resume_mode:
    states = list((experiment / 'training_states').glob('*.state'))
    assert states, f'No .state checkpoints found for --auto_resume under {experiment / "training_states"}'
else:
    assert not experiment.exists(), f'Experiment already exists (pass --auto_resume to continue): {experiment}'
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
sha256sum "$config" basicsr/archs/maritime_sfnet_arch.py basicsr/models/maritime_sf_model.py > "$output/source.sha256"
git rev-parse HEAD > "$output/revision.txt"
bash research/011_wavelet_unet/tools/start_swanlab_bridge.sh \
    "$config" "$PWD/tb_logger/probes" "$output/exit_code" "$output/swanlab"
export OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$gpu"
setsid -f bash -c '
    config=$1
    output=$2
    resume_flag=$3
    exec >>"$output/console.log" 2>&1
    printf "开始 %s，GPU %s，resume=%s\n" "$(date -Is)" "$CUDA_VISIBLE_DEVICES" "${resume_flag:-none}"
    if [[ -n "$resume_flag" ]]; then
        "$HOME/miniconda3/envs/basicsr/bin/python" -u basicsr/train.py -opt "$config" --auto_resume &
    else
        "$HOME/miniconda3/envs/basicsr/bin/python" -u basicsr/train.py -opt "$config" &
    fi
    echo $! > "$output/pid"
    wait "$(cat "$output/pid")"
    status=$?
    printf "%s\n" "$status" > "$output/exit_code"
    printf "结束 %s，退出码 %s\n" "$(date -Is)" "$status"
    exit "$status"
' probe "$config" "$output" "$resume_flag" </dev/null >/dev/null 2>&1
printf '已启动 %s，GPU %s，日志 %s/console.log\n' "$config" "$gpu" "$output"
