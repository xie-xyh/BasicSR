#!/usr/bin/env bash
# 双服务器实验产物同步脚本
#
# 用法（在本仓库根目录执行）：
#   bash scripts/sync_experiments.sh push    # 本机 → 远端（默认）
#   bash scripts/sync_experiments.sh pull    # 远端 → 本机
#   bash scripts/sync_experiments.sh both    # 先 push 后 pull，用 --update 合并双方新增
#   bash scripts/sync_experiments.sh push --full   # 连 training_states/ 和 visualization/ 一起同步
#
# 说明：
# - 只同步产物目录（experiments/results/tb_logger/swanlog），代码与配置走 git；
# - rsync --update 保证双向合并时旧文件不会覆盖新文件；
# - 排除规则只作用于 experiments/（results/ 的 visualization/ 是评估结果本体，会同步）；
# - 实验编号在任一台机器开跑前必须先"占号"并写入 research 日志，避免两边撞名。

set -euo pipefail

REMOTE_HOST="${SYNC_REMOTE_HOST:-basicsr2}"
REMOTE_ROOT="${SYNC_REMOTE_ROOT:-/home/xieyuhang/xyy/Learn/BasicSR}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="${1:-push}"
FULL="${2:-}"

SYNC_DIRS=(experiments results tb_logger swanlog)

EXCLUDES=(
  --exclude 'training_states/'   # 续训状态最大，换机器一般用不到
  --exclude 'visualization/'     # val 可视化图片量大，--full 时才同步
)
if [[ "${FULL}" == "--full" ]]; then
  EXCLUDES=()
fi

# 弱网韧性：断线快速感知 + 断点续传 + 自动重试
RSYNC_SSH="ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=10"
RSYNC_OPTS=(-a --update --partial --append-verify --timeout=300 --info=progress2 -e "${RSYNC_SSH}")
MAX_RETRIES="${SYNC_MAX_RETRIES:-100}"

run_rsync() {  # run_rsync [extra rsync args...] -- <src> <dst>
  local extra=()
  while [[ "$1" != "--" ]]; do extra+=("$1"); shift; done
  shift
  local src="$1" dst="$2" attempt=1
  while true; do
    if rsync "${RSYNC_OPTS[@]}" "${extra[@]}" "${src}" "${dst}"; then
      return 0
    fi
    if (( attempt >= MAX_RETRIES )); then
      echo "rsync failed after ${MAX_RETRIES} attempts: ${src} -> ${dst}" >&2
      return 1
    fi
    echo "rsync interrupted (attempt ${attempt}), retry in 10s..." >&2
    sleep 10
    (( attempt++ ))
  done
}

case "${MODE}" in
  push|pull|both) ;;
  *) echo "unknown mode: ${MODE} (push|pull|both)"; exit 1 ;;
esac

do_push() {
  for d in "${SYNC_DIRS[@]}"; do
    [[ -d "${LOCAL_ROOT}/${d}" ]] || continue
    echo "== push ${d}/ =="
    if [[ "${d}" == "experiments" ]]; then
      run_rsync "${EXCLUDES[@]}" -- "${LOCAL_ROOT}/${d}/" "${REMOTE_HOST}:${REMOTE_ROOT}/${d}/"
    else
      # results/ 的 visualization/ 是评估结果本体，不排除
      run_rsync -- "${LOCAL_ROOT}/${d}/" "${REMOTE_HOST}:${REMOTE_ROOT}/${d}/"
    fi
  done
}

do_pull() {
  for d in "${SYNC_DIRS[@]}"; do
    echo "== pull ${d}/ =="
    if [[ "${d}" == "experiments" ]]; then
      run_rsync "${EXCLUDES[@]}" -- "${REMOTE_HOST}:${REMOTE_ROOT}/${d}/" "${LOCAL_ROOT}/${d}/"
    else
      run_rsync -- "${REMOTE_HOST}:${REMOTE_ROOT}/${d}/" "${LOCAL_ROOT}/${d}/"
    fi
  done
}

[[ "${MODE}" == "push" || "${MODE}" == "both" ]] && do_push
[[ "${MODE}" == "pull" || "${MODE}" == "both" ]] && do_pull
echo "sync done (${MODE}${FULL:+ ${FULL}})."
