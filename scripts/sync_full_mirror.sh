#!/usr/bin/env bash
# 一次性全量镜像同步：本仓库整个目录（含 .git、experiments/results/tb_logger/swanlog 全量）
# 本机 -> basicsr2:/home/xieyuhang/xyy/Learn/BasicSR/
# 断线自动重试 + 断点续传；日志见 research/sync_full_20261114.log
set -uo pipefail

LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="basicsr2:/home/xieyuhang/Learn/BasicSR/"

attempt=1
max=1000
cd "${LOCAL_ROOT}"
while true; do
  if rsync -a --partial --append-verify --timeout=300 --info=progress2 "$@" \
      -e "ssh -o ServerAliveInterval=15 -o ServerAliveCountMax=10" \
      ./ "${REMOTE}"; then
    echo "RSYNC_ALL_DONE"
    exit 0
  fi
  if (( attempt >= max )); then
    echo "RSYNC_FAILED after ${max} attempts" >&2
    exit 1
  fi
  echo "rsync interrupted (attempt ${attempt}), retry in 10s..." >&2
  sleep 10
  (( attempt++ ))
done
