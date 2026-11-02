#!/bin/bash
# 002_SDLoRA：监听 GPU 显存，双卡空闲后自动启动 Dense 子集训练
# 用法：nohup bash research/002_sdlora/watch_and_train.sh > research/002_sdlora/watch.log 2>&1 &
set -u
cd /home/xieyuhang/Learn/BasicSR

THRESHOLD_MIB=3000   # 每张卡已用显存低于该值视为空闲
INTERVAL=60          # 轮询间隔（秒）
EXP_NAME=002_SDLoRA_OWH-Dense_L1LPIPS_15k_B4G2
CONFIG=options/train/SDLoRA/train_SDLoRA_owh_dense.yml

echo "[$(date '+%F %T')] watcher started, waiting for all GPUs busy < ${THRESHOLD_MIB} MiB"
while true; do
    ok=1
    used_str=""
    while read -r u; do
        used_str="$used_str $u"
        if [ "$u" -gt "$THRESHOLD_MIB" ]; then ok=0; fi
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if [ "$ok" = "1" ]; then
        echo "[$(date '+%F %T')] all GPUs free (used:${used_str} MiB), launching training"
        break
    fi
    echo "[$(date '+%F %T')] busy (used:${used_str} MiB), sleep ${INTERVAL}s"
    sleep "$INTERVAL"
done

nohup bash scripts/dist_train.sh 2 "$CONFIG" > "experiments/${EXP_NAME}_nohup.log" 2>&1 &
echo "[$(date '+%F %T')] training launched, launcher pid $!, log: experiments/${EXP_NAME}_nohup.log"
