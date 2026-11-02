#!/bin/bash
# 从 ModelScope 下载 SD2.1-base 必要文件（fp16 变体改名为标准名）
set -u
BASE_URL='https://modelscope.cn/models/AI-ModelScope/stable-diffusion-2-1-base/resolve/master'
DST='experiments/pretrained_models/sd21base'
mkdir -p "$DST"/{vae,unet,scheduler,text_encoder,tokenizer}

dl() {  # dl <remote_path> <local_path>
    if [ -s "$2" ]; then echo "skip (exists): $2"; return 0; fi
    echo "[$(date '+%F %T')] downloading $1"
    curl -sL --retry 5 --retry-delay 10 -C - -o "$2" "$BASE_URL/$1" \
        && echo "[$(date '+%F %T')] done $2" \
        || echo "[$(date '+%F %T')] FAILED $1"
}

dl model_index.json                                  "$DST/model_index.json"
dl scheduler/scheduler_config.json                   "$DST/scheduler/scheduler_config.json"
dl vae/config.json                                   "$DST/vae/config.json"
dl unet/config.json                                  "$DST/unet/config.json"
dl text_encoder/config.json                          "$DST/text_encoder/config.json"
dl tokenizer/merges.txt                              "$DST/tokenizer/merges.txt"
dl tokenizer/special_tokens_map.json                 "$DST/tokenizer/special_tokens_map.json"
dl tokenizer/tokenizer_config.json                   "$DST/tokenizer/tokenizer_config.json"
dl tokenizer/vocab.json                              "$DST/tokenizer/vocab.json"
dl vae/diffusion_pytorch_model.fp16.safetensors      "$DST/vae/diffusion_pytorch_model.safetensors"
dl text_encoder/model.fp16.safetensors               "$DST/text_encoder/model.safetensors"
dl unet/diffusion_pytorch_model.fp16.safetensors     "$DST/unet/diffusion_pytorch_model.safetensors"

echo "[$(date '+%F %T')] all downloads finished"
ls -laR "$DST"
