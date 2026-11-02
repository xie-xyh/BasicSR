# 002_SDLoRA 研究日志：扩散 LoRA 去雾基线

> 本文件记录 002_SDLoRA（SD2.1-base 冻结主干 + UNet LoRA 单步扩散去雾）从环境搭建、联调到正式训练、评估对比的全过程。
> 每条记录带时间戳，用于追踪研究节奏与决策依据。

## 任务目标

- 验证扩散先验 + LoRA 适配在 OWH **Dense（极浓雾）** 子集上是否优于普通回归网络基线（001 DEANet：Dense-test PSNR 24.04 / SSIM 0.8930），同时保住船舶结构。
- 期望目标：Dense-test 指标达到 moderate 等级（~27 dB）甚至更高。
- 约束：2× RTX 3090 24GB；不清理 GPU1 当前占用（22.7GB，有任务在跑），先用 GPU0 联调；训练时长与 001（约 6.2h）同量级。

## 方案要点（路线 A）

- SD 2.1-base 冻结主干 + UNet LoRA（rank 8，targets: to_q/to_k/to_v/to_out.0）；
- 有雾图 latent 直接作为单步扩散起点（不加随机噪声、固定 t=999，ε-prediction 解析重建 x0）；
- 损失：L1 + VGG 感知损失（纯配对监督，无 VSD/GAN，保证结构保真）；
- 复用框架：PairedImageDataset / L1Loss / PerceptualLoss / PSNR/SSIM / train.py / test.py 全部不动；
- 新增：`basicsr/archs/sdlora_dehaze_arch.py`（SDLoRADehaze）、`basicsr/models/sdlora_model.py`（SDLoRAModel，仅改 mod-8 test 填充 / LoRA-only checkpoint / LoRA-only EMA / 精简网络打印）、`options/train|test/SDLoRA/` 配置。

## 时间线

### 2026-11-02 22:04 — 任务启动

- 建立研究日志目录 `research/002_sdlora/`。
- 环境检查：磁盘剩余 663G（充足）；**huggingface.co 直连不通，hf-mirror.com 可达** → SD 权重下载走镜像（HF_ENDPOINT=https://hf-mirror.com）。
- 开始后台安装依赖：diffusers==0.31.0 / peft==0.13.2 / transformers==4.44.2 / accelerate==0.34.2 / safetensors。
- 计划并行推进：下载 SD2.1-base 权重（~5GB）→ 编写 arch/model/配置 → GPU0 单卡冒烟 → 挂 GPU1 监听脚本。

### 2026-11-02 22:15 — 代码骨架完成 + 一次环境事故

- 已完成（未验证）：`basicsr/archs/sdlora_dehaze_arch.py`（SDLoRADehaze：VAE/UNet 冻结 + LoRA，单步确定性重建）、`basicsr/models/sdlora_model.py`（SDLoRAModel：mod-8 test 填充 / LoRA-only checkpoint / LoRA-only EMA / 日志降噪）、`options/train|test/SDLoRA/` 两份配置、`research/002_sdlora/watch_and_train.sh` GPU 监听启动脚本。
- **事故**：Bash 工具默认 shell 在 miniconda **base** 环境而非 basicsr 环境，第一次 pip install 装错了位置，已终止并改装到 `/home/xieyuhang/miniconda3/envs/basicsr/`。后续所有 python/pip 命令必须用该环境的绝对路径。
- 网络：huggingface.co 不通，权重下载将走 hf-mirror.com。

### 2026-11-02 22:22 — 依赖装好，权重改走 ModelScope

- basicsr 环境依赖安装完成：diffusers 0.31.0 / peft 0.13.2 / transformers 4.44.2 / accelerate 0.34.2；`import basicsr` + 新注册项（SDLoRADehaze / SDLoRAModel）检查通过；pyiqa 导入正常（pip 的 transformers>=5.0 冲突警告实际无影响）。
- hf-mirror 对 SD2.1-base 返回 401（镜像侧问题，非网络问题）→ 改走 ModelScope（`AI-ModelScope/stable-diffusion-2-1-base`，API 与文件直链均 200）。
- 为省带宽下载 fp16 变体（unet 1.73GB + text_encoder 0.68GB + vae 0.17GB ≈ 2.6GB）并改名为标准文件名；arch 中 `torch_dtype=float16` 加载不受影响。下载脚本：`research/002_sdlora/download_sd21base.sh`（带断点续传），后台执行中。
- requirements.txt 已补充 SD-LoRA 依赖段。

### 2026-11-02 22:30 — 冒烟准备就绪，等待权重下载

- VGG19 感知损失权重本地已有缓存，无需下载。
- 冒烟数据集就绪：/tmp/sdlora_smoke（7 对 Dense train + 3 对 Dense test）；冒烟配置 `research/002_sdlora/smoke_sdlora.yml`（单卡、20 iter、关闭 tb/swanlab）。
- diffusers 0.31 API 面确认：UNet2DConditionModel.add_adapter / enable_gradient_checkpointing 可用。
- 权重下载进行中（~2.6GB，ModelScope 直连速度尚可）。

### 2026-11-02 22:38 — 数据管线与模型配置核验

- SD2.1-base 配置核验：UNet cross_attention_dim=1024、use_linear_projection=true（LoRA target to_q/to_k/to_v/to_out.0 为 Linear 层，命名匹配）；scheduler 为 PNDM 配置但 epsilon-prediction/1000 步，DDPMScheduler 加载兼容；VAE scaling_factor 取默认 0.18215。
- 数据管线实测（build_dataset 直测）：Dense train 8140 对、512×512 随机裁剪正常；Dense test 905 张。注意实际图片尺寸不完全一致（见到 800×533 与 800×600），mod-8 填充路径必须可靠。
- UNet 权重仍在下载（速度约 2~3MB/s，剩余 ~1.2GB）。

### 2026-11-02 22:36 — SD2.1-base 权重下载完成

- 全部文件到位且尺寸正确（unet fp16 1.73GB / text_encoder 0.68GB / vae 0.17GB + 全部配置），总耗时约 13 分钟。
- 下一步：GPU0 单元测试（前向/反向/显存/LoRA 梯度隔离）→ 20-iter 冒烟训练。

### 2026-11-02 22:39 — Phase 4 冒烟全部通过（GPU0 单卡）

- 单元测试：512px bs2 前向+反向 0.85s、峰值 8.2GiB；951M 总参数中仅 1.66M LoRA 可训（0.17%），梯度隔离正确；输出无 NaN。
- 修复 1 个 bug：CLIP 文本编码器 fp16 在 CPU 上 LayerNorm 不支持 → 改为 fp32 CPU 编码空 prompt（一次性操作）。
- 冒烟训练 20 iter：dataset/model/loss（L1+VGG）/EMA/验证/LoRA-only 保存全链路通过；约 0.3s/iter@256px。
- checkpoint 验证：net_g_latest.pth 仅 13MB（256 个 LoRA tensor × params/params_ema 双份）。
- test.py 验证：LoRA-only 加载 + 推理 + PSNR 7.0286 与训练期验证完全一致（EMA 链路正确）。注：冒烟仅 20 iter，指标无意义属正常。
- `--auto_resume` 验证：正确定位 10.state 并从 iter 10 续训。
- 冒烟用假数据遗留：experiments/smoke_SDLoRA/（13MB×2，正式训练前清理）。
- 512px 显存余量充足（实测 8.2GiB + VGG 损失 ~3GB ≈ 11GB），bs2/GPU 保守可行。
