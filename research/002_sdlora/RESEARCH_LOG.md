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

### 2026-11-02 22:42 — 冒烟遗留清理 + 监听脚本已挂起

- 5 次 git 提交完成（arch / model / options / requirements / research 分模块，75411c2 为最新）。
- `watch_and_train.sh` 已后台运行（pid 324715）：每 60s 轮询，双卡已用显存均 < 3GiB 时自动 `dist_train.sh 2` 启动正式训练（当前 GPU1 被占 22.7GB，等待中）。
- 已挂每 30 分钟例行监控（cron）：自动检查 watch.log / 训练日志、提取指标、异常诊断、完成后自动跑三子集评估并对比 001 基线。
- 正式训练启动前一切就绪；预估 512px bs4(2×2卡) 约 1.2~1.5s/iter，15k iter ≈ 5~6h。

### 2026-11-02 22:52 — 健壮性预跑启动 + watcher 修正

- 修正：watcher 首次启动时 stdout 未落盘 watch.log（被后台任务捕获），已重启并正确重定向（pid 328067）。
- GPU0 空闲，借机跑 300-iter 真实数据健壮性预跑（sanity_SDLoRA_dense：512px + VGG 感知损失 + AdamW + cosine，val 用小样本 3 张），目的：在正式训练前暴露 512px 全配置下的稳定性问题（loss 发散/NaN/显存）。
- 预跑期间 GPU0 被占，watcher 会等其结束再启动正式训练（符合预期）。

### 2026-11-02 23:00 — 预跑抓到真实 bug：Dense 训练集尺寸不齐

- 300-iter 预跑 v1 秒挂：`LQ (451, 800) is smaller than patch size (512, 512)`。扫描全部 8140 张 Dense train：尺寸高度异质，最小 297×265；512/448/384/256 可用率分别 83.6%/94.5%/98.6%/100%（001 DEANet 用 gt_size 256 所以从未触发）。
- 决策：保持 SD 原生 512px 训练分辨率（先验利用最大化），新增 `scripts/data_preparation/filter_min_size.py` 生成 `datasets/OWH/Dense_ge512/train/`（6803 对软链接，丢弃 1337 对小图），训练配置 dataroot 已切换。
- 预跑 v2（ge512 过滤集）进行中。

### 2026-11-02 23:12 — 公式化风险排查：三探针对比实验

- 预跑 v2 技术上通过（300 iter 无崩溃，0.57s/iter@512px bs2 单卡），但暴露隐患：loss 无趋势、val PSNR 卡在 7.0，可视化输出几乎全白。
- 机理分析：t=999 时 ᾱ≈0.0047，x0=(z-√(1-ᾱ)ε)/√ᾱ 把 UNet 输出放大 ~14.6 倍，纯 L1/LPIPS 监督下 LoRA 需要先学会"ε≈z"才能走出灾难区，收敛可能很慢。
- 处置：暂停 watcher（避免用未验证配置启动 6h 正式训练）；arch 新增 `prediction: eps|x0` 选项（x0=UNet 直接预测清晰 latent）。
- 三探针并发/顺序对比（384px、2k iter、单卡）：A) t999+eps（原方案）；B) t250+eps（放大倍数 ~1.8，温和）；C) x0 直出。以 loss 下降趋势 + 2k iter 时 val PSNR 定胜负。

### 2026-11-02 23:47 — 探针 A/B 收官：t250 碾压 t999

| 探针 | l_pix (100→2000 iter) | val PSNR @2k | 趋势 |
|---|---|---|---|
| A: t999+eps | 0.77 → 0.44（徘徊） | 6.32（比 @1k 的 6.90 还差） | 未收敛 |
| B: t250+eps | 0.27 → 0.16（持续降） | 11.41，SSIM 0.627，Best 在最后 | 健康收敛中 |

- 结论 1：t=999 的 ~14.6× 放大确实让纯像素监督失效，弃用。
- 结论 2：t=250 收敛正常但 2k iter 仅 11.4dB——LoRA 从 0 适配仍需量级更大的迭代数，正式训练 15k 预期能到可用区间。
- 探针 C（x0 直出）已启动，若显著优于 t250 则采用 x0（牺牲扩散 ODE 解释换收敛速度）。

### 2026-11-02 23:50 — t250 可视化确认：船体结构保住

- 探针 B @2k iter 输出图（experiments/probe_t250/visualization/）：浓雾船的船体、上层建筑、旗帜清晰可见，结构无扭曲、无幻觉——"保船舶结构"目标的技术路线（单步确定性 + LoRA 冻结主干 + L1 为主）初步验证有效。

### 2026-11-03 09:59 — 会话中断后的状态盘点

- 昨夜会话退出导致两个后台任务失联：watcher（当时处于我主动暂停状态，未重启）与探针 C（probe_x0 跑到 ~200 iter 被杀）。
- **双卡现已空闲**（GPU0 390MiB / GPU1 15MiB），但正式训练未启动——watcher 未在运行。
- 探针 C 残片数据：x0 直出 iter 100/200 的 l_pix = 0.152/0.126，同期比 t250（0.267@100）低得多，早期信号显示 x0 收敛最快（待补跑确认）。
- 待办：补跑 probe_x0 → 定稿公式化 → 更新正式配置 → 重启 watcher 或直接启动正式训练（GPU 已空，可直接启动双卡训练）。
- 注：goal 当前处于 paused 状态，等用户恢复后继续。

### 2026-11-03 10:10 — 会话恢复，探针 C 补跑中

- 已恢复 goal；/tmp 被系统清空导致冒烟验证集丢失，已迁移到 research/002_sdlora/smoke_val/（持久），相关 yml 路径已更新。
- 探针 C（x0）重跑中，iter 100/200 l_pix=0.152/0.126，与昨夜被杀前完全一致（种子确定性可复现）。
- AGENTS.md 已更新（目录树 + SD-LoRA 开发注意事项 + /tmp 易失警告），配置 dataroot 切换 Dense_ge512 已提交（a9662ce、6112be2）。

### 2026-11-03 10:27 — 正式训练启动（002_SDLoRA_OWH-Dense_L1LPIPS_15k_B4G2）

- 探针终局：x0 直出 2k iter val PSNR 13.78 / SSIM 0.637（t250: 11.41/0.627；t999: 6.32 弃用），可视化确认 x0 输出船体锐利、结构无幻觉 → **正式采用 prediction: x0**。
- 踩坑：swanlab 云端已要求 SDK>=0.9.0，env 里 0.8.4 的 online init 被拒导致双卡启动即崩；升级到 0.10.3 并验证 API 兼容后重启成功。
- 双卡正式训练运行中：512px、bs 2×2、~0.68s/iter、ETA 约 2.5h、SwanLab online（project: dehaze）。两卡各占 ~14.7GB。
- 监控：每 30 分钟 cron 例行检查；训练完成后自动跑三子集 test.py 评估对比 001。
