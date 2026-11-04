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

### 2026-11-03 10:31 — 训练健康检查通过

- iter 500/15000，0.64s/iter，双卡利用率 100%/85%，l_pix ~0.13-0.19 区间正常（x0 模式起点远低于 t999 的 0.7+）。
- 首次 validation @1000 预计 10:36 出现；下一次 cron 例行检查 10:47。

### 2026-11-03 10:36 — iter 1000 checkpoint 保存，首次 validation 进行中

- lpips 包已补装（最终 LPIPS 对比用）。
- 注意：Dense-test 905 张全量验证每轮约 10 分钟级，训练周期内 val_freq=1000 开销可接受。

### 2026-11-03 10:46 — 首次全量 validation @1000：PSNR 14.13 / SSIM 0.637

- 健康收敛中：l_pix 已降至 ~0.10（iter 1350）。距 001 基线 24.04dB 尚远，但曲线形态正常（对照：001 DEANet @1000 iter 时 PSNR 16.79——回归网络早期更快属预期，扩散 LoRA 后期发力需观察）。
- 发现：全量 905 张验证耗时 ~6.6 分钟/轮，15 轮将占 ~1.5h，ETA 修正为 ~4h。

### 2026-11-03 10:58 — 备好训练后评估工具

- 新增 `research/002_sdlora/eval_tools.py`：`lpips` 子命令（结果目录 vs GT 的 LPIPS 均值，含 001/002 不同命名前缀的配对逻辑）与 `montage` 子命令（hazy | 001 | 002 | GT 四方拼图，用于船舶结构目视检查）。
- 训练照常（下一 cron 检查 11:17）。

### 2026-11-03 11:07 — validation @2000：PSNR 15.50 / SSIM 0.683（+1.37dB/1k iter）

- 曲线健康：14.13 → 15.50。l_pix ~0.14。按当前增速外推收敛点约 20~23dB 区间（存在达不到 001 基线 24.04 的风险，继续观察；若 @5k 增速显著放缓则启动诊断：提高 lr/加大 rank/延长 iter）。

### 2026-11-03 11:20 — validation @3000：PSNR 16.38（增量 +1.37/+0.88，开始减速）

- 14.13 → 15.50 → 16.38。减速符合预期曲线形态；l_pix 降至 ~0.08。l_percep 从 19.8 → 11.2 大幅下降，感知质量在快速对齐。

### 2026-11-03 11:42 — 增速预警：@4000 PSNR 16.73（+0.35/1k，减速明显）

- 曲线：14.13 → 15.50 → 16.38 → 16.73，每 1k 增量 +1.37/+0.88/+0.35，按此外推 15k 收敛点可能仅 ~18-19dB，**达不到 001 基线 24.04 的风险较大**。
- 归因候选（按嫌疑排序）：
  1. **可训容量不足**：rank 8 仅 1.66M 参数（0.17%），且只挂在注意力投影上，卷积主干全冻结——跨域（浓雾海事）适配可能吃不下；
  2. **损失配比**：l_percep（权重 2.0，原始 ~5.7）梯度量级远大于 l_pix（0.09），感知损失主导可能压 PSNR；
  3. lr 1e-4 偏小（OSEDiff 用 5e-5 但有 VSD 引导）。
- 预案：若 @5000 增量 <0.3dB，则停训重启改进版（rank 16 + lr 2e-4 + perceptual_weight 0.5），15k 重跑约 4h。checkpoint 链完整，原 run 可随时 --auto_resume 续训回滚。

### 2026-11-03 11:55 — 切换 002b：rank 16 + lr 2e-4 + perceptual 0.5

- 002（rank8/lr1e-4/percep2.0）@5000 增量仅 +0.17dB（16.73→16.89），触发预案阈值（<0.3dB），停训于 iter ~5300。checkpoint 完整保留在 experiments/002_SDLoRA_OWH-Dense_L1LPIPS_15k_B4G2/，nohup 日志改名 *_nohup_r8_5k3.log，可 --auto_resume 回滚。
- 002b 已启动（`options/train/SDLoRA/train_SDLoRA_owh_dense_r16.yml`）：LoRA rank 16（容量翻倍 3.3M）+ lr 2e-4 + perceptual_weight 0.5（削弱感知损失对 PSNR 的压制）。双卡 ~15GB，0.7s/iter。
- 决策点：002b @5k 与 002 同期（16.89）对比；若仍平台期，下一杠杆是 conv 层也挂 LoRA 或 gt_size 降 256 换 batch/iter 效率。

### 2026-11-03 12:12 — 002b @1000：PSNR 14.35（002 同期 14.13，+0.22）

- 早期仅微弱领先，lr/rank 提升的效果通常在中段才分化，决策点仍为 @5k（002: 16.89）。
- 注意 ETA 升至 ~4.4h（rank16 略慢 + 每轮全量验证 ~7min × 15 轮）。

### 2026-11-03 12:29 — 002b @2000：PSNR 16.37（002 同期 15.50，领先 +0.87dB）

- 改进生效：同期差从 @1k 的 +0.22 扩大到 @2k 的 +0.87。002 的增量序列 +1.37/+0.88/+0.35/+0.17 对比 002b 目前 +2.02（@1k→@2k），形态明显更健康。
- 维持 002b 至 @5k 再评估（002 @5k: 16.89；002b 若按当前形态届时预期 ~19-20）。

### 2026-11-03 12:44 — 002b @3000：PSNR 17.09（002 同期 16.38，+0.71）

- 002b 增量序列：@1k→@2k +2.02，@2k→@3k +0.72；002 同期为 +0.88 → +0.35（失速）。002b 减速更缓，领先稳定在 +0.7~0.9dB。

### 2026-11-03 13:06 — 002b @4000：PSNR 17.42（002 同期 16.73，+0.69）

- 增量 +0.34（002 同期 +0.35 后失速到 +0.17）。关键看 @5k：002b 若增量 ≥0.3 则确认优于 002 的平台形态。

### 2026-11-03 13:16 — 002b @5000：PSNR 17.65 / SSIM 0.720（领先 002 同期 +0.76，但仍将平台）

- 002b 增量 +2.02/+0.72/+0.34/+0.22，与 002 同形态但整体高 ~0.7dB；外推 15k 约 19dB，距 24.04 仍差 ~5dB。rank/lr/损失配比杠杆不够。
- 更深归因：001 DEANet 的优势很可能在**样本吞吐与数据多样性**——DEANet 30k×bs80=240 万样本（Mixed 24k 图），002b 15k×bs4=6 万样本（Dense 6803 图），差 40 倍。
- 预案 002c：Mixed_ge512 过滤集（约 2 万张，含全部雾浓度）+ 其余沿用 002b。先让 002b 跑完拿完整数据点，同时现在就用 CPU 建好 Mixed_ge512。

### 2026-11-03 13:22 — 002c 预案就绪

- Mixed_ge512 过滤集已建：20399/24381 对可用（`datasets/OWH/Mixed_ge512/train/`）。
- 002c 配置草稿（`train_002c_draft.yml`）：Mixed 数据 + 30k iter + 其余沿用 002b（rank16/lr2e-4/percep0.5）。决策：让 002b 跑完 15k（约 16:30）拿完整曲线，若 <22dB 则启动 002c；若 002b 末段余弦降压有惊喜则重估。

### 2026-11-03 13:58 — 002b 终止于 7k，002c 热启动运行中

- 002b 曲线收官：14.35/16.37/17.09/17.42/17.65/17.82/17.98（@1k~@7k），增量衰减到 +0.15/1k，确认平台形态，跑完剩余 8k 预期仅 +1dB，止损切换。
- 002c（`options/train/SDLoRA/train_SDLoRA_owh_mixed_r16.yml`）：Mixed_ge512 20399 对 + 30k iter + rank16/lr2e-4/percep0.5 + **热启动 002b 的 net_g_7000 LoRA**。
- 踩坑：002b 中停导致无 net_g_latest.pth，首次启动 FileNotFoundError；改指 net_g_7000.pth 后成功。SwanLab 云端产生一条 29 记录的空 run（002c 首次启动残留，无碍）。
- 热启动生效证据：002c 初始 l_pix ~0.09（002b 冷启动为 ~0.19）。ETA 日志口径 5.2h，含验证开销实际约 7h（预计 ~21:00 完成）。

### 2026-11-03 14:14 — 002c @1000：PSNR 17.98（= 002b 跑了 7k 的终点！）

- 热启动 + Mixed 数据 + 新余弦高 lr 段效果显著：1k iter 即达 002b 7k 水平（002b @1k 仅 14.35）。
- 按形态推演：002c 从 17.98 起步还有 29k iter（含末段余弦降压），有望冲击 22-24 区间。@3k 数据点（~14:50）进一步确认形态。

### 2026-11-03 14:31 — 002c @2000：PSNR 18.03（增量仅 +0.05，警讯）

- 热启动跳升后立刻回到平台：17.98 → 18.03。说明瓶颈很可能不在数据多样性，而在**容量/目标函数**——rank16 注意力 LoRA 的表达能力已到顶。
- 若 @3k（~14:50）仍 <18.2，启动深度诊断：候选杠杆 ①conv 层也挂 LoRA（容量 ×5~10）②上调 gt_size 有效 batch ③换 Mist/Moderate 观察是否 Dense 信息本身到极限（DEANet Dense 也只有 24）。

### 2026-11-03 19:40 — 时间线更正 + 002c 完整曲线

- 更正：因多次后台 sleep 的实际墙钟远超预期，002c 实际跑到 17:46（iter 13450）才被终止，非此前以为的 3k。
- 002c 完整曲线（Mixed 数据 + 热启动）：17.98@1k → 18.03@2k → 18.18@3k → 18.78@8k → 18.91@9k → 19.01@10k → 19.07@11k → 19.13@12k → 19.19@13k。混合数据 + 长训持续有效但增量衰减至 +0.06/1k，外推 30k 仅 ~20dB。
- **关键排查**：VAE 重建天花板实测 mean 27.77dB（20 张 Dense-test GT，min 21.16）——latent 空间回归不是瓶颈，瓶颈在 UNet LoRA 容量/优化。
- 决策：切换 002d = 002c 配置 + **resnet 卷积挂 LoRA（conv1/conv2/conv_shortcut）** + 热启动 002c net_g_13000（注意力 LoRA 保留 Mixed 学习成果），20k iter。

### 2026-11-03 19:44 — 002d 启动确认

- 可训参数 3.3M → 11.7M（attn r16 3.3M + conv r16 8.4M，容量 ×3.5）；002c@13k 的 256 个注意力 LoRA tensor 双份（params/params_ema）成功热加载；conv LoRA 全新初始化。
- 速度 ~1.0s/iter（conv 反向开销），预计含验证总时长 ~6h（凌晨 ~01:40 完成）。

### 2026-11-03 20:04 — 002d @1000：PSNR 19.24（已超过 002c 跑 13k 的 19.19！）

- **容量假设证实**：conv LoRA 扩容 + 热启动使 1k iter 就超越 002c 全部 13k 训练。曲线起点 19.24，剩余 19k iter + 余弦末段降压，有望冲击 22-24 区间。
- 对照链：002(5k) 16.89 → 002b(7k) 17.98 → 002c(13k) 19.19 → 002d(1k) 19.24。每轮诊断都在收敛。

### 2026-11-03 20:26 — 002d @2000：PSNR 19.30（+0.07/1k，跳升后再入缓增）

- 模式识别：每次"热启动 + 余弦重启"带来一次性跳升（002b→002c: +1.2；002c→002d: +0.05→19.24），随后增量迅速衰减。剩余杠杆：部分解冻（up_blocks 全量微调，LoRA 只表达低秩更新的表达力极限已现）。
- 战略评估：002d 终点预计 ~19.5-20，距 24.04 仍有差距。但本研究的核心问题（扩散 LoRA 是否改善极浓雾+保结构）的判决需要 LPIPS/结构目视，不止 PSNR。002d 跑完后做全量对比（PSNR/SSIM/LPIPS + 拼图），再决定是否上 002e（部分解冻）或输出诚实结论。

### 2026-11-04 10:19 — 会话再次中断后的状态盘点

- 002d 昨夜随会话退出被杀，终止于 iter ~3.5k，最佳 PSNR 19.38@3k（checkpoint/训练状态保留至 3000，可 --auto_resume 续训）。
- 002d 曲线：19.24@1k → 19.30@2k → 19.38@3k（热启动跳升后缓增，+0.07/1k）。
- 002e 代码半成品：arch 已加 `unfreeze_up_blocks` 选项（未提交）；**sdlora_model.py 的 requires_grad 过滤保存/EMA 改造尚未做**——当前保存只含 lora_ 键，解冻参数会丢失，启用 002e 前必须补上。
- 待用户恢复后决策：①--auto_resume 续完 002d（~5h）②直接上 002e（部分解冻）③先对 002d@3k 做完整评估再定。

### 2026-11-04 11:47 — 暂停例行监控 cron

- 目标处于暂停状态、无训练进程，30 分钟例行监控连续空转，已删除（goal 恢复时会重建）。
- 当前断点：002d 终止于 ~3.5k（best 19.38@3k，checkpoint/训练状态完整，可 --auto_resume）；002e（部分解冻 up_blocks）arch 代码就绪未提交，model 保存/EMA 过滤改造待补。

### 2026-11-04 14:16 — 002d@3k 三子集中期评估完成

| 子集 | 002d@3k PSNR/SSIM | 001 DEANet final | 差距 |
|---|---|---|---|
| Dense | 19.38 / 0.737 | 24.04 / 0.893 | -4.66 |
| Mist | 21.98 / 0.779 | 29.71 / 0.967 | -7.73 |
| Moderate | 20.66 / 0.755 | 26.98 / 0.934 | -6.32 |

- 注意 Mist/Moderate 差距更大：002d 权重热启动自 Dense 适配链，且 Mixed 训练仅 13k iter；但 SSIM 普遍低 0.15+ 说明结构保真也尚未对齐，不只是雾去除程度问题。
- 进行中：001 Dense 出图（LPIPS 与拼图对比用）+ 002d Dense LPIPS 计算。

### 2026-11-04 14:30 — LPIPS 对比与中期结论

- Dense-test LPIPS：001 DEANet **0.0677** vs 002d@3k **0.1159**（越低越好）——感知质量也落后，002d 仅 3.5k iter 尚未充分训练是主因之一。
- 结构目视（montage/00007_cmp.png 等 8 组）：002d 船体轮廓/旗帜/上层建筑结构完整无幻觉，"保结构"目标成立；细节锐利度（船名文字级）逊于 001。
- 中期结论：扩散 LoRA 路线在结构保真上验证有效，但保真指标全面落后于专用回归网络。启动最后的大杠杆 002e（up_blocks 全秩解冻，可训 526M）冲刺。

### 2026-11-04 14:36 — 002e 止损，002d 续训至 20k

- 002e（up_blocks 全秩解冻，fp32 UNet）实测 4.8s/iter（ETA 13h+），时间成本不可接受，验证启动无碍后止损。代码（arch unfreeze_up_blocks + model requires_grad 过滤）已提交保留，后续有算力再启用。
- 002d 已 --auto_resume 从 3000 续训（0.75s/iter，ETA ~4h，约 18:30 完成 20k）。
- 监控 cron 已重建（含自动续训与完成后自动评估的指令）。
