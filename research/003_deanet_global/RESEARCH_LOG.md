# 003_DEA-Net 研究日志：浓雾低频/全局上下文改进（第一阶段：loss 对照）

> 本文件记录 003 系列（DEA-Net 改进，面向 OWH Dense 浓雾）从讨论、失败分析到对照实验训练的全过程。
> 每条记录带时间戳。背景与方案详见同目录 README.md。

## 任务目标

- 改进 DEA-Net 在 OWH **Dense（浓雾）** 子集上的表现（001 基线：Dense-test PSNR 24.04 / SSIM 0.8930，失败由全局低频主导——大片残雾与整体色调偏差，见 README §2）。
- 第一阶段（本日志）：控制变量的 loss 对照实验——003a（L1+FFTLoss）、003b（L1+GradientLoss）vs 001（纯 L1）。
- 约束：本机 2× RTX 4090 48G，每卡单跑一个实验；与 001 保持有效 batch 80 / lr 2.5e-4 / 30k iter / Mixed 数据 / EMA / seed 完全一致，唯一变量为 loss。

## 时间线

### 2026-11-09 19:20 — 失败案例分析（前置诊断）

- 复用 001 的 Dense-test 全量评估输出（905 张），逐图 PSNR 排名（`tools/rank_failures.py`）：min 11.83 / p5 18.44 / median 24.24。
- 最差样本三联图目视（`failure_cases/*_hazy_out_gt.jpg`）：06931 夜景全局色调错误（深蓝→蓝灰）、07619 船艏成片残雾+天空偏亮、01859 远景信息完全缺失（病态）。
- 结论：失败由全局低频主导，支持"瓶颈层加全局上下文 + 频域损失"的改进方向。

### 2026-11-09 19:35 — loss 实现与单元冒烟

- 新增 `basicsr/losses/spectral_loss.py`（FFTLoss：rfft2 实虚部 L1）、`basicsr/losses/gradient_loss.py`（GradientLoss：Sobel x/y 梯度 L1）；`DehazeModel` 新增 `fft_opt`/`gradient_opt` 配置槽，不配则不启用（对旧配置零影响）。
- 合成数据 6 iter 冒烟通过（训练+验证+保存）。

### 2026-11-09 19:47 — 单卡可行性探测（双 4090 48G）

- bs=40 双卡并发各跑一个实验：峰值 24.8 GiB/卡、利用率 99%、0.60 s/iter，互不拖慢 → 每卡单跑可行。
- bs=80 单卡：峰值 46.8 GiB、1.14 s/iter（每样本耗时仅省 ~5%，已 compute-bound）。
- 决策：正式实验用 **bs=80 单卡**——不为提速，而是让有效 batch（80）与 lr（2.5e-4）和 001（40×2 卡）完全一致，保证对照干净。
- 真实数据 loss 量级：l_pix ≈ 0.16，FFTLoss 原始 ≈ 0.03（取 weight 0.5 → 占比 ~9%），GradientLoss 原始 ≈ 0.21（取 weight 0.1 → 占比 ~13%）。

### 2026-11-09 19:55 — 正式配置 + 预检启动

- 正式配置：`options/train/DEANet/train_DEANet_003a_l1fft.yml`（003a，FFT weight 0.5）、`train_DEANet_003b_l1grad.yml`（003b，Gradient weight 0.1）；其余字段与 001 逐项对齐。
- 预检（双卡并发，`--force_yml name=preflight_003a/003b train:total_iter=100 val:val_freq=100`）：训练段正常，100 iter × ~1.2 s（双卡并发略慢于独占的 1.14 s），l_pix/l_fft/l_grad 数值正常无 NaN；正在跑三数据集验证（2711 张/实验）。
