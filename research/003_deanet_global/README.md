# 003 — DEA-Net 浓雾增强：低频建模 / 全局上下文改进

> 研究方向讨论与实验记录。目标：改进 DEA-Net 在 OWH Dense（浓雾）子集上的表现，重点关注低频残留雾与全局色调恢复。
> 基线：`001_DEANet_OWH-Mixed-L1_30k_B40G2`（DEANet base_dim=32，Mixed 训练 30k iter，L1 loss，EMA）。

## 1. 基线与现状

| 模型 | 测试集 | PSNR | SSIM | 结果目录 |
|---|---|---|---|---|
| 001 (net_g_30000, params_ema) | OWH-Dense-test (905 张) | 24.0423 | 0.8930 | `results/001_dense_eval/` |

- 001 推理评估已于 2026-11-04 完成（配置 `research/002_sdlora/test_001_dense.yml`），本研究直接复用其输出做失败分析。
- 有雾输入基线（hazy vs GT）mean PSNR 仅 11.44 dB，浓雾退化严重。

## 2. 失败案例分析（2026-11-09）

工具：`tools/rank_failures.py`（逐图 PSNR 排名）、`tools/make_compare.py`（hazy|输出|GT 三联图）。
完整排名：`failure_cases/001_dense_psnr.csv`。

逐图 PSNR 分布：min 11.83 / p5 18.44 / p25 21.85 / median 24.24。最差 5 张（输出 PSNR / 有雾 PSNR）：

| 图 | 输出 PSNR | 有雾 PSNR | 症状 |
|---|---|---|---|
| 01859 | 11.83 | 8.13 | 远景（山体）在有雾输入中**完全不可见**，模型无法复原 → 信息缺失型失败 |
| 06931 | 14.72 | 4.23 | 夜景，**全局色调错误**（GT 深蓝夜空 → 输出蓝灰），片状残雾 |
| 07619 | 13.98 | 6.12 | 船艏区域**成片残雾**，天空整体偏亮偏灰（GT 深蓝） |
| 01589 | 12.64 | 7.55 | 同类：大片均匀区域残雾 |
| 00917 | 15.44 | 9.23 | 同类 |

**诊断结论**：失败由**全局低频**主导——大面积均匀区域的残雾、整体色调/亮度偏差（尤其天空与夜景）。**不是**高频边缘振铃问题。这验证了讨论中的假设：DEA-Net 的 DEConv 是高通滤波器（强化局部梯度），而瓶颈层（1/4 分辨率、8×DEABlock、3×3 卷积）有效感受野有限，缺乏全局上下文来估计平滑的透射率场与全局大气光/色调。边缘问题（深度不连续处的光晕/残雾）大概率是低频估计不准的继发症状。

注意 01859 类（信息完全缺失）属于病态情况，纯复原网络难以解决，评估时应与"低频可恢复但模型没恢复"的案例（06931/07619 类）区分。

## 3. 改进方案（讨论记录，按性价比排序）

- **方案 1（首选）**：瓶颈层插入全局模块——Restormer 式通道自注意力（MDTA，空间复杂度线性）或 Fourier 全局滤波器（GFNet 式，FFT→可学习频域调制→IFFT，1/4 分辨率上开销可忽略）。隔 2 个 DEABlock 插 1 个，其余结构不动。
- **方案 2**：显式低频分支 / 频率解耦——DWT 取 LL 子带（或池化金字塔）走全局路径，与 CGAFusion 融合；DEConv 专心做高频。
- **方案 3（辅助，任何方案都叠加）**：损失侧——FFT 频域 L1 loss 约束全局色调/低频 + Sobel Gradient loss 保边缘（两者均已实现，见下）；必要时 dense 样本过采样。
- **方案 4（大改，备选）**：物理模型引导（显式估计 A/t，AOD-Net 式重构监督）。

**路线**：先方案 3 对照（零结构改动）→ 方案 1（新 arch `deanet_v2` 类名待定为 DEANetV2，注册名不可与 DEANet 重名）→ 视效果决定是否上方案 2。

## 4. 对照实验设计（控制变量）

原则：与 001 基线相比**只改 loss**，数据（Mixed train）、30k iter、优化器/lr/bs、EMA、seed 全部保持一致；结构改动（方案 1/2）必须等 loss 组出结论后再叠加，否则无法归因。

| 实验 | name（拟） | loss 配置 | 状态 |
|---|---|---|---|
| 001（对照组） | `001_DEANet_OWH-Mixed-L1_30k_B40G2` | L1 | ✅ 已完成，Dense 24.04/0.8930 |
| 003a | `003a_DEANet_OWH-Mixed_L1FFT_30k_B40G2` | L1 + FFTLoss(0.1) | 待开训 |
| 003b | `003b_DEANet_OWH-Mixed_L1Grad_30k_B40G2` | L1 + GradientLoss(0.1) | 待开训 |
| 003c（视 3a/3b 结果） | L1 + FFT + Gradient | 待定 |

实现（2026-11-09，已冒烟验证 6 iter 全链路通过）：

- `basicsr/losses/spectral_loss.py` — `FFTLoss`：rfft2（ortho 归一化）频谱实部+虚部的 L1，权重经 `loss_weight` 配置；
- `basicsr/losses/gradient_loss.py` — `GradientLoss`：Sobel x/y 梯度图的 L1（深度可分卷积按通道计算）；
- `basicsr/models/dehaze_model.py` — 新增 `fft_opt` / `gradient_opt` 配置槽（与 `contrastive_opt` 同模式），不配则完全不启用，对旧配置零影响。

loss 量级参考（冒烟，随机噪声数据，weight=1.0）：l_pix ≈ 0.6，l_fft ≈ 0.17，l_grad ≈ 1.7。噪声图的梯度值虚高，真实数据上 l_grad 会小得多；两个新 loss 初始权重暂定 0.1，开训后按 log 中各项占比再调。

评估约定：训练 val 只看趋势；最终对比统一用 `basicsr/test.py` 在 Dense-test（905 张）全量评估（主指标），Mist/Moderate 观察泛化。003a/3b 可分两台机器并行（开训前按 §9 占号 push）。

## 5. 实验记录

| 编号 | 改动 | 配置 | Dense PSNR/SSIM | 备注 |
|---|---|---|---|---|
| 001 | 基线 | `experiments/001_.../train_DEANet_owh_full.yml` | 24.04 / 0.8930 | 已完成 |

## 5. 待办

- [x] 实现 FFT 频域 loss（`basicsr/losses/spectral_loss.py`）+ Gradient loss（`basicsr/losses/gradient_loss.py`），DehazeModel 接 `fft_opt`/`gradient_opt`
- [ ] 写 003a/003b 训练配置（options/train/DEANet/），占号 push 后开训
- [ ] 实现 DEANetV2：瓶颈层插全局注意力块（等 loss 组结论）
- [ ] 区分失败类型：信息缺失型 vs 低频可恢复型，分别统计占比
