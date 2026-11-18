# 011 WaveletUNet：encoder-decoder + 小波本质分析（011a）

> 设计文档。前置阅读：`research/010_wavelet/README.md`（010a/010b 的失败诊断：表达力封顶 + 容量不足）、`research/008_dwt/README.md`（Haar 工具）。
> 本文件含严格理论推导（§2），是用户要求的"理论分析严谨"的核心交付。

## 1. 设计定位

010a/010b 的失败证明：单趟子带处理器（无编解码、无生成能力、容量 0.136M）不足以承担去雾任务。011a 的设计原则：

- **真四级 encoder-decoder**：容量与多尺度特征提取由 U-Net 提供，不再是单趟子带处理；
- **小波是表示语言而非装饰**：Haar DWT/IWT 作为无损重采样贯穿采样路径；输出头在**图像空间**做物理分解（底图 + 倍率 + 自由修正）；
- **不绑住生成的手**：倍率负责"存活细节的物理放大"，自由残差负责"已消灭细节的生成"（010a 的 D≈0 教训）；
- **理论推导严格**：§2 的每一条都有明确成立条件与适用范围，并标明图像空间严格、特征空间为启发式扩展。

## 2. 理论推导（严格）

### D1. Haar 正交性与无损性

Haar 滤波器组 {ll, lh, hl, hh} 在每个 2×2 块上构成 R² 的正交归一基（基向量两两正交、范数为 1）。记 DWT 的线性算子为 W，则 W^T W = I，故：

- **无损性**：IWT(DWT(x)) = x（单测重建误差 ~2.4e-7，float 精度极限）；
- **Parseval**：||x||² = ||Wx||²——残差能量可按子带精确分解，总 MSE = 各子带 MSE 之和，无交叉项。

### D2. 散射模型的小波域形式

对 I = J·t + A(1−t) 作用 W（线性）：

```
W(I) = W(J·t) + A·W(1−t)
```

常数场的细节子带为零（W_detail(1) = 0），故 W_detail(A·(1−t)) = −A·W_detail(t)。代入得：

```
W_detail(I) = W_detail(J·t) − A·W_detail(t)        ……(1)
```

### D3. 局部恒定 t 下的乘性恢复（严格）

Haar 细节系数是局部差分。设 t 在 Haar 支撑集内恒定，则 W_detail(J·t) = t·W_detail(J)，且 W_detail(t) = 0。代入 (1)：

```
W_detail(I) = t · W_detail(J)
⇒ W_detail(J) = (1/t) · W_detail(I)                ……(2)
```

**倍率 = 1/t，严格成立**。这给出倍率分支的物理合法性：网络只需估计一个标量场 1/t，不必生成纹理。

### D4. t 变化时的修正项（有限差分乘积法则）

t 在支撑集内变化时，有限差分乘积法则：

```
Δ(J·t) = t(x+1)·ΔJ + J(x)·Δt
```

代入 (1) 并解出 W_detail(J)：

```
W_detail(J) = (1/t)·W_detail(I) − ((J−A)/t)·W_detail(t)   ……(3)
```

第二项与 **t 的空间梯度**和**局部图像内容 (J−A)** 成正比——这是自由修正通道的理论位置：它必须能表达任意方向的边缘修正，**不能限定为小量**（010a 的 corr_scale=0.1 正是错在这里）。

### D5. LL/雾幕项需要全局上下文

W_LL(I) = W_LL(J·t) + A·W_LL(1−t)。A(1−t) 是空间平滑的全局场 ⇒ 其估计需要严格全局的感受野。瓶颈的金字塔池化（1×1/4×4/8×8 网格）提供了多尺度全局上下文——这是对"局部感受野修全局雾幕"这一失败模式的直接修正。

### D6. 子带监督与端到端损失的关系

由 Parseval（D1），||out−GT||² = Σ_subbands ||·||²。子带分解监督与端到端 MSE 在能量上可分解、不冲突；本项目使用 L1（不满足 Parseval 的精确分解，但方向一致），其作用是为各预测目标提供独立、干净的梯度信号（010a 实测：纯 RGB 损失下纹理分支梯度比底图小 400 倍）。

### 适用范围声明

- D2~D4 在**图像空间**严格成立（输出头所在位置）；
- 网络内部特征空间的 Haar 采样是**同构启发式扩展**（特征子带不再直接对应物理量），其价值是无损重采样与显式频带保留，不要求逐层满足物理推导。

## 3. 架构（`basicsr/archs/wavelet_unet_arch.py`）

```
有雾 RGB
  └─ HaarDWT₁ → LL1（3ch, 1/2 res）+ det1（9ch, 1/2 res）

Encoder（在 LL1 上）：
  enc1: 3→32 + 2×ResBlock → WaveletDown(32→64)
  enc2: 2×ResBlock(64)    → WaveletDown(64→128)
  enc3: 2×ResBlock(128)   → WaveletDown(128→256)
  bottleneck: ResBlock(256) + PyramidPool(1/4/8 网格) + ResBlock(256)

Decoder（镜像，跳跃 concat + 1×1 投影 + 2×ResBlock）：
  up3→dec3(128) → up2→dec2(64) → up1→dec1(32)

输出头（图像空间，1/2 res）：
  ll1_rec = ll1 + ll1_head(d1)
  gain = exp(avgpool₃ₓ₃(clamp(log_gain, −1, 2)))     # 有界 [0.37, 7.39]
  free = free_head(cat[d1, det1])                    # 自由生成，不加小系数
  det1_rec = det1 × gain + free
  aux = (ll1_rec, det1_rec)

输出 = HaarIWT₁(pack(ll1_rec, det1_rec))  → RGB
```

- 参数量 **4.692M**（vs DEA-Net 7.88M）；bs20 @512px 实测 **8.4 GiB / 0.16 s/iter** → 60k ETA ~3h；
- 奇数尺寸：HaarDWT 奇数 pad-1 + `_match_size` 对齐 skip（512/536×800/127×255 已验证）；
- 与 010a 的区别：有真编解码容量、有自由生成通道、瓶颈有严格全局上下文、监督在图像空间子带上。

## 4. 训练配置（`options/train/WaveletDehaze/train_WaveletUNet_011a.yml`）

与 004a 一致：数据（Mixed 过滤 20399 对、512px、bs20）、loss（L1 1.0 + FFT 1.0 + Grad 0.2 + subband LL1/detail 各 1.0）、EMA 0.999、seed 0、`periods=[60000]`、val_freq 2000、save 1e3；lr 2e-4 + warmup 500 沿用 010b。

## 5. 判据与止损

1. 主指标：Dense 终值目标 24.5+（004a 为 24.86）、Mist 目标 33+（004a 为 30.98）；
2. 机理：残差频谱 DC~0.02 与 >0.50 应同时改善；倍率图与雾分布空间对应（`last_gain` 可导出）；
3. 止损：@20000 Dense <20 且 Mist <26 → 方向封顶，回 DEA-Net 主干容量扩展。

## 6. 收尾评估协议

1. 全量评估：`python basicsr/test.py -opt research/011_wavelet_unet/test_011a_dense.yml`（三子集、params_ema、net_g_60000）；
2. 频谱：复制 `research/008_dwt/tools/residual_spectrum_008.py`，MODELS 精简为 hazy/003c/004a@60k/011a，OUT_DIR 改 `research/011_wavelet_unet`；
3. 倍率图：导出 `last_gain`（注意 img2tensor 不归一化，须 /255）；
4. 结论写入 §8，中文 commit（不 push）。

## 7. 启动清单（新会话）

1. `nvidia-smi` 选空闲卡；编号 011a 已占号；
2. 显存已实测（bs20 8.4 GiB），无需再探测；
3. 启动：
   ```bash
   CUDA_VISIBLE_DEVICES=<gpu> HF_ENDPOINT=https://hf-mirror.com nohup python basicsr/train.py \
       -opt options/train/WaveletDehaze/train_WaveletUNet_011a.yml > experiments/011a_nohup.log 2>&1 &
   ```
4. 监控：`python research/006_hihaze/tools/monitor_006a.py --log experiments/011a_nohup.log --csv research/011_wavelet_unet/metrics_011a.csv --total 60000 --tag 011a`；
5. 完成后按 §6 收尾。

## 8. 结果与结论

（待训练完成后填写）

## 9. 后续方向

- 有效 → LoRA 真实域微调（见会话讨论：LoRA 在此架构上比 DEA-Net 更有结构性优势——固定小波变换保证物理结构稳定，gain_net/base_branch 可精准投放）；
- 欠拟合 → 加宽 widths（64/128/256/512）或加深 ResBlock 数；
- 无效 → 小波方向整体封档，回 DEA-Net 主干容量扩展（level1/2 加块）。

## 10.5 实验记录

| 时间 | 事件 |
|---|---|
| 2026-11-18 17:16 | 011a 启动（GPU0 pid 2996118，日志 experiments/011a_nohup.log）。iter 400：0.172 s/iter（ETA ~2h50m）、l_pix 0.133/l_base 0.263/l_detail 0.010（从头训练正常）；监控 cron 每 30 分钟巡检（含全量评估/频谱/倍率图自动收尾） |
| 2026-11-18 17:26 | @2000：Dense 13.26/0.7310、Mist 20.05/0.8979（与 010a 同期同量级，从头训练正常起点；l_pix 已降至 0.106，比 010a 同期快） |
| 2026-11-18 17:54 | @4000/@6000/@8000：Dense 16.83/18.55/**19.87**、Mist 22.34/23.65/**24.45**、Moderate 19.42/21.12/**22.28**（@8000）。**收敛速度显著好于 010a**（010a @4000 仅 14.98，且已近停滞；011a @8000 已达 19.87 且仍在快速上升）。l_pix 0.068、l_base 0.134——自由生成通道 + 真编解码容量起效 |

## 10. 验证记录

- 2026-11-18 单元测试：512²/536×800/127×255/128² fwd+bwd 通过（含奇数尺寸 `_match_size` 修复）；参数量 4.692M；
- 2026-11-18 显存/速度探测：bs20 @512px 峰值 8.4 GiB、0.16 s/iter；
- 2026-11-18 冒烟训练（`smoke_011a.yml`，6 iter）：全链路通过，子带监督正常计算，产物已清理。
