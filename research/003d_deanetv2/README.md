# 003d — DEANetV2：瓶颈频域全局模块（FreqGlobalBlock）

> 设计文档与实验记录。实验名：`003d_DEANetV2_OWH-Mixed512_L1FFTGrad_60k_B16G1`。
> 配置：`options/train/DEANet/train_DEANet_003d_v2_512.yml`；与 003c **除网络结构外完全一致**（同数据、同 512px、同 60k、同 L1+FFT(1.0)+Grad(0.2)、同 lr/EMA/seed），结构是唯一变量，可直接对照 003c。

## 1. 设计动机（基于前序实验证据）

- **失败分析**（research/003_deanet_global README §2）：001 在 Dense 上的失败由全局低频主导——大片均匀区域残雾、整体色调/亮度偏差（06931 夜景、07619 天空），不是高频边缘振铃。
- **003a/003b 对照结论**：补 FFT 频域 loss（约束全局低频）与 Gradient loss（约束边缘）分别带来 Dense +0.12 dB，说明监督信号有效但逼近 loss 侧天花板。
- **结构瓶颈**：DEA-Net 的 DEConv 是差分（高通）卷积，强化局部梯度；瓶颈层（1/4 分辨率、8×DEABlock、3×3 卷积）有效感受野有限，缺乏真正的全局建模能力。**给瓶颈补全局上下文是逻辑上的下一步。**

## 2. 模块设计：FreqGlobalBlock（FGB）

参考近期频域全局建模工作：[GFNet](https://pubmed.ncbi.nlm.nih.gov/37030707/)（NeurIPS 2021，可学习频域全局滤波器）与 [SFHformer](https://github.com/deng-ai-lab/SFHformer)（ECCV 2024，FFT 与 Transformer 结合做图像复原；频域建模低频退化、空域建模局部细节的思路与去雾"雾=低频+细节=高频"的分解天然契合）。

结构（`basicsr/archs/deanet_v2_arch.py::FreqGlobalBlock`）：

```
x → GroupNorm → rFFT2 → × 可学习复数滤波器 W(u,v) → irFFT2 → + x（残差）
  → GroupNorm → FFN(1×1 conv → GELU → 1×1 conv) → +（残差）
```

设计要点：

- **真正的全局感受野**：频域乘法的每次操作都耦合所有空间位置，一次调用即覆盖整图——正是平滑透射率场/大气光这类全局低频量所需要的；
- **零初始化**：滤波器初始为 0，模块初始近似恒等映射，插入既有堆叠不扰动训练初期动态；
- **任意分辨率可用**：滤波器按规范尺寸存储（32×32），前向时双线性插值到当前频谱形状——训练 512px crop 与全尺寸测试图（533×800 等）都能工作（DEANet 的 mod-4 padding 由 DehazeModel.test 处理，与 v1 一致）；
- **代价极小**：瓶颈处（128×128，dim 128）FFT 开销可忽略，两个 FGB 仅 +0.41M 参数（7.79M→8.20M，+5%）。

## 3. 插入位置与整体结构

DEANetV2 = DEANet 原结构，仅在最深层 level3 的 8 个 DEABlock 之间插入 2 个 FGB：

```
fe_level_3 → DEA → DEA → FGB → DEA → DEA → FGB → DEA → DEA → mix1（CGAFusion）
```

其余（encoder/decoder/跳跃连接/CGAFusion）完全不动，保证与 003c 的对照只体现"频域全局模块"一个变量。

## 4. 验证记录

- 单元测试：512×512 与 536×800（=533×800 pad 到 4 的倍数）前向+反向通过；参数量 7.79M→8.20M。
- 冒烟训练（60 iter，512px 真实数据，L1+FFT+Grad 全配置）：训练正常，loss 正常下降。
- 显存探测（512px 单卡）：bs16 峰值 **39.3 GiB**；bs20 OOM（FGB 的 FFT 中间张量显存开销高于纯卷积）。**003d 取 bs=16**（相应 lr 1e-4 与 003c 保持一致；有效 batch 差异 16 vs 20 属已知小偏差，记录在案）。

## 5. 实验记录

| 时间 | 事件 |
|---|---|
| 2026-11-10 14:10 | 设计与实现完成，单元+冒烟通过，启动待安排 |

（训练启动后的进度记录在本文件持续追加。）
