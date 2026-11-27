# 015 — 浓雾上限突破与面向真实海雾 LoRA 的并行验证计划

> 设计文档（2026-11-28）。
> 前置依据：`research/014_maritime_sfnet/README.md`（`014d` 8k 全面反超 `011p`：Dense **21.9290 / 0.8894**、Mist **26.7390 / 0.9594**、Moderate **24.9567 / 0.9294**，正式 100k 主干 `014d_MaritimeSFNet_OWHscreened_100k_B16G1` 正在 `hust` GPU1 训练中）、`research/005_sim2real/README.md`（真实海雾偏蓝大气光与 3.8× 通道离散度）、`research/011_wavelet_unet/NEXT_RESEARCH.md`（`011q` `BandBalancedFFTLoss` 全三子集正增益 + Dense 48% 局部块 $t<0.1$）。
>
> **一句话定位**：趁 `hust` 训练 `014d` 100k 标准主干的同时，利用 `d533` 双 RTX 3090 空闲算力开展 **2×2 正交并行探针（`015a` 架构侧浓雾上限 vs `015b` 海事物理训练配方侧浓雾上限）**，并与 `014d@8k` 严格对标，为后续合体（`015c`）、物理域随机化继续预训练（`015_PhysPT`）与真实海雾 LoRA（`015_PhysLoRA`）提供因果实验依据。

---

## 1. 并行探针设计（`d533` 双 3090 正交双臂，严格对标 `014d@8k`）

全部沿用 `v2` 冻结验证清单（内部 283 图：Dense 94 / Mist 94 / Moderate 95）、`manual_seed: 0`、`gt_size: 512`、有效 `batch_size: 16`、Adam (`lr=2e-4`, `betas=[0.9, 0.99]`)、warmup `500`、60k 余弦调度前 `8,000` iter、EMA `0.999`、`val_freq: 2000`、`save_checkpoint_freq: 1000`（强制落盘断点防故障）、SwanLab 云端项目 `dehaze-probes`（经 `start_swanlab_bridge.sh` 隔离同步）。

| 探针臂 | 运行设备 | 架构变量 | 数据 / 损失变量 | 核心科学问题 |
|---|---|---|---|---|
| 参照基线 `014d@8k` | `d533`（已完成） | `MaritimeSFNet` (`3,3,8`, `basr_g_max=4.0`) | `v2` 标准清单 + `use_rot=true` + `L1+FFT+Grad` | Dense **21.9290 / 0.8894**、Mist **26.7390 / 0.9594**、Moderate **24.9567 / 0.9294** |
| **`015a` (`probe_015a_axial_basr`)** | `d533` **GPU 0** | `014d` + **`AxialGlobalProfile`**（100% 全图宽/高 1D 轴向全局调制）+ **`basr_g_max=6.0`** + **`basr_refine=true`**（增益门控浓雾精修头） | **与 `014d` 100% 严格一致** | 严格同 `v2` 协议下，100% 轴向全局上下文 + 放宽 $C^2$ 浓雾增益 + 高增益雾噪精修能否净推高浓雾架构上限？ |
| **`015b` (`probe_015b_maritime_recipe`)** | `d533` **GPU 1** | **与 `014d` 100% 严格一致** | **`use_rot=false`**（保海事天空→海天线→海面垂直梯度先验）+ **`BandBalancedFFTLoss`**（`011q` 实证全三子集增益）+ **`train_manifest_dense322.txt`**（Dense:Mod:Mist = 3:2:2 难样本聚焦） | 固定 `014d` 架构不变，修复旋转破坏海事各向异性 + 低频均衡频域监督 + 浓雾聚焦采样能带来多大浓雾提分且不伤轻雾？ |

### `015a` 两项架构升级的数学与兼容性约束

1. **`AxialGlobalProfile`（`use_axial_global: true`, `axial_kernel: 7`）**：
   - 将 P3 的有限条形大核（`31×4 = 124 px`）推广至 **100% 全图宽 & 全图高**，同时避开 `RegionContext` / `MDTA` 的无序空间混合暗斑；
   - 行分支聚合每行全宽 `mean + amax` 经 1D 深度卷积沿高度 $y$ 建模“天空→海天线→海面”全局雾度剖面；列分支聚合每列全高 `mean + amax` 沿宽度 $x$ 建模横向雾团与船体分布；
   - 无归一化（保 DC）、乘性增益严格有界于 $2\sigma(\cdot) \in (0, 2)$、末层严格零初始化（初始恒等误差 `0.00e+00`），并自动注册至 `iter_lora_target_modules()['axial_global']` 供后续真实海雾 LoRA 挂载。
2. **放宽 $C^2$ BASR 浓雾上限（`basr_g_max: 6.0`）+ 增益门控浓雾精修（`basr_refine: true`）**：
   - $C^2$ 零曲率对称 `tanh` 在原点满足 $G'(0)=1, G''(0)=0$（与 $g_{\max}$ 无关），将 `basr_g_max` 从 `4.0` 放宽至 `6.0` 可覆盖更深浓雾（$t \ge 0.167$）而不引入轻雾 Jensen 过冲；
   - 在仿射初恢复 $J_{\text{affine}} = G \odot I + B$ 后引入受浓雾增益门控 $w_{\text{dense}} = \sigma(2(\bar{G} - 1.5))$ 调节的轻量 2 层零初始化残差精修，定向抑制高增益浓雾区的 8-bit 量化台阶与雾-噪耦合伪影（呼应 HNDiff ECCV 2026）；
   - 新增标量诊断 `diag_basr_gain_p95`（95 分位增益）与 `diag_basr_refine` 同步进入 TensorBoard / SwanLab；
   - **向后兼容硬要求**：全部新开关默认 `False`，`014a–014e` 及正在 `hust` 上训练的 `014d` 100k 权重与前向输出逐比特不变。

### `015b` 海事浓雾训练配方的三项确定性改动

1. **`use_rot: false`（保留 `use_hflip: true`）**：消除 `transforms.py` 50% 上下翻转与 50% 90° 转置对 P3 水平/垂直海事各向异性的破坏；
2. **`BandBalancedFFTLoss(loss_weight=1.0, low_weight=0.00607, cutoff=0.02)`**：直接复用仓库已校准并在 `011q` 验证全三子集正收益（+0.172 / +0.297 / +0.227 dB）的低频均衡频域损失；
3. **`train_manifest_dense322.txt`**：由冻结 `v2` 训练清单确定性生成（全部 Dense ×3、Moderate ×2、Mist ×2，占比 Dense 43.1% / Moderate 28.9% / Mist 28.0%，无随机丢图、无测试泄漏）。

---

## 2. 验收判据与后续路线（衔接 `014d` 100k 与真实海雾 LoRA）

1. **8k 探针验收（与 `014d@8k` 21.929 / 26.739 / 24.957 逐图配对对比）**：
   - `015a` 与 `015b` 均需通过 `paired_eval.py` 283 图配对评估 + 频段/过冲/欠修归因；
   - 若单臂 Dense ≥ +0.2 dB 且 Mist/Moderate 不退化 → 组合为 `015c`（架构+配方合体）验证叠加增益。
2. **Stage 3（待 `hust` 上 `014d` 100k 完训）**：
   - 零重训测试 **TLC 局部统计推理** + 官方 2,711 图终评 + `real-test`（500 张）6 项无参考基线；
   - 启动 **`015_PhysPT`（海事物理域随机化继续预训练，20k–30k）**：50% 原 OWH + 50% 按 `005` 实测偏蓝大气光/3.8× 通道离散度/非均匀雾团/微噪在线重雾化 + 双视角浅层特征一致性正则，打造抗编码器捷径塌缩的黄金预训练基座。
3. **Stage 4（`015_PhysLoRA` 真实海雾 LoRA 适配）**：
   - 折叠冻结 `015_PhysPT`（对照纯 `014d` 100k）主干；在 `datasets/OWI/`（1,858 train / 400 val）上结合 **BASR 真雾物理迁移伪配对（PHATNet 思路）+ 海事 CLIP H2C 文本方向损失（BiLaLoRA 思路）+ `ChromaPreserveLoss`（005d）+ 合成回放（CoA 防遗忘）**，与 FreqDA / 均匀 ConvLoRA / BiLaLoRA 自动选层做完整顶会消融。

---

## 3. 预检实测与实验记录

### 3.1 `d533`（2×RTX 3090 24 GB）单元测试、冒烟与真实批次显存预检（2026-11-28）

- **向后兼容校验**：`verify_014.py` 全部 5 组配置（`014a–014e`）通过，`Strict load of live 014d EMA checkpoint PASSED`；
- **单元测试与 3 步冒烟训练**：`verify_015.py --smoke` 全部通过（`init_id = 0.00e+00`，`fold_err = 0.00e+00`，`dense322` 清单 55,314 行精确覆盖 23,697 对唯一 `v2` 训练图）；
- **单卡 RTX 3090 实测（`bs=16 @ 512×512`，`activation_checkpointing: true`）**：

| 探针臂 | 目标卡位 | 训练 / 折叠参数 | 峰值分配 / 预留显存 | 单步耗时 | 预计 8k 总耗时 |
|---|---|---:|---:|---:|---:|
| **`015a` (`probe_015a_axial_basr`)** | `d533` GPU 0 | 7.931M / 4.648M | 13.61 / **18.09 GiB** | 2.41 s/it | ~5.4 h |
| **`015b` (`probe_015b_maritime_recipe`)** | `d533` GPU 1 | 7.838M / 4.555M | 13.61 / **17.73 GiB** | 2.32 s/it | ~5.2 h |

### 3.2 并行启动与 SwanLab 追踪（`dehaze-probes`）

*(启动后自动在控制台日志与 `dehaze-probes` 看板追踪，待 8k 完成回填最终三子集 PSNR / SSIM 与配对归因结果。)*
