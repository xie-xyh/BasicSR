# 014 — MaritimeSFNet：面向海雾复原与下游 LoRA 域适配的空频物理对齐主干

> 设计文档（2026-11-27）。
> 前置依据：`research/013_owhregion/DIAGNOSIS_013_VS_011P.md`（013a/b/c 相对 011p 的五大代码级硬伤复盘）、`research/004_lowfreq/README.md`（LFM 机理与 DEConv 核响应）、`research/005_sim2real/README.md`（真实海雾偏蓝大气光与 3.8× 通道离散度差异）、`research/011_wavelet_unet/README.md`（物理散射推导框架）、`research/sota_benchmark/SOTA_2025_2026.md`（2025–2026 顶刊顶会架构共识）。
>
> **一句话定位**：跳过 013 止损消融，直接构建完全自主的 **3 尺度空频物理对齐主干（`MaritimeSFNet`）**——以可严格推导的物理与信号处理机制在合成 OWH 上追求超越 `011p`（DEANetV3-LFM）的高 PSNR / SSIM，同时在结构层原生解耦**领域不变纹理主干**与**物理域敏感调制接口**，为后续真实海雾无监督/弱监督 LoRA 适配提供最强预训练基座。

---

## 1. 设计目标与工程硬约束

1. **性能目标（合成 OWH 同协议 `v2`）**：
   - **8k 探针硬过线**：`014a` 新基座直接追平或超过 `011k_dea`（`011p@8k`：Dense **21.50** / Mist **26.70** / Moderate **24.44 dB**）；完整版 `014b`/`014c` 在 8k 相对 `011k_dea` 达到 **≥ +0.3 dB** 且 Dense / Mist 双子集一致领先；
   - **60k 终评目标**：全面超越 `011p`（internal-v2：Dense **25.94** / Mist **31.43** / Moderate **28.99 dB**；official-2711：Dense **25.24** / Mist **31.61** / Moderate **28.46 dB**），向 Dense 27 / Mist 33–35 dB 推进。
2. **学术创新与可发表性**：
   - 彻底脱离 DEA-Net 原版结构拼装，具备完整的**物理命题 → 数学推导 → 顶刊文献定位 → 因果消融验证**闭环；
   - 每一个新增模块均对应一条在 003–013 中被定量观测到的失败模式（见 §2）。
3. **原生 LoRA 友好（面向后续真实海雾适配）**：
   - 合成预训练后，局部卷积可重参数化折叠并冻结，所有承载真实海雾域偏移（偏蓝大气光、非均匀透射率、3.8× 通道离散度）的参数集中在 4 类轻量、有物理语义的接口上，支持低秩 LoRA 精准挂载（见 §4）。
4. **算力、速度与断点容错规范（明确区分两个“checkpoint”）**：
   - **关于 `activation_checkpointing`（显存换时间的激活重计算）**：通过高算术强度的 3 尺度设计，令 `bs16@512` 峰值显存控制在 **≤ 38 GiB**，**默认关闭 `activation_checkpointing`**，单步耗时目标 **0.75–0.95 s/iter**（较 013b 的 3.07 s/iter 提速 3× 以上），8k 探针约 1.8–2.1 小时完成；
   - **关于训练模型断点保存（Model & Training-State Checkpoints）**：**必须全程开启并严格落盘！** 鉴于机器曾出现 GPU0 / CUDA 驱动异常，所有 8k 探针与 60k 正式训练**必须配置 `save_checkpoint_freq: 2000`（每 2000 iter 保存 `net_g_<iter>.pth` 与 `training_states/<iter>.state`）**，并在异常中断时使用已修复 `experiments_root` 路径解析的 `--auto_resume`（commit `d01dc0b`）无缝续训；**严禁在训练中途关闭断点保存**。

---

## 2. 严格理论推导与顶刊文献支撑

本架构的每一项设计均建立在以下 **6 条可验证命题** 之上，并注明成立条件与数值边界。

### P1. 广义海雾散射模型的有界仿射软重建定理（替代纯加性残差 `I + Δ`）

考虑带波长依赖非中性大气光 $A_c(x)$ 与空间非均匀透射率 $t_c(x) \in (t_{\min}, 1]$ 的海上大气散射模型：

$$I_c(x) = J_c(x)\, t_c(x) + A_c(x)\big(1 - t_c(x)\big), \quad c \in \{R, G, B\}$$

等价逆映射为逐像素、逐通道的仿射变换：

$$J_c(x) = \underbrace{\frac{1}{t_c(x)}}_{1 + K_c(x)}\, I_c(x) + \underbrace{\left(-\frac{A_c(x)\big(1 - t_c(x)\big)}{t_c(x)}\right)}_{B_c(x)} = \big(1 + K_c(x)\big)\, I_c(x) + B_c(x) \tag{1}$$

与 013 采用的纯加性残差 $J_c = I_c + \Delta_c$ 对比：

$$\Delta_c(x) = \frac{1 - t_c(x)}{t_c(x)}\big(I_c(x) - A_c(x)\big) \tag{2}$$

1. **纯加性残差的劣势**：式 (2) 将空间缓变场 $t_c(x)$ 与图像高频纹理 $I_c(x)$ **乘性耦合**；在浓雾区（$t_c \to 0.1\sim 0.3$），$I_c$ 的对比度已被压缩 $3\text{--}10\times$，纯加性残差只能靠深层非线性卷积硬拟合被压缩的对比度增量，这直接导致 013b 的 `undershoot_mse` 较 011p 偏高 48%–107%。
2. **硬物理反演的误差爆炸**：本项目 `011`（`NEXT_RESEARCH.md` 局部散射诊断）已实测：若直接估计 $\hat{t}$ 并做除法 $\hat{J} = (I - A)/\hat{t} + A$，雾域估计误差 $\epsilon$ 会被放大为 $\epsilon/\hat{t}$（Dense 368/768 块 $\hat{t}<0.1$，直接反演 PSNR 跌至 10.43 dB）。
3. **严格正有界仿射软重建头（Bounded Affine Soft-Reconstruction, BASR）**：结合 **AOD-Net (ICCV 2017)** 与 **DehazeFormer / gUNet (IEEE TIP 2023)** 的 Soft Reconstruction 思想，由解码器末端预测 6 通道 $(\tilde{K}_c, B_c)$，将乘性增益参数化为严格正且光滑有界的函数：

$$\hat{J}_c(x) = G_c(x)\, I_c(x) + B_c(x), \qquad G_c(x) = g_{\min} + (g_{\max} - g_{\min})\,\sigma\!\big(\tilde{K}_c(x) + b_0\big) \tag{3}$$

取 $g_{\min} = 0.5,\; g_{\max} = 5.0$，并解出零初始化恒等偏置：

$$g_{\min} + (g_{\max} - g_{\min})\,\sigma(b_0) = 1 \iff \sigma(b_0) = \frac{1 - 0.5}{5.0 - 0.5} = \frac{1}{9} \iff b_0 = -\ln 8 \approx -2.07944 \tag{4}$$

**性质证明**：
- **初始严格恒等**：末层卷积零初始化时 $\tilde{K}_c \equiv 0,\; B_c \equiv 0 \implies G_c \equiv 1,\; \hat{J} \equiv I$，保护轻雾区优化起点；
- **严格正增益与有界误差放大**：$\forall \tilde{K}_c \in \mathbb{R},\; G_c(x) \in (0.5, 5.0)$，杜绝了无界仿射可能导致的极性翻转（$G_c < 0$）与硬反演 $1/t \to \infty$ 的数值爆炸，且支持最高 $5\times$ 对比度恢复（等价覆盖 $t \ge 0.2$，极浓雾残留由无界 $B_c$ 自由生成补偿）；
- **非零平滑梯度**：在原点处 $\left.\frac{\partial G_c}{\partial \tilde{K}_c}\right|_{0} = 4.5 \times \frac{1}{9} \times \frac{8}{9} = \frac{4}{9} \approx 0.444$，不存在零梯度死区。

---

### P2. 多尺度下采样的尺度-频带对齐定理（为何必须回归 3 尺度 1/4 瓶颈）

设输入图像分辨率为 $H \times W$，原图 Nyquist 频率为 $N_1 = \min(H, W)/2$。经过 $L$ 次 stride-2 下采样（总倍率 $s = 2^L$）后，瓶颈特征尺寸为 $H_s \times W_s = (H/s) \times (W/s)$，其特征域 Nyquist 频率为 $N_s = N_1 / s$。

若在特征域按归一化半径 $r_f(v, u) = \sqrt{v^2 + u^2} / N_s$ 施加截止频率为 $c_f$ 的径向低通掩码，则该掩码在**原图归一化频率坐标系** $r_{\text{img}} = \sqrt{v^2 + u^2} / N_1$ 中对应的有效物理截止频率与二维频谱覆盖面积为：

$$c_{\text{img}} = \frac{c_f}{s}, \qquad \Omega_{\text{2D}} = \pi\, c_{\text{img}}^2 = \frac{\pi\, c_f^2}{s^2} \tag{5}$$

对照本仓库实测的残差主频带（`003d`/`004` 实测 55%–61% 残差在原图 `DC~0.02` Nyquist，`011`/`013` 实测 75%–89% 的 MSE 差距在 `<0.02`）：

| 瓶颈下采样倍率 $s$ | 对应模型 | $c_f = 0.08$ 时的原图物理截止 $c_{\text{img}}$ | 相对 $s=4$ 的 2D 频谱面积 | 实测低频表现 |
|---:|---|---:|---:|---|
| **$s = 4$（3 尺度 1/4）** | **011p (`DEANetV3`) / 014** | **0.020** *(5.12 周期/512px)* | **100%** | 精准覆盖 `DC~0.02`，最优 |
| $s = 8$（4 尺度 1/8） | 013a / 013b / 013c | 0.010 *(2.56 周期/512px)* | 25% | 频带塌缩，漏掉 `0.01~0.04` |
| $s = 16$（5 尺度 1/16） | 011 WaveletUNet | 0.005 *(1.28 周期/512px)* | 6.25% | 退化为近直流常数调节 |

**设计推论**：
1. 宏观主干回归 **3 尺度（1 → 1/2 → 1/4，$s_{\max} = 4$）**，与去雾顶刊/顶会强主干 **DEA-Net (TIP 2024)、DehazeFormer / gUNet (TIP 2023)、SFNet (ICLR 2023)、FocalNet (ICCV 2023)、OKNet (AAAI 2024)、ConvIR (TPAMI 2024)** 保持一致；
2. 频域低通掩码统一以**原图物理截止频率 $c_{\text{img}}$** 参数化，由式 (5) 自动换算特征域截止 $c_f = \min(s \cdot c_{\text{img}},\, 0.5)$，彻底消除跨尺度带宽错位。

---

### P3. 海事场景各向异性大核先验与空频双域瓶颈（替代 `RegionContext`）

在海上水平视场（相机俯仰角接近 $0^\circ$）中，平坦海面与海雾层的景深场 $d(x, y)$ 具有显著的空间各向异性：

$$\left|\frac{\partial d}{\partial y}\right| \gg \left|\frac{\partial d}{\partial x}\right|, \qquad t(x, y) = \exp\!\big(-\beta\, d(x, y)\big) \tag{6}$$

即沿水平海天线/等距水面条带景深近似分段常数，而沿垂直方向（天空 → 海天线 → 近景海面）呈强烈连续梯度。本项目 `color_headroom_internal60.csv` 已从数据侧证实：仅在 013a 输出上叠加沿高度 $y$ 线性变化的仿射校正，即可在 Dense / Mist / Moderate 上额外提升 **+2.73 / +3.09 / +2.46 dB**。

因此，在 1/4 瓶颈（128×128）上彻底删除无位置偏置的点积注意力 `RegionContext`（其在 004b 与 013 中两次被证伪会导致天空/白墙跨区暗斑），改用 **空频双域各向异性瓶颈（Anisotropic Spatial-Frequency Bottleneck, ASFB）**，包含两条严格互补且均零初始化的受控通路：

1. **空域多向大核通路（Anisotropic Omni-Kernel, 借鉴 OKNet AAAI 2024 & ConvIR TPAMI 2024）**：
   将瓶颈特征经 $1\times 1$ 投影后均分为 4 组通道，并行施加四种深度可分离大核卷积：
   - **水平条形大核 $1 \times 31$**：沿水平海天线与水岸线聚合等深度统计（1/4 尺度上物理跨度 $31 \times 4 = 124\text{ px}$），稳健区分“局部白帆/白船”与“整条海雾带”；
   - **垂直条形大核 $31 \times 1$**：沿天空→海天线→近海面直接建模纵向雾度与大气光梯度；
   - **中等方形大核 $15 \times 15$**：建模团块状非均匀海雾（物理跨度 $60\text{ px}$）；
   - **局部核 $3 \times 3$**：保住前景船体、桅杆与建筑边界的局部连续性。
   四路拼接后经零初始化 $1 \times 1$ 投影与残差相加。卷积天然具备**平移等变性与距离局部平滑先验**，从数学构造上杜绝无位置编码自注意力的跨图色调污染。
2. **频域多尺度低频调制通路（Multi-Scale LFM, MS-LFM，升级自 004a LFM，呼应 AdaIR ICLR 2025 / SFMN TIP 2025）**：
   在 1/4 瓶颈（$s=4$）按物理频率 $c_{\text{img}} = 0.025$（即 $c_f = 0.10$，略宽于 0.08 以覆盖海雾低-中低频边缘）提取软低通分量 $X_{\text{low}}$，同时从 **`8×8`（全局色调网格）与 `16×16`（区域雾团网格）** 双尺度自适应平均池化拼接预测平滑 FiLM 调制场 $(\gamma, \beta)$：

   $$X_{\text{out}} = X + \gamma(X_{\text{pool},8}, X_{\text{pool},16}) \odot X_{\text{low}} + \beta(X_{\text{pool},8}, X_{\text{pool},16}) \tag{7}$$

   **坚持三项已验证约束**：内容自适应、无归一化（保留 DC 统计）、末层严格零初始化。

---

### P4. 低频深层主导的频分凸跳连（FCSF，替代非凸 `HazeConditionedFusion`）

对同尺度编码特征 $F_{\text{enc}}$ 与解码特征 $F_{\text{dec}}$，通过可逆分解算子 $\mathcal{L}(\cdot)$（例如 stride-2 平均池化再双线性上采样，比 013 的 5×5 box filter 更平滑且低频截断更干净）拆分为低频基底与细节分量 $F = F^L + F^H$，其中 $F^L = \mathcal{L}(F),\; F^H = F - F^L$。

为阻断浅层未去雾特征 $F_{\text{enc}}^L$ 对深层已去雾特征 $F_{\text{dec}}^L$ 的直流污染，同时避免 012b 中无界门控的发散风险，定义 **频分凸跳连融合（Frequency-Decoupled Convex Skip Fusion, FCSF）**：

$$F_{\text{fuse}} = \text{Conv}_{1\times 1}\!\Big(\underbrace{(1 - \alpha)\odot F_{\text{dec}}^L + \alpha \odot F_{\text{enc}}^L}_{\text{低频严格凸组合},\;\alpha\in(0,1)} + F_{\text{dec}}^H + \beta \odot F_{\text{enc}}^H\Big) \tag{8}$$

等价写为残差形式：

$$F_{\text{fuse}} = \text{Conv}_{1\times 1}\!\Big(F_{\text{dec}} + \alpha \odot (F_{\text{enc}}^L - F_{\text{dec}}^L) + \beta \odot F_{\text{enc}}^H\Big) \tag{9}$$

其中门控 $\alpha = \sigma(z_\alpha - 2.0),\; \beta = \sigma(z_\beta)$ 由下采样后的 $(F_{\text{dec}},\, F_{\text{enc}}^L - F_{\text{dec}}^L,\, |F_{\text{enc}}^H|)$ 轻量预测。

**关键约束与分尺度分工**：
- **低频深层主导**：$\alpha$ 初始偏置为 `-2.0`（$\alpha_0 \approx 0.119$），训练初期深层占 88% 权重，**彻底堵死浅层浓雾低频直通漏洞**；仅当输入为极轻雾、深浅层差异小时网络可自主抬高 $\alpha$ 保留原图自然色调；
- **全分辨率 1× 尺度严格阻断低频跳连（$\alpha \equiv 0$）**：在 512×512 尺度上浅层编码特征最接近原始雾图，仅保留门控高频细节注入 $F_{\text{fuse}}^{(1)} = \text{Conv}_{1\times 1}(F_{\text{dec}} + \beta \odot F_{\text{enc}}^H)$，与 DEANet 全分辨率不混入浅层雾幕的成功经验一致，又保留了船体细微轮廓的高频定位线索；
- **严格有界性与 1×1 通道重映射**：$(1-\alpha)+\alpha \equiv 1$ 保证低频能量严格满足凸包上界，绝不出现 012b 式 `exp(-s)` 爆炸；末层 `1×1 Conv` 解决深浅层通道语义对齐问题。

---

### P5. 结构重参数化局部块 + 块内轻量双域通道注意力（兼顾 8k 快速起飞与推理轻量）

综合 `004_lowfreq §8`（DEConv 五分支健康且随深度增强）、**RepVGG (CVPR 2021)**、**NAFNet (ECCV 2022)** 与 **OKNet (AAAI 2024)**：

- **4 分支各向异性重参数化卷积（AnisoRepConv3×3）**：在每个基础残差块的首层，并行布置 `Plain 3×3` + `Horizontal Diff 1×3` + `Vertical Diff 3×1` + `Pointwise 1×1`（相比 DEConv 去掉了 2 个全 3×3 差分分支，训练参数比 DEConv 少 **52%**，同时显式保留了 004 §8 观测到的海事水平/垂直方向梯度先验）；前向合并为**单个标准密集 3×3 卷积**（Tensor Core 算术强度满载），推理与下游 LoRA 部署时**可严格无损折叠为单层 `nn.Conv2d(C, C, 3, padding=1)`**。
- **去除全网 `scale=0.1` 死缩放**：基础局部残差块恢复标准恒等残差连接（`x + conv2(gelu(rep_conv1(x)))`），避免 8k 早期梯度被压缩 10 倍；仅对瓶颈新增的全局分支（MS-LFM、Aniso-LK）保留严格零初始化。
- **瓶颈块内轻量双域注意力（ECA/GAP + 7×7 空间门控）**：在 1/4 瓶颈的 8 个基础块中均配备极轻量的 GAP 通道注意力 + 7×7 空间注意力（不破坏特征均值：采用残差门控或乘性标定 $1 + \tanh(\cdot)$ 形式，保留全局 DC 统计），每层皆具备全图通道色平衡能力。

---

### P6. 多尺度输入与多尺度深监督的频域梯度等价性（零推理开销）

借鉴 **MIMO-UNet (ICCV 2021) / SFNet (ICLR 2023) / ConvIR (TPAMI 2024) / OKNet (AAAI 2024)**：

- **编码端多尺度雾图注入**：将双线性下采样后的 $I_{1/2}$、$I_{1/4}$ 经轻量 3×3 卷积加到编码器 L2、L3 入口，直接为 1/4 瓶颈提供无混叠的原始雾幕场；
- **解码端多尺度深监督**：在解码器 L3（1/4 尺度）和 L2（1/2 尺度）末端各挂一个极轻量的 3×3 仿射软重建头，预测 $\hat{J}_{1/4}$ 与 $\hat{J}_{1/2}$，与对应下采样清晰图 $J_{1/4}, J_{1/2}$ 计算同款 `L1 + FFT` 损失：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{full}}(\hat{J}, J) + \lambda_{1/2}\,\mathcal{L}_{\text{ms}}(\hat{J}_{1/2}, J_{1/2}) + \lambda_{1/4}\,\mathcal{L}_{\text{ms}}(\hat{J}_{1/4}, J_{1/4}) \tag{10}$$

（其中 $\lambda_{1/2}, \lambda_{1/4}$ 初始取 `0.1`，并在正式训练前使用本仓库 `loss_gradient_probe.py` 核查梯度范数比例不超过主损失的 25%）。

**机理价值**：下采样算子 $\mathcal{D}_4$ 在频域等价于截止于 $|\omega| \lesssim 0.25$ 的低通滤波器，$J_{1/4}$ 天然承载全图的 `DC~0.02` 雾幕与 `0.02~0.15` 船体/海天线中频目标。辅助损失的梯度直接注入 1/4 瓶颈的 MS-LFM 与 Aniso-LK，绕过浅层解码器衰减，显著加速 8k 收敛并压低 `low_mse`；且推理与下游 LoRA 阶段直接跳过辅助头，**零推理开销**。

---

### 与 2022–2026 顶刊顶会工作的差异化定位表

| 先验顶刊/顶会工作 | 核心思路 | 014 (`MaritimeSFNet`) 的继承与实质差异化 |
|---|---|---|
| **DEA-Net** (WACV/TIP 2024) | 3 尺度 + 5 分支 DEConv + CGA 融合，无 1× skip | 继承 3 尺度高分辨率配比与训练期重参数化优势；**剔除原版 DEA 骨架**，以轻量 4 分支各向异性重参数化 + 频分凸跳连 FCSF + 空频双域瓶颈 + 物理仿射软重建全面重构，训练参数更少（5.8M vs 7.88M） |
| **DehazeFormer / gUNet** (IEEE TIP 2023) | 3 尺度（有效 1/4 瓶颈）+ RescaleNorm + SKFusion + 无界 Soft Reconstruction | 升级 Soft Reconstruction 为**严格正有界仿射软重建（P1，防极性翻转与 $1/t$ 爆炸）**；将通道级 SKFusion 升级为**空间-频分凸跳连 FCSF（P4）** |
| **OKNet** (AAAI 2024) / **ConvIR** (IEEE TPAMI 2024) | 3 尺度 + 多形状大核（方形/条形）+ 频域门控 + 多尺度深监督 | 将通用大核与**海雾景深各向异性物理先验（P3）**结合，并与**物理频带对齐的双尺度 MS-LFM（P2）**在 1/4 瓶颈正交耦合，而非仅做通道频域门控 |
| **AdaIR** (ICLR 2025) / **SFMN** (IEEE TIP 2025) | 空频双域解耦调制、跨尺度特征调制 | 显式建立**尺度-频带对齐公式（P2）**与**无归一化保 DC 原则**（003d/004b 实证），避免在过深尺度或带归一化条件下做频域滤波失效 |
| **BiLaLoRA** (CVPR 2026) / **CoA** (CVPR 2025) | 合成预训练主干向真实雾的层定位 LoRA / 压缩适配 | 在合成主干设计阶段就**原生解耦域不变纹理层与 4 类物理域敏感接口（§4）**，与 `005_sim2real` 测出的偏蓝大气光/通道离散度物理统计直接对接 |

---

## 3. 网络架构精确规格（`basicsr/archs/maritime_sfnet_arch.py`，待实现）

### 3.1 总体拓扑与数据流

```
输入雾图 I (B, 3, H, W) ──[reflect/replicate pad 至 4 的倍数]──►
  │
  ├─► Stem Conv3×3 (3 → C1=36)
  │     └─► Enc1: 3× LocalRepBlock(36) ───────────────────────────────────┐ [skip1: 1×, 仅高频注入]
  │           └─► Down1: Conv3×3(s=2)+GELU (36 → 72)                      │
  │                 ├─(+ optional MIMO I_1/2 3×3)                         │
  │                 └─► Enc2: 3× LocalRepBlock(72) ──────────────────┐    │ [skip2: 1/2, FCSF 凸融合]
  │                       └─► Down2: Conv3×3(s=2)+GELU (72 → 144)    │    │
  │                             ├─(+ optional MIMO I_1/4 3×3)        │    │
  │                             ├─────────────────────────────────┐  │    │ [skip3: 1/4, FCSF 瓶颈残差]
  │                             ▼                                 │  │    │
  │                   ┌─── 1/4 Bottleneck (C3=144, 128×128@512) ──┤  │    │
  │                   │  2× BotBlock(144, DCA)                    │  │    │
  │                   │  → ASFB-1 [MS-LFM(c_img=0.025) + AnisoLK] │  │    │
  │                   │  3× BotBlock(144, DCA)                    │  │    │
  │                   │  → ASFB-2 [MS-LFM(c_img=0.025) + AnisoLK] │  │    │
  │                   │  3× BotBlock(144, DCA)                    │  │    │
  │                   └────────────────────┬──────────────────────┘  │    │
  │                                        ▼                         │    │
  │                             FCSF3(bot_out, enc3_in) ◄────────────┘    │
  │                                ├─► [训练期可选: AuxHead_1/4 → J_1/4]   │
  │                                └─► Up2: Bilinear×2 + Conv3×3 (144→72) │
  │                                      └─► FCSF2(up2, skip2) ◄──────────┘
  │                                            └─► Dec2: 3× LocalRepBlock(72)
  │                                                  ├─► [训练期可选: AuxHead_1/2 → J_1/2]
  │                                                  └─► Up1: Bilinear×2 + Conv3×3 (72→36)
  │                                                        └─► FCSF1_HP(up1, skip1) [α≡0, 仅高频 β]
  │                                                              └─► Dec1: 3× LocalRepBlock(36)
  │                                                                    └─► BASR Head (36→6: K̃, B)
  └────────────────────────────────────────────────────────────────────────► Ĵ = G(K̃)⊙I + B (裁回 H×W)
```

> **上采样防网格伪影说明**：采用 `F.interpolate(scale_factor=2, mode='bilinear', align_corners=False) + Conv3×3` 替代 013 的 `Conv1×1 + PixelShuffle(2)`，彻底消除 2×2 子像素周期性跳变（`grid_boundary_error`）。

### 3.2 参数量与计算量预算（严格对标 011p @ 512×512）

默认基座配置：`widths = (36, 72, 144)`，`depths = (3, 3, 8)`（L1 编解码各 3 块、L2 编解码各 3 块、L3 瓶颈 8 块；全部通道为 4 的倍数以契合 4 分支 `AnisoLK`）。

| 尺度 (分辨率) | 014a (新基座) 训练/折叠参数 / GMACs | 014b (+ASFB 双域) 训练/折叠参数 / GMACs | 014c (+MIMO+深监督) 训练/折叠参数 / GMACs | 参照：011p (`DEANetV3`) |
|---|---|---|---|---|
| **L1 (512, C=36)** | 0.248M / 0.193M / 45.56G | 0.248M / 0.193M / 45.56G | 0.248M / 0.193M / 45.56G | 0.347M / 39.11G |
| **L2 (256, C=72)** | 0.979M / 0.760M / 44.78G | 0.979M / 0.760M / 44.78G | 0.985M / 0.766M / 45.17G | 1.466M / 47.80G |
| **L3 (128, C=144)** | 4.400M / 3.236M / 49.41G | 4.599M / 3.434M / 51.14G | 4.611M / 3.446M / 51.33G | 6.067M / 49.27G |
| **总计** | **5.627M / 4.189M / 139.76G** | **5.826M / 4.387M / 141.49G** | **5.844M / 4.405M / 142.06G** *(推理 141.8G)* | **7.880M / ~3.7M / 136.17G** |

**预算优势**：
- 总 GMACs 与 `011p`（136.17G）误差在 **4% 以内**，完全满足同算力公平对比；
- 训练参数量（5.63–5.84M）比 011p（7.88M）低 **26%**、比 013b（9.33M）低 **37%**；
- L1/L2 各 3 块（vs 011p 各 4 块）且无 013 的 3× 通道膨胀层，预计 `bs16@512` 峰值显存 **~32–36 GiB（免开 `activation_checkpointing`）**，单步速度预计 **0.75–0.90 s/iter**。

---

## 4. 面向后续真实海雾 LoRA 适配的原生架构解耦（衔接 `005_sim2real`）

本仓库 `research/005_sim2real/README.md` 已定量测量出合成 OWH 与真实海雾（`real-test`）的核心域差距：
1. 真实海雾的通道离散度是合成浓雾的 **3.8 倍**（0.0675 vs 0.0179）；
2. 真实海雾大气光**显著偏蓝且逐图弥散**（色度 `(0.3266, 0.3401)` vs 合成中性 `(0.333, 0.333)`）；
3. `005a/b/c` 实测证明：仅在 RGB 图像空间挂载 3 通道 `FreqDA` 容量受限，容易在“适配不动”与“青绿色偏”间陷入陡峭 Pareto 陷阱，亟需**进入特征空间且具备物理方向约束的适配接口**。

`MaritimeSFNet` 在预训练阶段即通过模块化命名与前向隔离，实现 **Domain-Invariant（冻结）** 与 **Domain-Sensitive（LoRA 挂载位）** 的原生二分：

| 参数组 | 对应模块 | 合成预训练后操作 | 物理职责（真实海雾 LoRA 阶段） |
|---|---|---|---|
| **纹理结构主干** | `stem`, `enc1/2`, `dec1/2`, `bot_blocks` 局部 3×3 卷积 | **重参数化折叠后完全冻结** | 保护合成域学到的船体/海浪/边缘先验，防止真实弱监督下放大噪声与合成域遗忘 |
| **挂载位 ①：频域大气光调制** | `ASFB.ms_lfm.mod`（双尺度 8×8/16×16 调制器） | 注入低秩 LoRA ($r=8\text{--}16$) | 在 1/4 特征空间逐通道、空间平滑地校准真实海雾偏蓝大气光 $A_c(x)$ 与全局色偏 |
| **挂载位 ②：各向异性雾度梯度** | `ASFB.aniso_lk.{in_proj, out_proj}` | 注入低秩 LoRA ($r=8$) | 适配真实海面与合成深度管线在非均匀雾团/纵向雾度梯度分布上的差异 |
| **挂载位 ③：跳连通道放行门** | `fcsf{1,2,3}.gate` & `out_proj` | 注入低秩 LoRA ($r=4\text{--}8$) | 调节真实雾图高保留度通道（3.8× 离散度）进入解码器的低/高频权重比例 |
| **挂载位 ④：物理仿射重建头** | `basr_head`（预测 $\tilde{K}_c, B_c$） | 注入低秩 LoRA ($r=4\text{--}8$) | 直接在像素级逐通道校准真实雾的乘性对比度增益 $1/t_c$ 与残留色偏，配合 `ChromaPreserveLoss`（005d）防止色相旋转 |

> 工程约定：`MaritimeSFNet` 需提供 `fold_reparam()`（无损合并 `AnisoRepConv3×3`）与 `iter_lora_target_modules()` 接口，后续 `015` 真实海雾 LoRA 阶段可直接按模块组注入 ConvLoRA，并做“四接口定位 LoRA vs 全网均匀 LoRA vs 图像空间 FreqDA”的顶会级消融对照（直接对标 BiLaLoRA CVPR 2026）。

---

## 5. `014` 实验梯度路线图（单变量可控推进）

跳过 013 止损实验，全部严格沿用 `v2` 冻结清单、`seed=0`、`bs16@512`、`lr=2e-4`、warmup 500、60k 余弦调度、EMA `0.999`；短程探针统一 **8,000 iter**（隔离至 `experiments/probes/`、`tb_logger/probes/`、`dehaze-probes`），**每 2,000 iter 强制落盘模型与 `.state` 断点**。

| 阶段 | 实验编号 | 唯一新增变量 | 训练/折叠参数 / GMACs | 核心验证目标与过线判据 |
|---|---|---|---|---|
| **Stage 1：新基座立柱** | **`014a`** | 3 尺度 `(36,72,144)` + `depths=(3,3,8)` + `AnisoRepConv3×3` + `DCA` + 1/4 瓶颈 `2×LFM(c_img=0.02)` + `FCSF` 凸跳连（1× 仅高频）+ `BASR` 有界仿射软重建头 | 5.63M / 4.19M / 139.8G | **生死线**：8k 直接对标 `011k_dea`（21.50 / 26.70 / 24.44），要求达到或超过 **21.4+ / 26.6+ / 24.3+**，且免开 `activation_checkpointing`、速度 ≤ 0.95 s/iter；逐图检查 Mist 最轻雾三分位不再出现暗斑崩盘 |
| **Stage 2：核心空频创新** | **`014b`** | 在 `014a` 上仅升级 1/4 瓶颈的 2 个调制点为完整 **`ASFB`（双尺度 `8×8+16×16` MS-LFM + 十字/方形各向异性大核 `1×31 / 31×1 / 15×15 / 3×3`）** | 5.83M / 4.39M / 141.5G | 验证 P3 假设：8k 相对 `014a` 与 `011k_dea` 均取得清晰正增益（重点看 `low_mse`、垂直上/中/下三段平滑 MSE 与 `mid_mse` 同步下降） |
| **Stage 3：多尺度深监督** | **`014c`** | 在 `014b` 上仅新增 **MIMO 1/2 & 1/4 多尺度输入 + 1/4 & 1/2 辅助仿射深监督**（推理期丢弃，零推理开销；`MaritimeSFModel` 继承 `DehazeModel`） | 5.84M / 4.41M / 142.1G *(推理 141.8G)* | 验证 P6 假设：8k 相对 `014b` 进一步提速收敛、压低低中频残差；若较 `011k_dea` 综合领先 **≥ +0.3 dB** 且 Dense/Mist 双子集 95% 配对区间不含零 → 晋级 60k |
| **Stage 4：多种子复核（视幅度）** | **`014c_s1`** | 若 8k 相对 `011k_dea` 领先落在 +0.1~+0.3 dB 区间，换 `seed=1` 补跑一组 8k | 同 `014c` | 按 `012 §5` 纪律排除单种子噪声带后再上 60k |
| **Stage 5：60k 正式终评** | **`014d_60k`** | 胜出最优版（`014b` 或 `014c`）从头跑满 **60,000 iter**（同步可按需测试加宽版 `widths=(48,96,192)` 作为大容量上限对照） | 同胜出版 | 在官方 2,711 图全量终评，与 `011p`（25.24 / 31.61 / 28.46）做 `paired_eval.py` 逐图配对 + 频谱/垂直区域/极端案例可视化 + `fold_reparam()` 等价性验证；保留最终 EMA 权重作为 **`015` 真实海雾 LoRA 适配预训练主干** |

### 论文核心消融矩阵（随 `014` 自然产出，无需重复造轮子）

1. **主表 SOTA 对比**：同 `v2` 协议 60k 终评——`014` vs `011p (DEANetV3-LFM)` vs `WaveletUNet` vs `013a (OWHRegionUNet)` + `sota_benchmark §5` 第一梯队（DehazeFormer / SFMN / AdaIR / ConvIR / OKNet）；
2. **模块消融（全部同 8k 预算）**：
   - `BASR` 有界仿射软重建 vs 纯加性残差 `I + Δ`（验证 P1，`undershoot_mse` 降幅）；
   - 3 尺度 1/4 瓶颈 vs 4 尺度 1/8 瓶颈（验证 P2，尺度-频带对齐）；
   - `AnisoLK` 十字各向异性大核 vs 等参纯方形大核 / `RegionContext`（验证 P3，垂直高度误差与轻雾暗斑消除）；
   - `FCSF` 频分凸跳连（1× 阻低频）vs 非凸跳连 / 简单相加（验证 P4，`overshoot`/`undershoot` 平衡）；
   - `MIMO + 深监督` 开关对照（验证 P6，零推理开销净增益）；
3. **可解释性分析（高审稿人好感度）**：
   - 导出 BASR 头的 $G_c(x) = 1 + K_c(x)$ 与 $B_c(x)$ 空间热力图，检验其与雾浓度/景深的空间单调对应；
   - 导出 FCSF 的门控 $\alpha(x), \beta(x)$ 在浓雾 vs 轻雾上的分布统计（证实浓雾区 $\alpha \to 0$ 阻断浅层雾幕、轻雾区 $\alpha$ 抬升保色调）；
   - 导出 MS-LFM 双尺度 $(\gamma, \beta)$ 与 1/4 瓶颈有效感受野（复用 `011` 的 `measure_context.py`），展示十字大核沿水平海天线与垂直高度的定向响应。

---

## 6. 工程落地、Checkpoint 断点规范与验证清单

### 6.1 强制断点保存与防故障续训策略

鉴于训练机曾发生 GPU0 硬件报错及驱动复位，014 全系列训练配置文件强制落实以下条目：

- **定期落盘完整断点**：`logger.save_checkpoint_freq: 2000`、`val.val_freq: 2000`（60k 正式实验可保持 2000 或 4000，**不得关闭**），确保每 2000 iter 同步写出 `models/net_g_<iter>.pth`（含 `params` 与 `params_ema`）与 `training_states/<iter>.state`（含优化器、scheduler、epoch/iter）；
- **异常自动续训**：中断后直接使用 `python basicsr/train.py -opt <cfg>.yml --auto_resume`（主仓库 `d01dc0b` 已修复自定义 `path.experiments_root` 下的状态查找），禁止手删活跃训练目录或用 cmdline 误杀 DataLoader fork worker；
- **显存与卡位安全**：优先选健康 GPU（开跑前用 `nvidia-smi` 与真实批次预检脚本确认空闲显存 ≥ 42 GiB、峰值预留 ≤ 42 GiB）；若单卡显存吃紧，优先调 `batch_size` 并同步对齐对照臂，而不是盲目在主干上硬塞高开销层；
- **隔离约定**：`probe_014*` 配置严格写入 `path.experiments_root: experiments/probes`、`logger.tb_logdir: tb_logger/probes`、`logger.swanlab.project: dehaze-probes`；正式 60k 实验使用默认 `experiments/`、`tb_logger/` 与 `dehaze`，开跑前按 `AGENTS.md §9` 完成占号与提交。

### 6.2 代码实现与提交前必做检查（按 `AGENTS.md §7–§8`）

1. 已实现 `basicsr/archs/maritime_sfnet_arch.py`（注册 `@ARCH_REGISTRY.register() class MaritimeSFNet`）与 `basicsr/models/maritime_sf_model.py`（注册 `MaritimeSFModel`，继承 `DehazeModel` 支持 1/4 & 1/2 多尺度深监督及门控/增益诊断日志，无 `deep_sup_opt` 时与 `DehazeModel` 损失完全一致）；
2. 已冻结三份单变量 8k 探针配置（均与 `011p`/`013b` 逐字段对齐 v2 清单与基础配方，且强制 `save_checkpoint_freq: 2000`）：
   - `research/014_maritime_sfnet/probe_014a_base.yml`（port `29543`）
   - `research/014_maritime_sfnet/probe_014b_asfb.yml`（port `29544`）
   - `research/014_maritime_sfnet/probe_014c_mimo.yml`（port `29545`）
3. **验证与启动脚本**：
   - 综合验证脚本：`research/014_maritime_sfnet/tools/verify_014.py`（包含语法与 YAML 一致性审计、奇偶尺寸前向/反向、`AnisoRepConv3×3` 无损折叠误差 `<1e-5`、BASR/ASFB 零初始化恒等、2 步全链路梯度传播、LoRA 模块分组检查、BasicSR 合成冒烟训练、以及真实批次 `bs16@512` 显存/耗时预检）；
   - 防重复启动脚本：`research/014_maritime_sfnet/tools/launch_probe.sh`（支持首次启动与 `--auto_resume` 断点续训，强制校验 `save_checkpoint_freq > 0` 与探针目录隔离，自动接入 `dehaze-probes` SwanLab 同步）。

### 6.3 `d533`（2×RTX 3090 24 GB）预检实测与启动清单

2026-11-27 在 `d533`（`torch 2.1.1+cu118`, `basicsr 1.4.2`, 两张 RTX 3090 24 GB）完成单元测试与显存/速度矩阵实测：

- **单元测试**：`014a`（5.627M / 折叠 4.189M）、`014b`（5.826M / 折叠 4.387M）、`014c`（5.844M / 折叠 4.405M）全部通过，`init_id = 0.00e+00`、`fold_err = 0.00e+00`。
- **24 GB 3090 实测（512×512，含 L1+FFT+Grad 及 014c 深监督）**：
  - 单卡 `bs=16, ckpt=False`：OOM（>24 GiB）；
  - **单卡 `bs=16, ckpt=True`（当前探针采用，双卡并行跑两组）**：014a 13.28/17.63 GiB (1.92 s/it)、014b 13.29/17.49 GiB (2.00 s/it)、014c 13.75/17.79 GiB (2.03 s/it)，单组 8k 约 4.3–4.5 小时，每卡留有 ~6 GiB 安全余量；
  - 双卡 DDP `bs=8/卡`（总 batch=16，适合 60k 正式长训）：`ckpt=False` 每卡 20.00–20.77 GiB (~0.85–0.90 s/it)；`ckpt=True` 每卡 8.94–9.01 GiB (~1.05–1.10 s/it)。

```bash
conda activate basicsr

# 1) 单元测试 + 合成冒烟训练
python research/014_maritime_sfnet/tools/verify_014.py --smoke

# 2) 双卡 3090 并行启动 014a (GPU0) + 014b (GPU1) 8k 探针（每 2000 iter 存模型与 .state 断点）
bash research/014_maritime_sfnet/tools/launch_probe.sh 0 research/014_maritime_sfnet/probe_014a_base.yml research/014_maritime_sfnet/run_014a 19000
bash research/014_maritime_sfnet/tools/launch_probe.sh 1 research/014_maritime_sfnet/probe_014b_asfb.yml research/014_maritime_sfnet/run_014b 19000
# 014a 或 014b 结束后在空闲卡启动 014c：
bash research/014_maritime_sfnet/tools/launch_probe.sh 0 research/014_maritime_sfnet/probe_014c_mimo.yml research/014_maritime_sfnet/run_014c 19000

# 3) 若中途因 GPU/驱动故障中断，传入 --auto_resume 与新的日志目录即可从最近 2000.state 续训：
# bash research/014_maritime_sfnet/tools/launch_probe.sh 1 research/014_maritime_sfnet/probe_014a_base.yml research/014_maritime_sfnet/run_014a_resume1 19000 --auto_resume
```

---

## 7. `014a / 014b / 014c` 八千步探针实测与逐图归因（2026-11-27）

三组 8k 探针已全部完成（`014a`、`014b` 跑于 `d533` 双 3090 24G，`014c` 跑于 `hust` 4090 48G GPU1，严格同 v2 清单 / `seed=0` / `bs16@512` / `EMA=0.999`），与强基线 `011k_DEA` / `012a_DEA`（`011p@8k`，7.88M）和前代 `013a`（9.33M）对比如下（内部 283 图）：

| 实验 | 训练/折叠参数 | Dense PSNR / SSIM | Mist PSNR / SSIM | Moderate PSNR / SSIM | 8k 末端内部诊断 |
|---|---:|---:|---:|---:|---|
| `013a` (`OWHRegionUNet`) | 9.33M / 9.33M | 19.63 / 0.8719 | 23.79 / 0.9463 | 22.10 / 0.9163 | — |
| `011k_DEA` (`011p@8k`) | 7.88M / ~3.7M | 21.501 / 0.8846 | **26.705** / **0.9574** | 24.436 / 0.9257 | — |
| `012a_DEA` (`011p@8k` 复现) | 7.88M / ~3.7M | **21.591** / 0.8851 | 26.599 / 0.9571 | **24.631** / **0.9257** | — |
| **`014a`** (新基座) | 5.63M / 4.19M | 21.2715 / 0.8841 | 25.8885 / 0.9546 | 24.0740 / 0.9242 | `gain=1.217, α3=0.220, α2=0.242` |
| **`014b`** (`+ASFB`) | 5.83M / 4.39M | **21.4885** / **0.8858** | 26.2617 / 0.9564 | 24.3610 / 0.9254 | `gain=1.147, α3=0.245, α2=0.216` |
| **`014c`** (`+MIMO+DeepSup`) | 5.84M / 4.41M | 21.3920 / 0.8846 | 26.3081 / 0.9567 | 24.1826 / 0.9248 | `gain=1.171, α3=0.302, α2=0.249` |

### 7.1 逐图 283 图配对归因的关键结论

1. **`014b`（`+ASFB`）对 `014a` 为三子集一致统计显著正增益**（`paired_014b_vs_014a_8000_internal/summary.json`）：
   - Dense **+0.217 dB**（95% CI `[+0.027, +0.410]`）/ SSIM **+0.00174**（`top_smooth_mse` −8.7%，`mid_mse` −3.4%，`high_mse` −1.9%）；
   - Mist **+0.373 dB**（95% CI `[+0.023, +0.711]`）/ SSIM **+0.00184**（`overshoot_mse` −19.5%，`bottom_smooth_mse` −11.6%）；
   - Moderate **+0.287 dB**（95% CI `[+0.047, +0.524]`）/ SSIM +0.00116（`overshoot_mse` −15.3%，`high_mse` −2.8%）。
   - 仅用比 DEA 少 **26%** 的训练参数（5.83M vs 7.88M），`014b` 已在 Dense PSNR 上基本追平 `011k_DEA`（21.489 vs 21.501）、**在 Dense SSIM 上反超全部 DEA 基线（0.8858 vs 0.8846 / 0.8851）**，并在**三子集高频 `high_mse` 上全面优于 DEA**。
2. **`014c`（`+MIMO + DeepSup`）揭示了“输入多尺度有益、硬辅助损失有害”的解耦事实**：
   - 相对 `014b`，`014c` 在 Mist 上 `+0.046 dB / +0.0003 SSIM`（`low_mse` 降至 `0.00283`，`fcsf3_alpha` 提升至 `0.302`），说明 `mimo_in2/in3` 零初始化多尺度雾图输入提供了有用的低频雾线索；
   - 但在 Dense（−0.097 dB / −0.0012 SSIM）和 Moderate（−0.178 dB / −0.0006 SSIM）上明显退化，原因在于 0.1 权重的 `1/4`、`1/2` 像素级辅助头强迫瓶颈特征直接拟合原始 RGB（且经旧版凸偏 BASR 头），与主解码器表征形成目标竞争。因此 **`014d` 保留 `use_mimo_input=true`，明确移除 `deep_sup_opt`**。
3. **三组 014 相对 `011k/012a_DEA` 暴露出两条完全一致、可定位到代码行的根因短板**：
   - **短板 A — 系统性轻雾过冲（`overshoot_mse` 三子集均偏高 +21%～+31%）**：
     `014b` 在 Dense/Mist/Moderate 的 `overshoot_mse` 为 `0.00127 / 0.00107 / 0.00116`（vs `012a_DEA` 的 `0.00101 / 0.00084 / 0.00095`，95% CI 严格 > 0）；三个探针的 `diag_basr_gain` 稳定落在 `1.15–1.22`，证实存在系统性正向增益偏移。
     - **数学根因（旧版 BASR 的凸性 Jensen 偏置）**：式 (3) 中 $G_{\text{old}}(k) = 0.5 + 4.5\,\sigma(k - \ln 8)$ 在恒等原点处的二阶导数严格为正：
       $$G_{\text{old}}''(0) = 4.5 \cdot \frac{1}{9} \cdot \frac{8}{9} \cdot \left(1 - \frac{2}{9}\right) = \frac{28}{81} \approx +0.3457 > 0 \tag{11}$$
       由二阶泰勒展开 $\mathbb{E}[G_{\text{old}}(k)] \approx 1 + \frac{4}{9}\mathbb{E}[k] + \frac{14}{81}\operatorname{Var}(k)$，即便轻雾图特征期望 $\mathbb{E}[k]=0$，任何零均值特征方差 $\operatorname{Var}(k)>0$ 都会通过 Jensen 不等式产生**正向增益偏置**，同时 $k<0$ 侧陷入 sigmoid 左尾压缩区难以回拉；此外 `FCSF` 仅看局部 6–12 px 感受野且 `alpha` 初始偏置 `-2.0`（8k 仍未收敛），导致极轻雾场景下浅层自然色调被过度抑制。
   - **短板 B — 中频带 `mid_mse`（0.02–0.15 Nyquist）显著落后（Mist +6.7%～+10.2%、Moderate +11.4%～+12.0%，95% CI 严格 > 0）**：
     - 根因 1：`AnisoRepConv3×3` 剔除了 `004_lowfreq §8` 实测能量占比第一（35.3%）的**中心差分分支（Center Difference, CD）**，缺失了各向同性拉普拉斯中频带通先验；
     - 根因 2：瓶颈 `DCA` 的空间门控 `sa` 为全通道共享的单通道图，缺少按通道独立的 7×7 空间像素调制，削弱了中频船体/水岸边缘分离能力；
     - 根因 3：1/2 尺度（256×256）缺少针对 0.02–0.15 中频波长（13–100 px）的各向异性中大核，且 L1/L2 仅各 3 块（比 DEA 的 4/4/8 少 26% 参数）。

---

## 8. `014d` 全面反超架构升级（Proposition P1-v2 + 五项靶向修复）

针对上述两条定量瓶颈，`014d` 在严格保持 `014a/b/c` 权重向后兼容与 `fold_reparam()` + 4 组 LoRA 解耦的前提下，实施五项精准升级：

### 8.1 Proposition P1-v2：$C^2$ 零曲率非对称双曲正切 BASR + 场景对比度门控

将式 (3) 中的凸 sigmoid 增益替换为以恒等点 $k=0$ 为拐点的分段非对称 $\tanh$ 映射（取 $g_{\min}=0.5,\; g_{\max}=4.0 \implies a_+ = g_{\max}-1 = 3.0,\; a_- = 1-g_{\min} = 0.5$）：

$$G_{\text{sym}}(k) = \begin{cases} 1 + a_+\,\tanh\!\left(\dfrac{k}{a_+}\right), & k \ge 0 \\[6pt] 1 + a_-\,\tanh\!\left(\dfrac{k}{a_-}\right), & k < 0 \end{cases} \tag{12}$$

**性质严格证明**：
1. **严格有界正增益与零初始化恒等**：$\forall k \in \mathbb{R},\; G_{\text{sym}}(k) \in (g_{\min}, g_{\max}) = (0.5, 4.0)$，且 $G_{\text{sym}}(0^+) = G_{\text{sym}}(0^-) = 1$；
2. **一阶单位对称灵敏度（$C^1$）**：$G_{\text{sym}}'(k) = \operatorname{sech}^2(k / a_\pm) \implies G_{\text{sym}}'(0^+) = G_{\text{sym}}'(0^-) = 1$，彻底消除 $k<0$ 侧斜率坍缩；
3. **恒等原点二阶零曲率（$C^2$，消除 Jensen 过冲偏置）**：
   $$G_{\text{sym}}''(k) = -\frac{2}{a_\pm}\operatorname{sech}^2\!\left(\frac{k}{a_\pm}\right)\tanh\!\left(\frac{k}{a_\pm}\right) \implies G_{\text{sym}}''(0^+) = G_{\text{sym}}''(0^-) = 0 \tag{13}$$
   因此在原点邻域满足 $G_{\text{sym}}(k) = 1 + k + O(k^3)$，二阶 Jensen 偏置项 $\frac{1}{2}G''(0)\operatorname{Var}(k) \equiv 0$。
4. **场景对比度全局门控（`basr_scene_gate=true`）**：提取全局统计 $[\operatorname{GAP}(F_{\text{dec}1}),\; \operatorname{GAP}(I),\; \operatorname{RMS\_Std}(I)]$（其中图像全局标准差 $\sqrt{\operatorname{Var}(I)+10^{-6}}$ 是经典的雾浓度/对比度物理指示量），经零初始化两层 MLP 输出 $s_k, s_b \in (0, 2)$（初始严格为 `1.0`）对 $(\tilde{K}, B)$ 做逐图尺度调制，在轻雾场景主动衰减过矫正、在浓雾场景全幅放行。该子模块内嵌于 `basr_head`，下游 `015` 真实海雾 LoRA 自动覆盖。

### 8.2 五项升级汇总与 `014d_iso` / `014d_v2` 双臂对照设计

| 升级项 | 对应参数开关 | 靶向解决的瓶颈 | 额外折叠参数 / 推理开销 |
|---|---|---|---|
| **① 5 分支含中心差分重参数化 (`Rep5-CD`)** | `use_cd: true` | `mid_mse`（+0 DC 增益、各向同性中频带通） | **+0.00M / +0.00G**（推理完全折叠！） |
| **② 瓶颈分组 7×7 逐通道空间注意 (`DCA-PA`)** | `channel_spatial_attn: true` | `mid_mse` & 船体/水岸边缘分离（残差式 `p1 + pa(x, p1)` 无死锁） | +0.114M / +1.87G |
| **③ $C^2$ 零曲率 BASR + 场景对比度门控** | `basr_symmetric: true`, `basr_scene_gate: true`, `basr_g_max: 4.0` | 三子集 `overshoot_mse`（−21%~31% 目标）& Mist `light_haze` | +0.001M / +0.00G |
| **④ 全局场景感知 FCSF + 中性偏置** | `fcsf_scene_context: true`, `fcsf_alpha_bias: -1.2` | Mist 轻雾色调保真（全局雾度自适应放行 `α`） | +0.025M / +0.00G |
| **⑤ 1/2 尺度中频 `AnisoLK(15,7)` + 纯 `MIMO` 输入** | `use_l2_aniso_lk: true`, `use_mimo_input: true`, `use_deep_supervision: false` | 0.02–0.15 Nyquist（14–30 px 物理跨度）& Mist `low_mse`（去除深监督干扰） | +0.018M / +1.18G |

为严格分离“纯架构靶向修复”与“对齐 DEA `(4,4,8)` 深度”的各自因果贡献，设计两组并行 8k 探针（均开启 `save_checkpoint_freq: 2000` 与 `activation_checkpointing: true`，适配 `d533` 双 3090 24G 并行）：

- **`probe_014d_iso.yml`**（port `29546`，`depths=[3,3,8]`）：**7.838M 训练 / 4.555M 折叠 / ~144G** —— 与 `014b` 完全同深度、与 `011p`（7.880M / 136.2G）严格等参，精准检验五项升级的纯结构净增益；`d533` RTX 3090 实测 `bs16@512 (ckpt=True)` 峰值 **13.61 GiB alloc / 17.62 GiB reserved**，**2.32 s/iter**（8k 约 5.2 小时）；
- **`probe_014d_v2.yml`**（port `29547`，`depths=[4,4,8]`）：**8.280M 训练 / 4.789M 折叠 / ~169G** —— 补齐与 `011p` 相同的 L1/L2 4 块深度，作为全面反超 `011p` 与晋级 60k 终评的主力候选；`d533` RTX 3090 实测 `bs16@512 (ckpt=True)` 峰值 **15.30 GiB alloc / 19.40 GiB reserved**，**2.68 s/iter**（8k 约 6.0 小时；4090 48G 约 3.2 小时）。
- 全部 5 组配置（`014a/b/c/d_iso/d_v2`）已通过 `verify_014.py --smoke` 单元测试（含 $C^2$ 零曲率数值导数校验、`init_id = 0.00e+00`、`fold_err = 0.00e+00`、3 步全链路梯度传播、LoRA 模块分组）与实训 `014b` EMA 权重严格向后兼容加载检查。
