# 006 Hihaze 模块移植：SDA 雾密度适配器（006a 冻结探针）

> 设计文档。前置阅读：`research/004_lowfreq/README.md`（004a 基线与结论）、`research/005_sim2real/README.md`（冻结骨架+适配器训练范式）。
> 移植来源：Hihaze（Chen et al., arXiv:2610.25832, 2026-10，单步扩散高分辨率真实去雾），官方代码仓目前仅有 README，模块按论文公式自行实现。

## 1. 背景与动机

real-test（500 张真实雾图）目检发现：模型对**空间非均匀雾**的处理是短板——同一图内浓雾区残雾未除尽、薄雾区又有过度拉对比的风险。Hihaze 针对同一问题提出的 SDA（Spatial Degradation Adapter）用一个小卷积编码器从有雾原图提取**多尺度雾密度特征**，显式监督后注入解码器，在四个真实基准上取得 SOTA（Dense-Haze PSNR 超基线 3.62dB）。

004 的结论（§7.3）指出"低频残差仍有 54.6% 占比，下一步可做显式多尺度低频分支"——SDA 本质就是**带显式监督的多尺度退化分支**，与本项目自有路线图重合。

## 2. 006a 定位与假设

- **定位**：低成本冻结探针（同 005 FreqDA 范式），回答一个问题——"显式雾密度信息对确定性 DEA-Net 主干有没有用"。给出收益下界；有效再上 006c（解冻/重训逼近上界），无效则低成本止损。
- **假设**：注入密度特征后，Dense-test PSNR 相对 004a（24.86）有可归因的提升；SDA 特征图应呈现与雾分布的空间对应（可视化验证）。

## 3. 结构设计（`basicsr/archs/deanet_v4_arch.py`）

```
有雾图 I ──► SDAEncoder(轻量conv) ──► {F_512(16ch), F_256(32ch), F_128(64ch)}
                │ head×3（1×1 conv → 3ch）监督 T_d/T_r/T_s（仅训练期）
                ▼
DEANetV3-LFM 骨架（冻结，加载 004a params_ema，strict）
  ... mix2(x_down2, x_up1) ──► [P1] x += zero_conv(cat[x, F_256])   (64ch, 1/2 res)
  ... 4×DEBlock(up_level1) ──► [P2] x += zero_conv(cat[x, F_512])   (32ch, full res)
  ... up3 ──► out ──► [OUT] out += zero_conv(cat[out, F_512])       (3ch)
```

- **插入点选择**（设计讨论见会话记录 2026-11-14）：P1 是细节重建起点（雾浓处需更强重建）；P2 做薄雾区负向门控（防过锐化放大噪声，real-test 小图教训）；F_128（1/4 尺度）本期只监督不注入，预留给二期 P3（与 LFM 的 γ/β 预测器联动）。
- **不注入 encoder 侧**：encoder 特征经 mix1/mix2 skip 已流入 decoder，注入路径不清晰且多一个消融变量。
- **零初始化**：三个注入点输出卷积全零 → 初始输出与 004a 严格恒等（单元测试 max|out−base| = 0.00e+00，512²与 536×800 两规格通过）。
- **新增可训练参数 0.080M**（总 7.960M）。

## 4. 监督设计（`basicsr/models/sda_model.py`，Hihaze App. A.2.1 移植）

Φ(x) = AvgPool₁₁(Sobel 幅值) + 局部对比度（11×11 局部标准差），灰度通道平均后逐像素标量图。

| 目标 | 公式 | 范围 | 含义 |
|---|---|---|---|
| T_d | 1 − (Φ(I)+ε)/(Φ(I_c)+ε) | [0,1] | 结构衰减（≈雾密度） |
| T_r | (Φ(I_c)−Φ(I_b))/(Φ(I_c)+ε) | [−1,1] | 冻结主干欠修/过修方向 |
| T_s | sigmoid((Φ(I_c)−τ)/κ) | (0,1) | 结构置信 |

ε=1e-3、τ=0.03、κ=0.01（论文值）。三尺度 head 输出与 AvgPool 到对应分辨率的目标做 Smooth L1，求和后 ×1.0（`sda_opt.loss_weight`）。I_b 为冻结主干输出（训练期 no-grad 二次前向获得）。

与 Hihaze 的两处差异（有意为之，记录备查）：①Hihaze 分两段顺序训练（先 encoder 后分支），006a 为联合单段训练以匹配单次探针流程；②注入用 cat+1×1 conv 替代论文未公开细节的 "convolutional adapter"。

## 5. 实验配置（`options/train/DEANet/train_DEANetV4_006a_sda.yml`）

与 004a 严格对齐的变量：数据（OWH Mixed filter_min_size→20399 对、512px、bs20、同增广）、loss 权重（L1 1.0 + FFT 1.0 + Grad 0.2）、EMA 0.999、验证三子集、seed 0。
改动项：网络 DEANetV4（含骨架加载）、lr 1e-3（新模块，同 005 适配器惯例）、total_iter 20k、val_freq 1000（加密观察早期信号）、新增 sda_opt 辅助监督。

**判据**：
1. 主指标：Dense-test 全量 PSNR/SSIM > 004a（24.86/0.9006），且训练内验证曲线单调收敛；
2. 机理：SDA head 的 T_d 通道可视化（`tools/export_sda_maps.py`，待写）与雾分布空间对应；残差频谱工具复验（若有效，看中低频段能量变化）；
3. 止损线：@10k 时 Dense PSNR 仍 ≤ 004a 同期水平（24.7 量级）且 l_sda 未收敛 → 判无效，转 P3/LFM 联动或其他方向。

## 6. 验证记录

- 2026-11-14 单元测试：512²与 536×800 前向+反向通过；零初始化严格恒等（max|out−base| = 0）；可训练参数 0.080M；冻结参数无梯度。
- 2026-11-14 冒烟训练（合成数据 6 iter 全链路，`smoke_006a.yml`）：dataset→model→L1+FFT+Grad+SDA→save→validation 全部通过，l_sda 0.62→0.48 正常下降；冒烟产物 experiments/smoke_006a_sda 已清理。
- 2026-11-14 显存/速度探测（512px bs20 fwd+bwd+Adam）：峰值 21.6 GiB、0.99 s/iter（004a 为 44.6 GiB、1.28 s/iter）→ 20k ETA ≈ 6~7h（含验证）。

## 7. 实验记录

| 时间 | 事件 |
|---|---|
| 2026-11-14 | 设计定稿（P1/P2/OUT 注入点；P3 LFM 联动留二期）；arch/model/config 实现完成，单元+冒烟+显存验证通过 |

## 8. 结果与结论

（待训练完成后填写）

## 9. 后续方向（视结果择一）

- **006c**：解冻微调或从头重训（006a 有效时逼近上界）；
- **P3（LFM 联动）**：用 SDA 的 1/4 尺度密度特征替代 LFM 的 pooled 调制预测器；
- **006b（ASC-lite）**：可靠性监督 CGAFusion（Hihaze ASC 的确定性网络降级移植），独立分支；
- 真实域延伸：006a 权重作为 005 系列自监督域适配的新骨架（替代 004a）。
