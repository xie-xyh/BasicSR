# 010 物理分解预测目标：WaveletDehazeNet（010a）

> 设计文档。前置阅读：`research/004_lowfreq/README.md`（频谱证据）、`research/008_dwt/README.md`（Haar 工具与高频假设证伪）、`research/007_rf_capacity/README.md` §6（协议教训）。
> 方向来源：用户设计——"先恢复底图，再按位置恢复纹理"。本文件同时是**新会话启动实验的操作手册**（§6）。

## 1. 核心思想：改变预测目标的分解方式

不再让网络直接回归 RGB，而是按大气散射模型把预测目标拆成三个物理项，分别用匹配的机制恢复：

```
清晰图 = 底图分支恢复的低频场 + 有界倍率 × 原纹理 + 权重 × 邻域修正
```

与此前失败改动的根本区别：003d/004b/006a/007a 都是在主干上**加机制**（频域滤波/全局注意力/信息注入/堆叠调制器），本设计是**换预测目标的参数化**——LFM 证明有效的"结构性归纳偏置"象限。

物理推导（模块二的合法性）：t、A 局部恒定时，I(x)−I(y) ≈ t·(J(x)−J(y))（雾幕项相减抵消）。Haar 细节系数是局部差分的线性组合 ⇒ **D_hazy ≈ t·D_clear，倍率 ≈ 1/t**。网络的任务从"生成纹理"降维为"估计一个标量场"。

## 2. 架构（`basicsr/archs/wavelet_dehaze_arch.py`）

```
有雾 RGB
  └─ HaarDWT ×2（复用 008_dwt 的 HaarDWT/HaarIWT，正交无损）
       LL2 (3ch, 1/4 res) + detail2 (9ch, 1/4) + detail1 (9ch, 1/2)

模块一 BaseBranch（feat=32）：
  输入 cat[LL2, 每方向纹理能量(3ch)]（6ch）
  → conv3×3 ×2 → GAP 全局上下文广播（雾幕是全局场，局部 RF 不够）
  → base_rec = LL2 + head(f)；F = f（空间特征，供纹理分支）

模块二/三 TextureBranch（粗/细两个尺度各一）：
  strength = avgpool3x3(|D|)
  log_gain = gain_net(cat[D, F, strength]) → avgpool3x3 平滑
  gain = exp(clamp(log_gain, −1, 2))        # 有界 [0.37, 7.39]，防噪声放大
  corr = DEConv(D) × 0.1                     # corr_scale 固定小系数，防喧宾夺主
  w = sigmoid(conv(cat[F, strength]))
  D_rec = D × gain + w × corr
  last_gain 保留 detach 副本供机理分析（倍率图 ≈ 1/t）

IWT 两级重建回 RGB（mod-4 测试填充由 DehazeModel.test 处理）
```

参数量 **0.136M**（vs DEANet 7.88M）；bs20 @512px 实测 **3.3 GiB / 0.05 s/iter**——60k 全量 ETA **~1h**（含验证），试错成本极低。

**容量风险（明确记录）**：0.136M 参数的容量上限是主要未知项。若终值显著欠拟合（Dense < 24），优先加宽 `feat`（32→64）/ 加深分支，而不是改结构——所有相关旋钮已参数化。

## 3. 训练配置（`options/train/WaveletDehaze/train_WaveletDehazeNet_010a.yml`）

与 004a 严格一致：数据（Mixed 过滤 20399 对、512px、bs20）、loss（L1 1.0 + FFT 1.0 + Grad 0.2）、lr 1e-4、EMA 0.999、seed 0、`periods=[60000]`、val_freq 2000、save_checkpoint_freq 1e3。

新架构无需逐点同期判读（轨迹不同属正常），以**终值 + 频谱 + 目检**为主判据。

## 4. 判据与止损

1. **主指标**：Dense-test 终值 PSNR/SSIM 对比 004a（24.86/0.9006）与 003c（24.73/0.9007）；新架构首版能进 24.5+ 即证明分解方向有生命力；
2. **机理验证（三个现成抓手）**：
   - 残差频谱：DC~0.02 与 >0.50 应同时改善（设计的直接承诺）；
   - 倍率图 vs DCP 透射率图的相关性（`last_gain` 可直接导出）；
   - real-test 小图目检：重点看 Haar 块状伪影与噪声放大（有界倍率 + 平滑的保护是否生效）；
3. **止损**：@20000 Dense 仍低于 003c 同期、或出现 smoothing 救不回的块状伪影 → 停。

## 5. 收尾评估协议

1. 全量评估：`python basicsr/test.py -opt research/010_wavelet/test_010a_dense.yml`（模板已含三子集、params_ema）；
2. 频谱分析：复制 `research/008_dwt/tools/residual_spectrum_008.py`，MODELS 精简为 hazy/003c/004a@60k/010a（结果目录 `results/010a_dense_eval/visualization/OWH-Dense-test`，模板 `{id}_010a_dense_eval.png`），OUT_DIR 改为 `research/010_wavelet`；
3. 倍率图导出：hook `net.texture_coarse/texture_fine.last_gain`，与 DCP 透射率图对照；
4. 结论写入 §8，中文 commit（不 push）。

## 6. 新会话启动清单

工作目录 `/home/yxy/xyy/Learn/BasicSR`，先 `conda activate basicsr`：

1. `nvidia-smi` 选空闲卡；编号 010a 已占号（本文件）；
2. 显存已在 bs20 实测 3.3 GiB，无需再探测；
3. 启动：
   ```bash
   CUDA_VISIBLE_DEVICES=<gpu> HF_ENDPOINT=https://hf-mirror.com nohup python basicsr/train.py \
       -opt options/train/WaveletDehaze/train_WaveletDehazeNet_010a.yml > experiments/010a_nohup.log 2>&1 &
   ```
4. 监控：`python research/006_hihaze/tools/monitor_006a.py --log experiments/010a_nohup.log --csv research/010_wavelet/metrics_010a.csv --total 60000 --tag 010a`（幂等）；建议挂 30 分钟 cron，止损按 §4；
5. 完成后按 §5 收尾。

## 6.5 实验记录

| 时间 | 事件 |
|---|---|
| 2026-11-18 14:14 | 010a 启动（GPU0 pid 2607672，日志 experiments/010a_nohup.log）。iter 900：0.08 s/iter（ETA ~1h20m）、l_pix 0.118（从头训练正常起点）；监控 cron 每 30 分钟巡检（含全量评估/频谱/倍率图自动收尾） |

## 7. 验证记录

- 2026-11-18 单元测试：512²/536×800/128² fwd+bwd 通过；参数量 0.136M；倍率图有界且平滑（init 时 gain ∈ [0.91, 1.09]）；HaarDWT/IWT 复用 008_dwt 工具（其重建误差 ~5e-7 已验证）。
- 2026-11-18 显存/速度探测：bs20 @512px 峰值 3.3 GiB、0.05 s/iter。
- 2026-11-18 冒烟训练（`smoke_010a.yml`，6 iter）：全链路通过，产物已清理。

## 8. 结果与结论

（待训练完成后填写）

## 9. 后续方向（视结果）

- 有效但欠拟合 → 加宽 feat（32→64）/ 加深分支（`n_convs` 化）/ 加第三级小波；
- 有效且拟合好 → 与 005 真实域适配组合（物理分解 + 自监督域适配是天然搭档）；倍率图作为透射率估计器可直接输出可解释产物；
- 无效 → 若 DC~0.02 也没改善，说明底图分支全局上下文不足；若 >0.50 没改善而 DC 改善，说明倍率通道无效、问题回到 loss 侧（008 §7）。
