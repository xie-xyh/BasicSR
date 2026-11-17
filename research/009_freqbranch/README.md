# 008 频分处理方向：低频分支 + 门控融合（009a DEANetV7-LowBranch）

> 设计文档。**新会话启动实验请直接读 §6 启动清单逐步执行**，监控与收尾协议见 §5。
> 前置阅读：`research/004_lowfreq/README.md`（LFM 验证）、`research/007_rf_capacity/README.md` §6（30k/60k 调度混杂与预算错配教训）。

## 1. 背景与证据链

本项目 003~007 系列确立的事实：
1. 残差误差 55~61% 集中在 DC~0.02 超低频带（全局色调/残雾），DEA-Net 瓶颈理论感受野仅 ~197px；
2. 004a 的 LFM（FFT 软掩码低频带 + FiLM 调制）是唯一验证有效的结构改进（Dense +0.13dB，DC~0.02 残差 −4.9%）——**"低频全局校正需要独立通路"成立**；
3. 但全局通路只是 2 个 FiLM 调制器，容量太薄；007a 把 LFM 从 2 块加到 4 块为负结果（同预算对照落后 0.11dB、中高频段全面落后）——**简单堆叠调制器不是加深通路的正确形式**；
4. 文献侧：OctaveConv（ICCV 2019）与 Frequency-Content 双流去雾（I&VC 2023）证明"高/低频分路处理"可行，且其核心价值是**低频在低分辨率下处理（效率 + 天然全局感受野）**。

009a 的假设：**把 LFM 的 FiLM 调制器升级为有容量的低频处理分支**（分频 → 粗网格处理 → 门控融合），是"低频通路加深"的正确形式。

## 2. 模块规格（`basicsr/archs/deanet_v6_arch.py`）

与 004a 的唯一变量：level3 的两个 LFM（位置 [2,5]）替换为 LowFreqBranchBlock。

```
x（128ch，1/4 分辨率，level3 主链特征）
  ├─ 主通路：x（全频保留，不做硬分流，003d 教训）
  └─ 低频分支：
       rFFT(x) × 软掩码 M(cutoff=0.08, edge=0.04) → irfft → x_L   # 与 LFM/FreqDA 同一掩码约定
       x_L → adaptive_avg_pool 到 32×32
           → [3×3 conv 128→128 + GELU] × 2 + 3×3 conv（末层零初始化）
           → bilinear 上采样回 (H,W) → y_L
  融合：g = sigmoid(conv3x3(cat[x, y_L]))（256→128）
        out = x + g ⊙ y_L        # 加性门控，非 (1−g) 加权，更保守
```

- **32×32 网格上 3×3 卷积 ≈ 全局感受野**（128px 特征图上的 197px 瓶颈感受野问题在此分支不存在）；
- **末层零初始化 → 整块恒等**：y_L=0 时 out=x，网络起点严格等于无分支 DEA-Net（单测 max|out−in| = 0）；
- 参数量：V6 总 9.265M（004a 7.880M；新增 1.476M，其中门控卷积占大头 2×0.295M）；
- 显存：bs12 @512px 实测峰值 27.3 GiB、0.86 s/iter → bs20 估 ~46 GiB（与 004a 的 44.6 同量级，**启动前必须实测**）。

## 3. 训练配置（`options/train/DEANet/train_DEANetV7_009a_lowbranch.yml`）

与 004a 严格一致：数据（Mixed 过滤 20399 对、512px、bs20）、loss（L1 1.0 + FFT 1.0 + Grad 0.2）、lr 1e-4、EMA 0.999、seed 0、**`periods=[60000]` / total_iter 60000 / val_freq 2000 / save_checkpoint_freq 1e3**。
（调度一致性是同期对比有效的前提——007a §6.2 教训：改调度长度会使同期对比失效；save 1e3 保证任意 iter 都有同预算对照 checkpoint。）

预期成本：60k × ~1.2 s/iter ≈ 20~23h 单卡 48G。

## 4. 判据与止损

1. **主指标**：Dense-test 同期 PSNR > 004a 轨迹（参照表 `research/007_rf_capacity/ref_004a_dense_trajectory.csv`），终值目标 > 24.86；Mist/Moderate 参照 30.98/28.02；
2. **机理**：残差频谱 DC~0.02 绝对能量 ≤ 004a 同预算 checkpoint 的值（004a@60k 为 236.6M；中途分析务必用 004a 同 iter checkpoint 对照，007a §6.6 的预算错配教训）；
3. **止损**：@10000 Dense 同期低于 004a 同期（21.8674）→ 判负终止；门控 g 恒 ≈0（分支学不动）→ 同样终止，说明低频信息对主干冗余；
4. 合理预期 +0.1~0.2dB（LFM 家族量级），不期待数量级突破。

## 5. 收尾评估协议

1. 全量评估：`CUDA_VISIBLE_DEVICES=<gpu> python basicsr/test.py -opt research/009_freqbranch/test_009a_dense.yml`（模板已含三子集、params_ema、net_g_60000）；
2. 频谱分析：复制 `research/007_rf_capacity/tools/residual_spectrum_007.py` 为 `residual_spectrum_009.py`，MODELS 精简为 hazy/003c/004a@60k/009a（009a 结果目录 `results/009a_dense_eval/visualization/OWH-Dense-test`，文件名模板 `{id}_009a_dense_eval.png`），OUT_DIR 改为 `research/009_freqbranch`；
3. 结论写入本文 §8（指标表 + 频谱表 + 是否达到判据 + 后续建议），中文 commit（不 push）。

## 6. 新会话启动清单

按顺序执行（工作目录 `/home/yxy/xyy/Learn/BasicSR`，先 `conda activate basicsr`）：

1. **GPU 检查**：`nvidia-smi`，选空闲 48G 卡（GPU0/1 可能被其他用户占用，勿挤占）；确认编号 009a 未被占用（`git log --oneline | grep 009a` 应只有本文档相关提交）；
2. **bs20 显存探测**（bs12 已测 27.3 GiB，bs20 未实测，必须补测）：
   ```bash
   python - <<'EOF'
   import torch
   from basicsr.archs.deanet_v6_arch import DEANetV7
   net = DEANetV7(base_dim=32, branch_res=32).cuda(); net.train()
   opt = torch.optim.Adam(net.parameters(), lr=1e-4)
   torch.cuda.reset_peak_memory_stats()
   x = torch.rand(20, 3, 512, 512).cuda()
   for _ in range(3):
       opt.zero_grad(); out = net(x); out.abs().mean().backward(); opt.step()
   print(f'peak {torch.cuda.max_memory_allocated()/2**30:.1f} GiB')
   EOF
   ```
   若峰值 >48 GiB：把配置 `batch_size_per_gpu` 降为 16 并在本文档记录（注意 bs 改动是混杂变量，判读时注明）；
3. **启动训练**：
   ```bash
   CUDA_VISIBLE_DEVICES=<gpu> HF_ENDPOINT=https://hf-mirror.com nohup python basicsr/train.py \
       -opt options/train/DEANet/train_DEANetV7_009a_lowbranch.yml > experiments/009a_nohup.log 2>&1 &
   ```
   启动后 10 分钟检查 iter 100 的 loss（l_pix ~0.14 量级）与显存；
4. **监控**：运行 `python research/006_hihaze/tools/monitor_006a.py --log experiments/009a_nohup.log --csv research/009_freqbranch/metrics_009a.csv --total 60000 --tag 009a`（幂等解析验证点）；建议挂 30 分钟 cron 巡检，判据/止损按 §4，同期参照表在 `research/007_rf_capacity/ref_004a_dense_trajectory.csv`；
5. **完成后**按 §5 收尾。

## 7. 验证记录

- 2026-11-17 单元测试：512²/536×800 fwd+bwd 通过；LFB 零初始化恒等（max|out−in| = 0）；V6 总参数量 9.265M。
- 2026-11-17 显存/速度探测：bs12 @512px 峰值 27.3 GiB、0.86 s/iter（bs20 待启动时补测）。
- 2026-11-17 冒烟训练（`smoke_009a.yml`，合成数据 6 iter）：dataset→model→L1+FFT+Grad→save→validation 全链路通过，冒烟产物已清理。

## 8. 结果与结论

（待训练完成后填写）

## 9. 后续方向（视结果）

- 有效 → 008b：branch_res 消融（16/64）或 n_convs 消融（2/5）；与 005 真实域适配组合（以 V6 替换 004a 骨架）；
- 无效/负 → 频分方向封顶，转向 level1/2 加块（DEA-Net 消融 +2.52dB 的容量杠杆，需 Octave 式低频降采样腾显存）或 GCANet 式瓶颈空洞卷积。
