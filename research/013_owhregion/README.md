# 013 — OWHRegionUNet：自研架构主线

> 设计文档与实验记录。013a 的架构设计与实现细节见 `experiments/013a_owhregion_20261120/README.md` 与快照内 `basicsr/archs/owh_lfm_unet_arch.py`；架构已合入主仓库 `basicsr/archs/owh_lfm_unet_arch.py`（参数化 widths/depths，默认值与 013a 快照同构）。

## 1. 定位

OWHRegionUNet 是完全自研的四尺度去雾主干（RestorationBlock 门控块 + HazeConditionedFusion 分频带跳连增益 + RegionContext 区域注意力 + 瓶颈 LFM + DetailRefinement 条形核），7.537M 参数（与 DEA-Net 7.88M 同量级），作为脱离 DEA 骨架的创新主线。

- 013a：基线 60k（widths 48/96/176/320，与 011p 同协议同配方），2026-11-20 启动于 GPU1 快照，运行中；
- 013b：容量探针（widths 64/128/224/320，9.332M），验证"同步骤 −2 dB 差距是否为容量不足"。

## 2. 013b 探针设计（2026-11-21）

- 单变量：仅 widths（瓶颈 320 不动，浅中层加宽，沿用 011f 的位置选择）；数据/种子/配方与 013a/011p 逐字段一致（v2 清单、seed 0、batch 16、lr 2e-4、warmup 500、L1+FFT+Grad、60k 余弦的前 8k）；
- 判据：013b@8k 对照 013a@8k（Dense 19.49 / Mist ~24 参照训练日志）；领先 ≥0.3 dB → 容量假设成立，进 60k；<0.1 → 容量不是瓶颈，回到结构归因；
- 显存：bs16@512 + 激活重计算峰值分配 30.04 GiB（013a 为 22.71），3.07 s/iter（013a 2.08）；
- 显存注意：不加 checkpointing 时 bs16@512 峰值超 47 GiB（OOM），**本架构必须开 activation_checkpointing**。

## 3. 实验记录

| 时间 | 事件 |
|---|---|
| 2026-11-21 09:35 | 013b 加宽探针启动（GPU0），对照 013a 同步骤轨迹；013a 继续在 GPU1 跑 60k（22k/60k） |
| 2026-11-21 09:37 | 013b SwanLab 同步：[probe_013b_wide](https://swanlab.cn/@xieyh/dehaze/runs/tl3srtbu) |
| 2026-11-21 14:15 | **GPU0 硬件故障**：nvidia-smi 报 "GPU requires reset"（utilization 读数 N/A），013b 在 iter 2,900 后日志停摆、进程空转，已终止；训练状态保全在 2000.state。013a（GPU1）不受影响。013b 等 GPU0 恢复或 013a 完成后从 2000 续训（--auto_resume） |
