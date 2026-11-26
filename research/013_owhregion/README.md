# 013 — OWHRegionUNet：自研架构主线

> 设计文档与实验记录。013a 架构已合入主仓库 `basicsr/archs/owh_lfm_unet_arch.py`（参数化 widths/depths，默认值与 013a 原快照同构）。正式实验目录 `experiments/013a_owhregion_20261120/` 仅保留最终模型、训练配置、合并训练日志与完整 SwanLab 记录。

## 1. 定位

OWHRegionUNet 是完全自研的四尺度去雾主干（RestorationBlock 门控块 + HazeConditionedFusion 分频带跳连增益 + RegionContext 区域注意力 + 瓶颈 LFM + DetailRefinement 条形核），7.537M 参数（与 DEA-Net 7.88M 同量级），作为脱离 DEA 骨架的创新主线。

- 013a：基线 60k（widths 48/96/176/320，与 011p 同协议同配方），2026-11-20 启动于 GPU1 快照，2026-11-25 完成；
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
| 2026-11-21 09:37 | 013b SwanLab 同步：[probe_013b_wide](https://swanlab.cn/@xieyh/dehaze-probes/runs/jl4vix71) |
| 2026-11-21 14:15 | **GPU0 硬件故障**：nvidia-smi 报 "GPU requires reset"（utilization 读数 N/A），013b 在 iter 2,900 后日志停摆、进程空转，已终止；训练状态保全在 2000.state。013a（GPU1）不受影响。013b 等 GPU0 恢复或 013a 完成后从 2000 续训（--auto_resume） |
| 2026-11-22 00:55 | 机器重启后 GPU0 恢复（admin 复位），013b 从 2000 断点在 GPU0 续训（--auto_resume，新 SwanLab run：[qh12m3bz](https://swanlab.cn/@xieyh/dehaze-probes/runs/vndiwx1f)）；另发现并终止一个 00:35 启动的 013a resume 重复进程（活跃的 resume 由 19:48 启动，已在 45.2k，未受影响） |
| 2026-11-22 01:50 | **事故链与机器状态**：(1) 我在 00:53 误将 013a 活跃训练的 DataLoader worker（pid 156946，fork 进程与父进程同 cmdline）判为重复进程并终止，导致 013a 在 iter ~45,200 崩溃（断点 44,000 保全，损失 ~1,200 步）——责任与教训（fork worker 不能用 cmdline 判重）如实记录；(2) 恢复尝试中发现 GPU0 再次 ERR（013b 二次挂起于 3,100，断点仍为 2,000）；(3) GPU1 虽 nvidia-smi 显示健康，但新 CUDA 上下文创建失败（"No CUDA GPUs are available"，驱动级中毒），整机 GPU 栈需管理员重启/重载驱动。当前无训练进程运行，断点齐全，等机器修复后恢复（013a→GPU1 优先，013b 等 GPU1 空出，不再信任 GPU0） |
| 2026-11-24 21:55 | 机器修复后恢复 013a：从 44,000 断点在 GPU1 续训（修复了我两次失败的恢复尝试留下的归档错位：44,000.state 从 _archived_20261122_014520 搬回活跃目录，按 launch.sh 的环境约定 PYTHONPATH=快照 + cwd=仓库根启动；--force_yml 指定 44000.state 覆盖配置里过期的 36000）。SwanLab：resume run `i7ps5kmf`（后由合并记录取代）。ETA ~8.7h 到 60k。013b 仍等 GPU1 空出（GPU0 不可信） |
| 2026-11-25 06:07 | 013a 完成 60k 训练与最终验证，保留 `net_g_latest.pth` |
| 2026-11-26 18:11 | 将三段有效轨迹合并为单一 SwanLab 记录 [kfn2awlu](https://swanlab.cn/@xieyh/dehaze/runs/kfn2awlu)：100–36,000 取首次训练，36,100–44,000 取第一次续训，44,100–60,000 取最终续训，剔除重复和回滚区间；三条 loss 各 600 点且无缺口，验证指标完整。旧 run `ztx65lc8`、`25k5iuo3`、`i7ps5kmf` 已删除。实验目录同步清理代码快照、中间 checkpoint、原始 TensorBoard 与重试日志，仅保留最终模型、配置、合并日志和合并 SwanLab 本地记录 |
| 2026-11-26 18:30 | 013b 两条 probe 记录迁移到独立 SwanLab 项目 `dehaze-probes`（链接已更新），正式 `dehaze` 只保留 013a；本地 probe 训练与 TensorBoard 分别归入 `experiments/probes/`、`tb_logger/probes/` |
