# 011h：Wavelet U-Net 容量正式对照

- 控制组：`011h_control_WaveletUNet_OWHscreened_60k_B20G1`，宽度 32/64/128/256。
- 候选组：`011h_candidate_WaveletUNet_OWHscreened_60k_B20G1`，宽度 64/96/160/256。
- 两组从头训练 60,000 步，沿用 v2 数据、随机种子、batch 20、损失和调度。训练期间只评估内部验证图。
- 四份数据清单是 v2 筛查结果的原样副本；`sha256.json` 记录配置和清单校验值。原图、权重和逐图预测未包含在这个目录中。

在仓库根目录先做预检：

```bash
bash research/011_wavelet_unet/tools/run_confirmatory.sh \
  research/011_wavelet_unet/confirm_011h \
  . --check
```

按仓库约定同步实验编号与配置后，将最后一个参数改成 `--run`。两组正常完成 60,000 步后才读取最终 EMA 做官方测试。结果写入 `official_60000`。
