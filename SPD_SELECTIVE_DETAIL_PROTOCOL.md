# Selective-Detail SPD 网络修改协议

## 目标

在不改变现有训练、验证和正式测试数据划分的前提下，新增
`selective_detail` SPD 变体。旧 `base_detail`、`detail_only` 网络及其 checkpoint
保持兼容，不覆盖任何历史结果。

该变体针对现有 Detail-only SPD 的两个结构限制：

1. 原门控只有一个空间通道，同一位置的 64 个 Detail 通道共享修正强度；
2. 原修正上限使用整张特征的单一 RMS，无法适应不同通道的能量差异。

## 网络结构

```text
Frozen B0 Encoder Detail
  -> 原始 Detail + 3/5/9 多尺度高频残差
  -> 1x1 证据投影
  -> 2 x Depthwise Residual Block
  -> 高频残差修正 Delta_D
  -> Spatial confidence x Channel confidence
  -> 逐通道 RMS 10% 有界修正
  -> Clean Detail

Frozen B0 Encoder Base -> 严格恒等旁路
```

修正量在输出前再次经过固定高通投影，避免向 Detail 注入 DC/Base-like 偏移。
所有输出层保持零初始化，因此未训练模型严格复现 B0。

## 参数量

- 原 Detail-only SPD 特征分支：84,289；
- Selective-Detail SPD 特征分支：65,937；
- 减少 18,352，约 21.77%。

两者训练时仍使用相同的独立 Denoising Head，正式 E2 特征融合部署可不携带该 Head。

## 数据与评价不变

- SAR 训练集：原 1500 对 Noisy/GTruth；
- checkpoint 选择：原 `5120_*` 50 对；
- SAR 内部确认：原 `5632_*` 50 对；
- QXS validation：原 manifest 的 `val@0, N=1000`；
- QXS formal test：原 manifest 的 `test@0, N=1000`；
- metric profile：`paper`；
- output policy：`clamp_then_uint8`。

网络变化后的结果必须写入新的唯一目录，不覆盖 Detail-only SPD 的已有结果。

## 执行顺序

以下均为训练或批量评价命令，由用户在 Windows CMD 中执行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_spd_selective_detail_training_seed42.cmd
run_spd_selective_detail_validation.cmd
```

只有 validation1000 通过以下闸门后，才执行正式 test1000：

- NAB/F 低于当前 retrained Detail-only SPD；
- FMI、QAB/F、VIF 任一项相对当前 Detail-only SPD 的下降不超过 0.2%；
- SSIM 不低于当前 Detail-only SPD；
- correction map 不出现大面积饱和或固定纹理。

正式测试命令：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_spd_selective_detail_on_b0_test1000.cmd
```

完成正式评价后，先将新实验登记到 `EXPERIMENT_REGISTRY.json`，然后运行：

```cmd
.venv\Scripts\python.exe maintain_experiment_index.py --check --render
```
