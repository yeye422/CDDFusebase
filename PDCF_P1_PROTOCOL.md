# PDCF P1：冻结特征跨模态可预测性审计

## 1. 目的

P1 只训练双向跨模态预测器，不训练 Refactor、Fusion、Encoder 或 Decoder。本阶段回答：

1. Optical 特征是否能预测部分 SAR Base/Detail；
2. SAR 特征是否能预测部分 Optical Base/Detail；
3. 非线性轻量预测器是否优于固定均值和 `1×1` 线性投影；
4. 预测残余对 C0 输出是否具有不可由置零、打乱或等能量噪声复制的作用。

P1 不宣称 Predictable 是真实共享信息，也不宣称 Residual 是纯互补信息。

## 2. 特征标准化

使用冻结 C0 Encoder 和确定性的训练集中心裁剪，联合统计 Optical/SAR 的逐通道均值与标准差。Base 和 Detail 分别统计：

```text
Base  → Base mean / scale
Detail → Detail mean / scale
```

无效 SAR 像素不进入统计。统计量保存在 P1 checkpoint 中，验证和后续 P2 必须复用，禁止根据验证集重新估计。

## 3. 预测损失

预测损失由四项等权组成：

```text
Optical → SAR Base
Optical → SAR Detail
SAR → Optical Base
SAR → Optical Detail
```

四项分别计算归一化特征上的 Smooth L1，再取平均。目标特征使用 stop-gradient，C0 Encoder 全程冻结。

同时训练参数量更小的双向 `1×1` 线性基线。固定均值基线在归一化空间中为零输出。

## 4. 已完成的自动检查

- 特征统计能够排除 mask 无效像素；
- Base/Detail 和两个预测方向严格等权；
- Predictor 能收到梯度；
- 均值与线性基线指标字段完整；
- zero、spatial-shuffle、batch-swap、RMS-matched noise 干预满足预期约束；
- 原始 Residual 能重建回 C0 特征路径。

## 5. Dry-run

该命令只计算一个统计批次和一个训练批次，不保存训练结果：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_pdcf_p1_predictability_dry_run.cmd
```

## 6. 正式 Predictor 训练

该命令属于正式训练，由用户在 Windows CMD 中执行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_pdcf_p1_predictability_seed42.cmd
```

训练输出：

```text
models/pdcf_p1/predictability_seed42/
  config.json
  history.csv
  best.pth
  last.pth
```

`history.csv` 同时记录 Predictor、线性基线和固定均值基线的 Base/Detail、双方向 NRMSE。

## 7. val200 Residual 干预审计

训练完成后执行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_pdcf_p1_predictability_val200.cmd
```

审计方法：

| 名称 | 操作 |
|---|---|
| `c0` | 原始 C0 输出 |
| `original` | `Predictable + 原始 Residual`，应重建 C0 |
| `zero_both` | 两个模态 Residual 均置零 |
| `zero_optical` | 仅 Optical Residual 置零 |
| `zero_sar` | 仅 SAR Residual 置零 |
| `spatial_shuffle` | 保持值分布，打乱空间位置 |
| `batch_swap` | 在样本之间交换 Residual |
| `rms_noise` | 替换为逐样本等 RMS 高斯噪声 |

输出包含完整 14 项融合指标、逐图结果、相对 C0 的变化、输出 RMS 偏移和固定样本预览。

## 8. P1 判定

进入 P2 前至少要求：

1. Predictor 在完整验证集的 Base/Detail、两个方向上稳定优于固定均值；
2. Predictor 相比 `1×1` 线性基线存在稳定优势，而不只是训练损失更低；
3. train/val 误差差距没有显示明显记忆；
4. `original` 路径数值上复现 C0；
5. 对真实 Residual 的输出响应不能被 spatial-shuffle、batch-swap 或 RMS-matched noise 等价复制；
6. SAR Residual 置零产生的影响具有稳定结构，而不是只改变全局亮度或制造伪影。

val200 只用于快速淘汰。候选通过后还要运行完整 val1000，并预先冻结最小效应量和继续条件。

## 9. 实验登记

正式训练或 val200/val1000 结果生成后，必须新增 `EXPERIMENT_REGISTRY.json` 条目，并执行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
.venv\Scripts\python.exe maintain_experiment_index.py --check --render
```

本阶段不运行 test1000。

