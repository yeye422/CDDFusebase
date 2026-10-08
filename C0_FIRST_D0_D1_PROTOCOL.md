# C0-first：D0 诊断 → D1 训练约束消融

本轮**不更改网络架构**、不更新原 C0 checkpoint，也不将 E1/C1 改动并入 C0。GPU 运行仍需在本地完成。

## D0（无需训练）

执行：`run_d0_c0_first_diagnosis.cmd`

比较 C0、E1-StemOnly、E1-Unfrozen、C0-EncoderOnly 的相同 val24 / val200 图像。

输出：`test_result/reliability_v5/d0_c0_first/val200/{summary.csv,per_image.csv,channel_correlations.csv,metadata.json}`

核心指标：

- Base/Detail 跨模态 Pearson 相关：同时报告有符号均值、绝对均值、有效通道比例，避免相关性抵消；
- 每个中间特征的高频能量占比（特征去均值后，FFT 径向频率 >=0.25 cycles/pixel）；
- 与 C0 对应特征的归一化 L2 漂移和 Cosine；
- 原 Decoder 的 Optical / SAR 单模态重建 MAE/MSE（注意 Decoder 有输入残差连接，不能单独把低重建误差等同于完好分解）；
- 0.02 标准差的归一化 SAR 加性扰动敏感性（**不属于物理 Speckle 模拟**）。

看 D0 的 `S_Detail_drift_rel`、`S_Detail_perturb_rel`、`base_cc_mean`、`detail_cc_mean`、`s_recon_mae` 和其他特征漂移。D0 不能独立证明噪声来源，也不能依赖高低频比例判断散斑真值。

## D1（沿用 C0，不改架构）

**先分析 D0，再训练 D1**，以避免重复无依据的训练。

执行：

```cmd
run_d1_c0_constraints.cmd A0
run_d1_c0_constraints.cmd A1
run_d1_c0_constraints.cmd A2
run_d1_c0_constraints.cmd A3
run_d1_c0_validation.cmd
```

|组别|训练损失|
|---|---|
|A0|C0 Frequency-Minimax|
|A1|Minimax + 0.05 × 归一化相关性分解|
|A2|Minimax + 0.05 × 归一化单模态重建|
|A3|Minimax + 上述两个约束|

约束参考值由**初始化的 C0 checkpoint** 在固定 train-center-crops 前四批计算，随 checkpoint 保存，避免在训练中移动参考值及验证集泄漏。归一化分母下限 1e-3。系数 0.05 只是预注册 Pilot 值，不能当成最优值；若梯度尺度失衡，要先检查日志，再登记新一轮实验。

D1 使用 C0 的 encoder_only、train800、val200、5 epochs、seed42；PatchEmbed 1e-4、共享 Encoder 1e-5，融合与 Decoder 参数冻结。每轮保存 checkpoint。

D1 的代码修正点：在 `run_epoch()` 中 Frequency-Minimax 总损失被构造后，**才叠加** D1 约束。这保证新增分解损失确实进入反向传播。原 C0 默认 D1 权重全 0，保持既有训练行为。

固定评估输出：

- `test_result/reliability_v5/validation_d1_c0_constraints_final/d1_vs_c0_summary.csv`
- `test_result/reliability_v5/validation_d1_c0_constraints_final/d1_vs_c0_per_image.csv`

必须检查 Qabf/FMI/VIF/Nabf、各 epoch loss，以及 D0 特征诊断的变化。若 A1/A2 仅使指标下降，则停止后续架构改造。若能够稳定保护结构，再考虑加入 SAR 特有结构-噪声约束。

**注意**：val200 已多次参与研究选择，不能用作最终独立测试。
