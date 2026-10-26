# Detail-only SPD + M_L 训练协议

## 研究问题

在保持当前 Detail-only SPD 架构、B0 初始化、数据划分和优化参数不变的条件下，
检验高置信线结构软掩码能否缓解去噪造成的细线与边界损失，并进一步改善融合中的
结构保留与伪边缘控制。

## 唯一实验变量

```text
M_L = line_joint_high
lambda_L in {0.10, 0.25}
```

对照组为已经完成的 `spd_detail_only_seed42`，其 `lambda_L=0`。

三组实验均使用：

- 相同 B0 Encoder 检查点；
- 相同 1500 对训练数据；
- 相同官方验证空间分组；
- 相同随机种子 42；
- 相同 30 epoch、batch size、patch size、学习率；
- Base 分支恒等映射，Encoder 冻结；
- Detail 特征蒸馏权重 0.05；
- 扰动一致性权重 0.1。

## 线结构损失

对去噪输出与近似参考的空间梯度差进行高置信 Mask 加权：

```text
L_line = sum(M_ref * M_L * |grad(S_hat) - grad(G)|)
         / sum(M_ref * M_L)
```

损失按有效 Mask 权重归一化，因此不同图块的 Mask 覆盖率不会直接改变损失尺度。
由于 Encoder 与 Base 均冻结，该损失只更新 Detail SPD 和去噪 Head。

## 正式训练命令

先运行低权重：

```cmd
cd /d "D:\#imageInfusion\CDDFuse"
run_spd_detail_only_ml_w010_seed42.cmd
```

确认训练稳定后再运行高权重：

```cmd
cd /d "D:\#imageInfusion\CDDFuse"
run_spd_detail_only_ml_w025_seed42.cmd
```

## 训练审计

`history.csv` 除原有损失外增加：

- `line_mask_mean`：有效区域内 Mask 平均权重；
- `line_mask_support`：有效区域内非零 Mask 比例；
- `line_active_fraction`：包含至少一个非零线结构像素的 batch 比例。

如果 `line_active_fraction` 很低，不直接提高 `lambda_L`，应先检查随机裁剪是否经常避开
结构区域。当前实验不使用 line-aware crop，以保证与 `lambda_L=0` 基线只有损失项不同。

## 选择规则

先在 SAR 内部验证集比较 PSNR、SSIM、MAE、edge MAE、edge correlation；再在 QXS
验证集进行融合比较。只有验证集通过后才运行一次正式 test1000。

优先要求：

1. edge correlation 提升，且 PSNR/SSIM 不明显下降；
2. QXS 验证集 NAB/F 下降；
3. FMI、QAB/F、VIF 不低于 Detail-only 基线的合理波动范围；
4. 若 0.25 比 0.10 更强但损害信息指标，选择 0.10，不继续扩大权重。

## lambda_L=0.10 训练状态

`spd_detail_only_ml_w010_seed42` 已完成 30 epoch，最佳检查点来自第 25 轮：

```text
checkpoint: models\v21_softmask\spd_detail_only_ml_w010_seed42\best.pth
sha256: 8578608ca8270ca5b86e8337d867c4faedc4f4d88972356364fc82f38913b0e0
```

验证命令：

```cmd
cd /d "D:\#imageInfusion\CDDFuse"
run_spd_detail_only_ml_w010_validation.cmd
```

该命令只使用 SAR 内部 test50 和 QXS validation1000，不使用正式 test1000。

## lambda_L=0.10 验证结论

SAR internal-test50 相比无 Mask Detail-only：edge correlation 提升约 0.27%，
edge MAE 降低约 0.056%，但 PSNR 降低 0.0044 dB，整体变化很小。

QXS validation1000 相比无 Mask Detail-only：QAB/F、SSIM、VIF 仅有极小提升，
FMI 与 MS-SSIM 略降，NAB/F 从 0.091596 上升到 0.092567（恶化约 1.06%）。

因此该版本记为开发集负结果，不进入正式 test1000。暂缓 lambda_L=0.25；若继续线结构
方向，应先解决 Mask 短碎片和损失对伪边缘的保护问题，而不是直接增大损失权重。
