# 连续线结构引导的 Detail 修正门控协议

## 目的

前一轮 `Detail-only SPD + M_L` 将线结构候选加入去噪训练损失，SAR 边缘指标仅有
微小改善，融合 NAB/F 反而恶化约 1.06%。本实验不再训练 SPD，而是检验结构候选
是否更适合作为融合阶段的局部决策证据。

## 推理期结构候选

融合阶段没有 GTruth，因此使用当前无 Mask Detail-only SPD：

```text
M_raw   = FSE(SAR)
M_clean = FSE(SPD_denoised_SAR)
M_joint = sqrt(M_raw * M_clean)
M_high  = clamp((M_joint - 0.5) / 0.5, 0, 1)
```

随后删除短碎片，只保留满足以下条件的连通结构：

- 面积至少 32 像素；
- 最大空间跨度至少 16 像素；
- 长宽比至少 2.0，或者面积至少 128 像素。

这些条件只定义连续结构候选，不代表道路或建筑语义标签。

## Detail 门控

设原始 Detail 为 `D_raw`，SPD 修正后的 Detail 为 `D_clean`：

```text
D_candidate = D_clean + p * M_line * (D_raw - D_clean)
p in {0, 0.25, 0.5, 1.0}
```

- `p=0`：当前无 Mask Detail-only SPD，作为严格基线；
- `p>0`：只在可信连续线结构处恢复部分原始 Detail；
- Base、Encoder、Fusion 和 Decoder 均不改变；
- 不训练任何参数。

## 验证命令

```cmd
cd /d "D:\#imageInfusion\CDDFuse"
run_spd_line_gated_detail_val1000.cmd
```

只允许在 QXS validation1000 选择门控强度，不使用 test1000。

## 判定标准

候选必须相对 `p=0` 同时满足：

1. NAB/F 不恶化；
2. QAB/F 或 FMI 出现可解释增益；
3. VIF、SSIM、MS-SSIM 不出现系统性下降；
4. Mask 覆盖率和保留质量可审计。

若所有 `p>0` 都不能改善，说明“恢复原始 Detail”不是正确方向，应转向学习式选择性
注入或直接使用可靠性抑制，而不是继续调节 Mask 阈值。

## validation1000 结论

三种 `p>0` 均未通过判定标准。随着恢复强度增加，FMI 与 MS-SSIM 略有上升，
但 NAB/F 单调恶化，PSNR、SSIM、VIF 整体下降。`p=0.25` 的变化已接近数值噪声，
且只有 41.1% 图像的 NAB/F 改善。

正式验证集的过滤后 Mask 平均非零覆盖率为 10.47%，平均保留 35.9 个连通结构，
仍明显高于冒烟样本。结果说明即使进行连续性过滤，“在候选线结构处恢复原始 Detail”
仍会重新引入伪边缘。这条手工门控路线停止，不进入 test1000。
