# M_L 线结构 Soft Mask 协议

## 定义

分别从含噪 SAR 与多时相参考提取全局校准的连续线结构响应：

```text
M_L_noisy = FSE(Noisy)
M_L_reference = FSE(GTruth)
M_L_joint = sqrt(M_L_noisy * M_L_reference)
M_L_high = clamp((M_L_joint - 0.5) / 0.5, 0, 1)
```

`M_L_joint` 用于审计原始结构响应；正式训练候选使用高置信重映射后的
`M_L_high`。二者都是结构候选权重，不是真实道路或建筑标签。

采用高置信重映射的原因是：原始连续响应在城市、农田等纹理区域覆盖过宽，
会把弱纹理和散斑团块也纳入结构保护；重映射保留原始响应大于 0.5 的部分，
并保持连续权重，避免退化为硬二值标签。

## 完整准备命令

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_prepare_spd_line_masks.cmd
```

该命令执行：

1. 在训练集 Noisy+GTruth 上抽取512对完成全局校准；
2. 为1500对训练图生成 Noisy、Reference、Joint Mask；
3. 为100对验证图生成相同Mask；
4. 输出前24对预览及逐图审计统计。

输出目录：

```text
D:\#imageInfusion\datasets\SAR despeckling filters dataset\derived_masks_v1
```

训练时使用：

```text
derived_masks_v1\train\line_joint_high
derived_masks_v1\validation\line_joint_high
```

如果原始 Mask 已经生成，可直接复用已有校准参数，重新生成高置信 Mask：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_generate_spd_high_line_masks.cmd
```

该命令会重新生成训练集 1500 对和验证集 100 对的 Mask、统计与四联预览，
不会启动网络训练。

## 审计文件

```text
train\summary.json
train\per_image_stats.csv
train\previews\
validation\summary.json
validation\per_image_stats.csv
validation\previews\
```

重点检查：

- `joint_gt_0p5` 与 `joint_gt_0p7` 覆盖率；
- `high_nonzero_fraction`、`high_gt_0p5` 与 `high_mean`；
- `noisy_reference_correlation`；
- `perturbation_correlation` 与 `perturbation_mae`；
- `flat_joint_gt_0p5` 与 `flat_high_gt_0p5`；
- 连通区域数量与面积；
- 道路、岸线和建筑边缘的连续性；
- 水面、农田纹理和散斑团块的误响应。

Mask审计通过前，不启动 Detail-only + M_L训练。
