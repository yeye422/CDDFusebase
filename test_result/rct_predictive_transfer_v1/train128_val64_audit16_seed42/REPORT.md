# RCT结构预测增量与冻结成图验证

本报告由既定协议自动生成。没有训练RCT融合网络；E2权重全程冻结。

预测验证场景：64；请求成图场景：16；成功校准条件：384。

预测主读出使用排除固定人工16图后的验证场景；仍来自已有pilot，不是地理独立测试集。

## 光学结构预测

自动准入：`not_established`。

| 对照 | 共同场景 | 相对有利变化 | 改善场景 | 有利差95%区间 |
|---|---:|---:|---:|---|
| constant | 48 | 0.143908 | 47 | [0.109011, 0.181761] |
| self | 48 | 0.000465426 | 31 | [8.6497e-05, 0.000713424] |
| self_wide | 48 | -0.0118117 | 5 | [-0.0134918, -0.00721525] |
| rolled | 48 | 0.00070781 | 32 | [0.000278994, 0.000938659] |
| swapped | 48 | 0.000649386 | 33 | [0.000241043, 0.000879685] |

## SAR结构预测

自动准入：`not_established`。

| 对照 | 共同场景 | 相对有利变化 | 改善场景 | 有利差95%区间 |
|---|---:|---:|---:|---|
| constant | 48 | 0.0111073 | 32 | [0.00657006, 0.0157667] |
| self | 48 | 0.0022599 | 32 | [0.0011015, 0.00346239] |
| self_wide | 48 | 0.000463482 | 27 | [-0.000820917, 0.0017627] |
| rolled | 48 | 0.00232102 | 33 | [0.00123504, 0.00342701] |
| swapped | 48 | 0.00233371 | 32 | [0.00117629, 0.00353603] |

## 光学条件的成图传递

自动判断：`not_established_for_this_frozen_operator`。

只以预定正向插值作为主判断；负向条件完整报告，不事后挑有利符号。

| 检查 | 结果 |
|---|---|
| prediction | not_established |
| beats_amplitude_and_relation_controls | False |
| image_quality_non_decreasing_means | True |
| opposite_source_retained | True |
| enough_manual_scenes | True |
| manual_signal | False |
| negative_path_guard | True |
| all_requested_scenes_excited | True |
| sufficient_quantized_effect | True |

| 参照 | 共同场景 | 本源候选传递差 | 有利差95%区间 |
|---|---:|---:|---|
| e2 | 16 | 0.000147414 | [9.34419e-05, 0.00020443] |
| optical_rolled_plus | 16 | 1.74778e-05 | [-1.68125e-05, 4.756e-05] |
| optical_swapped_plus | 16 | -5.36231e-06 | [-3.63761e-05, 2.41058e-05] |
| optical_shuffled_plus | 16 | 5.81544e-05 | [1.52351e-05, 0.000104734] |
| optical_native_plus | 16 | -3.29546e-05 | [-5.79548e-05, -1.09219e-05] |
| optical_uniform_plus | 16 | 2.85243e-05 | [-7.20041e-06, 6.34266e-05] |

## SAR条件的成图传递

自动判断：`not_established_for_this_frozen_operator`。

只以预定正向插值作为主判断；负向条件完整报告，不事后挑有利符号。

| 检查 | 结果 |
|---|---|
| prediction | not_established |
| beats_amplitude_and_relation_controls | False |
| image_quality_non_decreasing_means | False |
| opposite_source_retained | True |
| enough_manual_scenes | True |
| manual_signal | False |
| negative_path_guard | True |
| all_requested_scenes_excited | True |
| sufficient_quantized_effect | True |

| 参照 | 共同场景 | 本源候选传递差 | 有利差95%区间 |
|---|---:|---:|---|
| e2 | 16 | -0.000113428 | [-0.000161142, -7.00313e-05] |
| sar_rolled_plus | 16 | -6.72415e-06 | [-1.69705e-05, 2.54139e-06] |
| sar_swapped_plus | 16 | -9.20892e-06 | [-2.36883e-05, 3.02186e-06] |
| sar_shuffled_plus | 16 | -6.7167e-05 | [-9.77669e-05, -4.00272e-05] |
| sar_native_plus | 16 | 1.96546e-05 | [1.01735e-05, 3.23403e-05] |
| sar_uniform_plus | 16 | -5.70975e-05 | [-7.86209e-05, -3.81915e-05] |

## 解释与结果入口

- 预测头为固定HOG的线性Ridge，额外输入已在训练集上对本源上下文做无标签残差化。
- 成图算子为带条件的源Detail插值；正结果只是局部冻结可行性，负结果不否定所有非线性预测器或RCT算子。
- 同一场景全部来源、条件和符号匹配最终浮点图像MAE；不使用质量指标挑强度。
- 梯度候选、走廊连续性和人工小样本均不是完整结构真值。请复核previews与gray中的实际图像。
- 14项逐图指标、均值/标准差及共同样本数在metrics；非有限值保留且按共同交集统计。
- 完整配对统计见[prediction_comparisons.csv](prediction_comparisons.csv)和[transfer_comparisons.csv](transfer_comparisons.csv)。
- 效应不足的场景：无。这些场景不能解释为恒等安全回退。
