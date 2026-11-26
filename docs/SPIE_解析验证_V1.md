# SPIE 解析融合验证

本轮在锁定的 QXS-SAROPT val1000 上比较四种全图融合方法。

| 组别 | 输出 |
|---|---|
| P0 | 光学 Base 锚定 LWPD E2，从头训练的第 16 轮归档 |
| P1 | 有效 SAR 像素取光学和 SAR 等权平均，其余像素取光学 |
| P2 | SPIE 强度参考场 \(B\) |
| P3 | 以同一个 \(B\) 和有向梯度场求解的筛泊松图像 |

P0 复用已按现行评价器计算的同场景逐图指标，脚本核对 checkpoint、清单、原始元数据及场景顺序。

## 固定算子

首版参数在评测前固定。P2 在有效 SAR 区域将光学亮度与掩码归一化高斯平滑后的 SAR 按 0.75 和 0.25 混合。SAR 平滑尺度为 0.8 像素，有效支持不足 0.95 时采用光学亮度。

P3 以光学前向差分为锚。SAR 梯度取 0.8 像素尺度，使用 1.6 像素尺度检查同源梯度的方向和幅值稳定性。SAR 混合权重由双尺度稳定度及相对幅值决定，上限为 0.6。光学响应幅值至少达到 2/255 且两源方向近似反向时，SAR 梯度沿光学极性定向。SAR 有效支持不足时沿用光学梯度。

求解 `αI + DᵀD`，右端为 `αB + Dᵀv`。`α = 0.25`，`D` 为右侧和底部出域分量为零的前向差分。正交 DCT-II 对角化该离散系统。求解残差按未裁剪浮点结果计算，评价统一将输出裁剪至 `[0, 1]` 后转换为 uint8。[筛泊松重构的频域分析](https://grail.cs.washington.edu/projects/screenedPoissonEq/)

## Windows CMD

```cmd
cd /d D:\#imageInfusion\CDDFuse
"D:\anaconda\envs\lightglue_env\python.exe" -u run_spie_analytic.py val --device cuda
```

`smoke` 子命令在一景上检查 P0 归档和解析求解。`val` 将新结果写入 `test_result\spie_analytic_v1\val1000`，保存 P1–P3 的灰度、RGB、三联图及四组 14 项逐图指标。`paired_comparison.csv` 报告四组共同有限样本上的均值、样本数和 95% 配对区间，NAB/F 的 `candidate_minus_control` 为负时表示改善。`p3_solver_diagnostics.csv` 记录求解残差、P3 对 P2 的改动及裁剪比例。完整评测后登记独立比较组。

先读 P3 对 P2 的配对差，判断加入有向梯度场后的净收益，再读 P3 对 P0 的差值，确认解析方法在现有深度基线下的优势和代价。P3 对 P2 衡量的是首版梯度构造与重构的联合增量。结果将用于决定是否进入学习式设计。
