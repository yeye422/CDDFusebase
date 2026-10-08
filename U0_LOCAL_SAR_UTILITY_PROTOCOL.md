# U0：局部 SAR Detail 边际贡献可验证性实验

状态：实验代码和协议已经提交，U0 尚无实测结果。

路线：C0 -> R1（性能假设未通过）-> U0 -> 仅有条件地进入 U1。
不继续直接叠加 R2 SoftMatch、R3 FSE、Anchor 监督或多分支 Router。

## 1. 来自 R1 的证据

C0 val200：FMI 0.657075，QAB/F 0.738678，NAB/F 0.036790，VIF 0.326856。
R1 learned：FMI 0.653702，QAB/F 0.737443，VIF 0.325671。
Invert：FMI 0.658755，QAB/F 0.739642，VIF 0.327741，但 SSIM、MS-SSIM 和 NAB/F 变差。
Mean/Shuffle 通常比 Learned 更接近 C0，当前学习的空间位置尚未证明有效。
增大 R1 正向幅度时 FMI、QAB/F、VIF 下降，而 SSIM/MS-SSIM 上升。

因此：原训练目标与多项融合评价指标存在 trade-off，但不等于“反向 gate 就是真实效用”。

## 2. 核心可证伪问题

在 C0 权重、Optical Detail、Base 路径均固定时，相同强度的局部 SAR Detail 增强与抑制，是否因区域而呈现方向相反且跨扰动强度可重复的效用变化？

定义局部渐变掩码 M_r：

D_f(r,±ε) = DetailFuse_C0[D_o + (1±ε M_r) D_s]

F(r,±ε) = Decoder_C0[I_o,B_f,D_f(r,±ε)]

F_0 = C0 原输出。

在同一 tile ROI 比较 QAB/F、NAB/F、FMI、VIF、SSIM 的 ±ε 输出及 C0 原输出。

一定要区分：

- Direction：Q(+ε) 优于 Q(-ε)，不保证任一方案比 C0 好。
- Benefit：某一方向的 QAB/F 必须真正超过 C0。
- Pareto proxy：QAB/F 提升，且 FMI、VIF、NAB/F 相对 C0 不恶化超过指定容差。

上述判断全是局部无参考代理指标的实验证据，不是 SAR 可靠性标签、感知真值或已验证的 U1 训练标签。

## 3. 固定协议

| 项目 | 设置 |
|---|---|
| 基线 | C0 frequency_minimax_baseline，固定 epoch_005.pth |
| 数据 | QXSLAB_SAROPT，固定 split manifest |
| 发现 | train 前 24 对，offset 0 |
| 复验 | val 前 24 对，所有设置完全不变 |
| 最终留出 | test，不用于 U0 选择 |
| 区域 | 4×4 网格，每格至少 48×48 |
| 掩码 | 8 像素 cosine taper，避免硬边 |
| 强度 | ε = 0.10、0.20；同格正负绝对值相同 |
| 有效像素 | SAR 非零像素占比至少 90%；排除项记录 |
| 评估格式 | clamp -> uint8，与 C0 原评估一致 |
| 局部指标 | QAB/F↑、NAB/F↓、FMI↑、VIF↑、SSIM↑ |
| 方向阈值 | QAB/F 对比至少 0.0002；只作为描述性阈值 |
| 基本统计单位 | 图像，而不是同一图的 16 个 tile |

原始论文指标按全图定义；这里对同一 ROI 进行成对比较。局部 FMI/VIF/QAB/F 的绝对数值不能和全图 val200 指标直接比较。

## 4. 验证内容

1. C0 identity：缓存特征路径的零扰动结果必须与正式 C0 forward 的结果在浮点误差内一致。
2. 有效扰动：output_changed_fraction 报告 uint8 输出在 ROI 内真正改变的比例。
3. 跨区域影响：outside_change_fraction 报告输出差异溢出 ROI 的比例，避免把非局部 Decoder 效果当局部效用。
4. 方向存在性：区分 q_direction、q_beneficial_direction、pareto_direction。
5. 两种方向：每图是否同时存在对 C0 有益的正扰动格和负扰动格。
6. 稳定性：两种 ε 下的 q_beneficial_direction 是否非零且相同。
7. 多指标：QAB/F、FMI、VIF、NAB/F 是否出现同向的非退化效用，而非单项 trade-off。
8. 跨 split：发现阶段 train24；固定 val24 仅复验，不用 val 调参数。
9. 科学限制：已有 val200 被用于前序 R1 决策，不能把 val24 视为从未使用过的最终测试。

## 5. Windows CMD 运行

先在项目目录执行 git pull。

发现阶段：
    run_u0_local_utility_discovery.cmd

发现存在可重复收益信号时，按完全相同协议复验：
    run_u0_local_utility_validation.cmd

输出目录：
    test_result\reliability_v5\u0_local_utility_train24\
    test_result\reliability_v5\u0_local_utility_val24\

关键输出：

- tile_counterfactuals.csv：逐图、逐格、逐强度的 C0/正/负局部评价和方向判定。
- strength_summary.csv：方向对比、真增益、多指标保真统计。
- per_image_summary.csv：逐图的正/负有益区域与外溢率。
- tile_stability.csv、stability_summary.json：两种扰动强度的稳定性。
- protocol.json：checkpoint SHA256、manifest 摘要、全部参数和口径。
- skipped_tiles.csv：存在无效 SAR tile 时记录排除项。

## 6. 进入 U1 的条件

不能只因为 Q(+ε) ≠ Q(-ε) 就训练一个 Utility Router。

要同时看到：有真实像素变化；正负扰动相对 C0 都分别能在某些区域改善 QAB/F；这种方向分布跨 ε 稳定；保真指标不被普遍牺牲；在 val24 上可复验；代表性 ROI 的人工检查排除散斑、量化、配准误差造成的伪收益。

若这些不成立，停止通过空间门控估计 SAR utility 的路线。

若成立，U1 才以经过独立检验的局部对照为依据，研究通过 (D_o,D_s) 与可选 E_x/R_0 预测区域边际效用，并要求优于均值、随机、仅光学等对照。指标代理本身仍然不能称为真实 utility label。

备注：局部无参考分数不能证明正确边缘与 speckle 已被区分；4×4 网格和两种 ε 只是可行性阶段，不能替代尺度、偏移和样本外鲁棒性检验。
