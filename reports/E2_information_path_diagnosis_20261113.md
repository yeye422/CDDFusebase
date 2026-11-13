# E2 信息损失路径定位：证据审查与冻结前向诊断

日期：2026-11-13。对象：`yeye422/CDDFusebase` 的 `main@0fc5998`，主模型 `lwpd_e2_base_anchor_v1`，权重 `models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。**状态：完成代码路径和已有实验数据的诊断；冻结权重新增前向干预脚本已经编写并通过合成检查，但因当前环境缺少用户的 QXS 原始图像及模型权重，尚未运行真实四场景实验。脚本和合成测试已提交到该仓库，用户拉取后可直接运行。**

## 1. 问题与可检验分歧

研究问题不是“Base/Detail 哪个能量大”，而是光学与 SAR 的互补结构究竟在编码分解、融合、还是最终解码阶段未能进入输出。

实际源码路径：

```text
O / S ─→ 共享Encoder ─→ LWPD（两模态分别Haar LL/LH/HL/HH）
                         ├→ Bo, Bs ─→ Bf=Bo+CrossModalBaseFusion(Bo,Bs)
                         └→ Do, Ds ─→ Df=MultiScaleDetailFusion(Do,Ds,Bf)
                                          ↓
                Decoder(concat(Bf,Df)→1×1→4 Restormer→输出卷积→Sigmoid)
                                          ↓
                                         F
```

LWPD 有显式一级固定 Haar 与方向门控，`MultiScaleDetailFusion` 的“三尺度”只是膨胀率为 1/2/3 的同尺寸卷积；并没有第二次真实小波子带分解或明确的父子尺度系数关系。Decoder 还会对两个特征支路进行通道混合和非线性变换，因此不能将 LWPD 内部 Base/Detail 解释为最终融合图频带的严格分工。

## 2. 已核实的实测证据

### 2.1 输出模态相似性呈现强烈场景差异

已读取 `test_result/e2_scratch_rccr_l2_v1/val1000/seed42/e2/diagnostics.csv` 及同组 `metrics.csv`。此处 `e2/` 使用光学 Base E2 的既有权重 `epoch_016.pth`，评价版本为 `reference_v2_candidate_with_paper_fmi`。1000 张图的分组为：

| 条件 | N | SSIM(O,F)均值 | SSIM(S,F)均值 | QAB/F均值 | VIF均值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 全部 | 1000 | 0.30370 | 0.75227 | 0.81579 | 0.39330 |
| SSIM(S,F)>SSIM(O,F) | 668 | 0.09675 | 0.96918 | 0.86679 | 0.45548 |
| SSIM(O,F)>SSIM(S,F) | 332 | 0.72010 | 0.31584 | 0.71319 | 0.26819 |
| SSIM(S,F)>0.95 | 608 | 0.06773 | 0.99491 | 0.87743 | 0.46869 |

这里反映的是**模态相似性不对称**，而不是“668 张确定只使用 SAR”、“608 张复制错误”或“模态贡献率”。至少需要源图 O/S 相互相似性、局部独有目标、重建和受控干预才能评价互补信息损失。

### 2.2 单源重建阶段并非明显不收敛，但尚无最终权重下的逐图重建质量结论

已有训练报告记录：第 4 轮单模态验证重建损失，光学约 `0.001542`、SAR 约 `0.002615`；训练结束时光学 Base 模型融合输出 `val1000` 的 SSIM 约 `0.527984`。这说明重建目标能够被拟合，不等于已经证明最终权重可以无损恢复两模态的所有有效结构。

因此，**尚不能将信息丢失首因定位在 LWPD**，也不能彻底排除 Encoder/LWPD 对光学微细结构的削弱。新脚本的最终 checkpoint 单源重建质量检查正是为补这项证据。

### 2.3 Base 全局关系的远端耦合已经存在；不是已证明的主要结构瓶颈

固定 Scene4 的光学 Base E2 前向验证了，远端 SAR Q/K 幅值变化使全局交互目标区 Base RMS 改变约 `0.004002`，普通局部改为 `0`。但完整 C 相对原全局使该图 SCD 减少约 23.4%、VIF 减少约 13.8%，光学梯度一致性的变化很小。

这些事实证明了一个**可干预的因果路径**，并没有证明它对真实自然场景上的结构缺失有主导作用。C 不宜直接训练；普通局部仍可作为对照。

### 2.4 融合阶段和监督偏向具有更直接的风险证据

已有相同训练目标的另一组 `E2-B` 六图诊断中，原始强度+10倍梯度代价对 SAR 复制的均值约 `0.393006`，对应 E2-B 输出约 `0.527014`，而光学复制约 `9.289792`。**这是 E2-B 的六图诊断，不能将数值直接归给当前光学 Base checkpoint**；但它提示 max 强度与 max 梯度目标可以强烈偏向 SAR 外观。

此外，另一个从头训练的 `structure_joint` 候选模型在1000张验证图上的 `sar_weight_mean` 平均约 `0.99204`，`optical_weight_mean` 平均约 `0.00619`。该模型并非当前基线；它说明即使增添显式结构路由，在既有训练设置下仍可能学到极端 SAR 权重，而不是自动实现互补融合。

LL-context 的完整验证与单图开关实验也发现：新分支改善原训练目标，却伴随部分结构指标和 NAB/F 恶化，关闭新分支不能恢复原 E2。这支持“结构自由度 + 共同适应 + 原监督目标”存在相互影响，不能把所有退化归因于新增分支本身。

## 3. 当前瓶颈定位（区分证据强度）

| 位置/机制 | 证据支持度 | 当前判断 |
| --- | --- | --- |
| 输入/Encoder/LWPD | 较弱 | 已有重建损失较低，但未测最终checkpoint的逐图单源重建高频保留；暂不能指认为首因 |
| Base 全局通道交互 | 中等（有干预） | 证实远端统计耦合，未证实它是结构损失的主要来源 |
| 跨源 Base/Detail 融合及模态选择 | 中等（代码、相似性和替代模型结果） | **首选排查位置**；两路信息已进入融合器，但最终表现出强烈场景级单源相似性 |
| Decoder 对 Base/Detail 的重新组织 | 代码风险明确，实测归因较弱 | 存在通过 Base 生成输出高频、通过 Detail 改写大尺度对比的可能；尚需冻结输入干预 |
| 训练目标 / 共同适应 | 中等（六图诊断与失败对照） | 是最重要的混杂因素之一；即便新架构提高信息容量，也未必使监督奖励互补保留 |

**最有依据的暂定结论**：优先排查跨模态融合路径与原监督目标造成的模态偏向；**尚未完成融合器与 Decoder 二者之间的定量归因**。目前没有证据支持直接宣布“LWPD 太弱”或“Decoder 是主因”。

## 4. 已实现的补充诊断入口与解释原则

诊断脚本已放入仓库根目录 `diagnose_e2_information_path.py`，Windows CMD：

```cmd
cd /d "D:\#imageInfusion\CDDFuse"
.venv\Scripts\python.exe diagnose_e2_information_path.py --scene-ids 4 44 1267 13273 --output-dir test_result\e2_information_path_v1\fixed4_20261113
```

脚本执行：

1. 严格核验场景属于原 val 清单、checkpoint 版本、全模型前向与手动拆解路径一致；保留 SHA256 与源图哈希，不覆盖旧实验目录。
2. 使用**训练结束时的原权重**，分别解码 `Bo+Do`、`Bs+Ds`，测量源图重建 MAE、SSIM、高频相关系数与高频能量比，判断进入融合器之前是否已有明显结构损失。
3. 对 Bo/Do/Bs/Ds 分别做 `±5%` 特征幅值双向干预，每次重新计算 `Bf、Df` 与输出，报告单位幅值的输出局部敏感性。**不将响应比值称为模态贡献率。**
4. 固定 Df 测 Decoder 的 Base 输入敏感性；固定 Bf 测 Detail 敏感性，报告输出变化中的低/高频能量。
5. 对 Df 建立 fine/middle/coarse **代数可重建的诊断金字塔**，在 Decoder 入口对不同尺度分别做小幅双向扰动，判断不同尺度的输出作用。这个诊断金字塔不是已实现的新网络，也不是物理频带真实性证明。
6. 单独记录去掉 Base correction 的较大干预，但明确标记为分布外对照，不用于直接归因。

输出 `diagnosis.json`（逐图与汇总）和 `summary.csv`（关键路径对照）。脚本已通过 `py_compile` 和合成模型张量检查；**未在真实模型 checkpoint+QXS 场景执行**，需要在用户本地环境运行后才能完成相应量化结论。

### 判定条件

- **单源重建本身已丢失光学有效细节**：先研究 Encoder/LWPD，而不是继续更改 Detail 融合；对强SAR散斑背景不能把高频能量比高视为重建良好。
- **单源重建好、Optical Detail 路径敏感性低、SAR Detail 路径敏感性高**：优先排查 Detail 融合或训练目标；需控制两路输入特征的幅值差和所在尺度，避免“输入比较小→输出比较小”的伪结论。
- **两源 Detail 都能影响 Df，但 Decoder 对 fine Detail 响应很弱**：优先检查 Decoder 与 Base/Detail 混合，而不是盲目增大 Detail 频段。
- **各条路径都有明显输出响应，却仍高度贴近单源**：架构容量不足的解释变弱，应优先检查 max 强度/max 梯度目标的偏好与局部有效信息的取舍。

这些是“下一步应检查什么”的决策规则，而非未跑数据就能得出的事实。

## 5. 下一步研究决策

本轮不建议直接训练“跨尺度关系保持”和“保守型跨尺度重建”。只有在新增真实场景响应诊断之后，才能有根据地确定二者优先级。

- 若瓶颈位于 **Detail 融合器**：验证模态内跨尺度关系作为融合条件是否优于同容量卷积/普通金字塔；先不加入额外自由高频恢复。
- 若瓶颈位于 **Decoder 重建**：验证基于已有融合特征的受约束跨尺度合成；保留 SAR 独有信息通路，不让“更像光学”成为唯一目标。
- 若主要瓶颈位于 **监督目标**：继续增加跨尺度模块很可能重演 LL-context/PAC 的目标拟合与指标背离，应先用现有损失实验确定可行目标或在架构设计中加入可审查的行为约束。

研发选择只依据固定验证集和预先限定的开发场景；不使用已看过的 test1000 做架构调参。新参考评价器有版本差异，不能混用旧 `paper` 的 NAB/F、QAB/F 与 MS-SSIM 数值。

## 6. 仓库依据

- `direct_fusion.py`, `lwpd.py`, `e2_base_anchor.py`, `net.py`
- `test_result/e2_scratch_rccr_l2_v1/val1000/seed42/e2/diagnostics.csv` 与同组指标 CSV
- `test_result/e2_scratch_struct_joint_v1/val1000/seed42/gate_stats.csv`
- `reports/lwpd_pac/E2_固定六图诊断结果分析_20261104.md`
- `reports/lwpd_pac/LWPD_LL_CONTEXT_V1失败原因与单样本干预_20261104.md`
- `reports/C区域均衡通道关系_单场景机制诊断_20261113.md`
- `reports/lwpd_pac/LWPD_E2_BASE_ANCHOR_TECHNICAL_REPORT.md`

## 7. 仓库执行入口（2026-11-13 更新）

先运行合成 smoke test：

```cmd
cd /d "D:\#imageInfusion\CDDFuse"
.venv\Scripts\python.exe tests\test_e2_information_path_smoke.py
```

真实权重诊断：

```cmd
.venv\Scripts\python.exe diagnose_e2_information_path.py --scene-ids 4 44 1267 13273 --output-dir test_result\e2_information_path_v1\fixed4_20261113
```

脚本只进行冻结前向，不修改checkpoint，不访问test数据，不自动登记实验注册表。真实数据结果尚未产生，不能当作已完成的模块因果归因。
