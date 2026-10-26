# 可靠性驱动 Optical–SAR 融合研究实验计划

版本：v5.0（基于阶段 0–1 实验结果修订）  
日期：2026-09-30  
状态：主线已由“SAR Detail 残差注入”调整为“频率职责解耦与可靠性证据折扣”

## 1. 修订结论

本计划替代 v3.0 的“Reliability × Fusion Demand 控制 SAR Detail residual”主线。
已有完整训练、正式测试和 70 组 Pareto 扫描表明：

1. B0 能传递大量 SAR 高频，但同时扩散 speckle 和伪边缘；
2. B1-v5 能显著降低伪边缘，但主要依靠整体抑制 SAR 高频，QAB/F、VIF 和 FMI 大幅下降；
3. learned DemandGate 不优于固定门；
4. A1′ 中增大 SAR residual scale 会持续降低 QAB/F 和 VIF；
5. epoch 1、scale=0 的信息保持指标最好，说明当前有效路径主要来自 SAR Base、Optical Detail 与早期 BaseFuse 调整，而非 SAR Detail residual。

因此，后续主方法冻结为：

```text
Optical：负责最高频自然纹理与颜色主体
SAR：负责 Base 和经过结构验证的中尺度信息
Reliability：折扣不可信 SAR 证据，不直接生成或注入 SAR Detail
```

旧 A1/A1′/双门控方案保留为失败消融，不继续扩展。

## 2. 已完成工作与实验证据

### 2.1 数据与评测协议

- 数据：QXS-SAROPT；
- 固定划分：Train 8,000 / Validation 1,000 / Test 1,000；
- split 内容 SHA-256：`5199869db0d72c04edffe037dee80ecf55dcb49f6c14e429fd8b543ec687d2a7`；
- 初始化：`CDDFuse/models/qxslab_saropt_from_ivf/CDDFuse_OS_20260924_012025/best.pth`；
- 输出范围：正式实验统一使用 `clamp`；
- 评价：论文口径 14 项指标，主指标为 NAB/F、QAB/F、VIF、FMI、SSIM、MS-SSIM；
- test 集不用于 checkpoint 或超参数选择。

### 2.2 B0 与 B1-v5 正式 1000 对测试

| 模型 | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---:|---:|---:|---:|---:|---:|
| B0 原 CDDFuse | 0.6946 | 0.7138 | 0.1145 | 0.5312 | 0.5737 | 0.3948 |
| B1-v5 | 0.2976 | 0.3825 | **0.0522** | 0.5012 | 0.5811 | 0.2359 |

B1-v5 相对 B0：NAB/F 下降约 54%，但 QAB/F、VIF、FMI 分别下降约 46%、40%、57%。其视觉结果更自然、speckle 更少，但建筑强散射、屋顶轮廓、港口目标等 SAR 信息明显减弱。B1-v5 证明共同损失能够抑制噪声扩散，但没有实现选择性结构保留。

### 2.3 A1、固定门与 A1′验证

固定前 200 对 validation：

| 模型 | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---:|---:|---:|---:|---:|
| B1-v5 pilot | 0.2861 | **0.0317** | 0.4804 | **0.6134** | 0.1620 |
| A1 learned gate | 0.2783 | 0.0357 | 0.4908 | 0.6034 | 0.1652 |
| A1 fixed gate=0.5 | **0.2885** | 0.0353 | **0.4917** | 0.6045 | **0.1666** |
| A1′ residual-only | 0.2789 | 0.0357 | 0.4902 | 0.6039 | 0.1653 |

DemandGate 虽未数值塌缩，但没有优于固定门，不能作为必要组件。

### 2.4 A1′ 70 组 Pareto 扫描

扫描 10 个 epoch × 7 个 residual scale，每组固定 200 对 validation。34 组通过宽松闸门，但最佳结果集中在 epoch 1 和较小 scale。

若要求非零注入，当前最好候选为 epoch 1、scale=0.10：

| FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---:|---:|---:|---:|---:|---:|
| 0.3833 | 0.5322 | 0.0030 | 0.5114 | 0.5890 | 0.2607 |

epoch 1、scale=0 的结果更高：FMI 0.3915、QAB/F 0.5420、VIF 0.2611，同时 NAB/F 仅 0.0033。随着 scale 增大，NAB/F 略降，但 QAB/F 与 VIF 单调下降。当前 residual 更像高频抵消项，而非可信 SAR Detail 注入项。

该扫描只用于方向诊断。由于 70 个配置共享同一 200 对 validation，不能将其结果直接作为最终模型结论。

## 3. 修订后的研究定位

### 3.1 中心问题

> 在无融合真值条件下，能否将 Optical 的自然高频纹理与 SAR 的稳定中尺度/Base 结构进行职责解耦，并通过可验证的局部可靠性折扣 SAR 中的不稳定证据，从而在降低 speckle/伪边缘的同时保留建筑、道路、岸线和强散射目标？

### 3.2 三个可检验假设

- **H1：频率职责解耦有效。** Optical 提供最高频、SAR 只进入 Base/中尺度，能够比 B1-v5 保留更多 QAB/F、VIF 和 FMI，同时比 B0 降低 NAB/F。
- **H2：结构选择优于全局抑制。** 固定结构可靠性图能使稳定结构区域的 SAR 保留率高于平坦区噪声传递率。
- **H3：学习型 Reliability 具有额外价值。** Learned Reliability 在定位、融合质量和未见退化鲁棒性上显著优于固定结构规则，而不是仅将所有 SAR 权重压低。

### 3.3 创新边界

本文不再宣称：

- SAR Detail residual 是有益信息载体；
- Reliability 与 Demand 双门控必然互补；
- 最大梯度或高响应等同于有效结构；
- 8-bit PNG 上的 Gamma 扰动是严格物理 speckle。

拟保留的创新主张为：

1. 面向 Optical–SAR 异质频率职责的显式解耦；
2. 结合 SAR 专用局部统计、方向连续性和跨尺度持续性的结构证据；
3. 由局部相对退化监督学习 Reliability，并将其作为 SAR 中尺度/Base 证据的折扣；
4. 同时验证可靠性定位、结构选择性、融合质量和未见退化鲁棒性。

## 4. 冻结方法架构

### 4.1 共同编码

```text
(B_o, D_o) = Encoder(O)
(B_s, D_s) = Encoder(S)
```

颜色处理维持 Optical RGB → YCbCr，只融合 Y 通道，最后复用 Optical Cb/Cr。论文准确称为亮度—结构融合。

### 4.2 C1：最小职责解耦基线

```text
B_f = BaseFuse(B_o + B_s)
D_f = D_o
F   = Decoder(O, B_f, D_f)
```

`D_s` 不进入融合。C1 回答 SAR Base 是否已经足以提供互补信息。

### 4.3 C2：Optical Detail refine

```text
D_o_bar = D_o + T_o(D_o)
B_f     = BaseFuse(B_o + B_s)
D_f     = D_o_bar
F       = Decoder(O, B_f, D_f)
```

`T_o` 使用卷积、归一化和零初始化末层。删除 `T_s`、ResidualCandidate、DemandGate 和所有 SAR Detail 注入参数。C2 是后续 Reliability 的直接母基线。

### 4.4 C3：固定结构证据

固定可靠性图由四类证据构成：

1. 多方向、多尺度 Ratio-of-Averages SAR 边缘；
2. structure-tensor coherence；
3. 3/5/9 像素尺度上的响应持续性；
4. 跨尺度稳定的强散射点分支。

```text
R_line  = E_RoA * C_tensor * P_scale
R_point = E_CFAR * P_point
R_fixed = max(R_line, R_point) * M_valid
```

Optical/SAR 边缘一致性只能作为辅助正证据，不能把 Optical 中不存在的 SAR 目标自动判为噪声。

### 4.5 C4：Learned Reliability

第一版 ReliabilityNet 输入：

```text
D_s
多尺度 RoA
structure-tensor coherence
跨尺度持续性
相对高频能量/孤立度
局部低秩或非局部自相似残差（可选教师 cue）
```

网络保持轻量：

```text
3×3 Conv → Depthwise Residual Block ×2 → 1×1 Conv → Sigmoid
```

输出单通道 `R_s∈[0,1]`。第一阶段 `R_s` 只作用于损失目标，融合损失对其使用 stop-gradient。

### 4.6 C5：Reliability-gated Base

只有 C4 通过后，才允许：

```text
R_b     = AvgPool(R_s)
R_floor = 0.25 + 0.75 * R_b
B_f     = BaseFuse(B_o + stopgrad(R_floor) * B_s)
```

0.25 的下限用于避免网络重新退化成 Optical-only。C5 用于判断特征级折扣是否比 loss-only 多出实际收益。

## 5. 频率职责与融合损失

### 5.1 Optical 低频外观

\[
L_{opt-low}=L_1(G_{\sigma=2}(F),G_{\sigma=2}(O))+DSSIM(G_{\sigma=2}(F),G_{\sigma=2}(O)).
\]

### 5.2 Optical 高频保持

最高频只向 Optical 对齐：

\[
L_{opt-high}=Charbonnier(H(F)-H(O)).
\]

`H` 首选一层 Haar DWT 的 LH/HL/HH，也可用固定 Laplacian high-pass。两种实现只能选一种进入冻结协议。

### 5.3 SAR 中尺度结构

先计算有符号的 Sobel-x/y 分量，并以分量模长决定来源：

\[
\mathbf g_o=\nabla G_{\sigma=2}(O),
\qquad
\mathbf g_s=clip_{p99}(\nabla G_{\sigma=2}(S)).
\]

对 C1/C2 令 `R=1`；C3 使用 `R_fixed`；C4/C5 使用 `stopgrad(R_s)`：

当 `R|g_s|>|g_o|` 时选择 `R g_s`，否则选择 `g_o`，得到有符号目标
`g_tar`。模长只用于来源选择，损失监督保留 x/y 方向和符号，避免零梯度及错误方向：

\[
L_{sar-mid}=Charbonnier(\nabla G_{\sigma=2}(F)-\mathbf g_{tar}).
\]

总损失：

\[
L=L_{opt-low}+\lambda_hL_{opt-high}+\lambda_mL_{sar-mid}+\lambda_rL_{rel}.
\]

C1/C2/C3 中 `λ_r=0`。禁止重新加入未经结构筛选的最大 SAR 高频梯度目标。

### 5.4 固定终点参考归一化 minimax

首次 C0/C1/C2 pilot 表明 normalized PCGrad 在五轮内持续向 Optical 漂移；中间
epoch 的短暂优势不得用于 checkpoint 选择。因此下一轮仅保留 C0 架构，并在训练前
用无增强的 training split 中心 crop 计算冻结参考：

\[
r_{opt}=L_{opt}/(L_{opt}^{0}+\epsilon),
\qquad
r_{sar}=L_{sar-mid}/(L_{sar-mid}^{0}+\epsilon).
\]

训练目标改为：

\[
L_{balanced}=\tau\log
\left(\exp(r_{opt}/\tau)+\exp(r_{sar}/\tau)\right),
\qquad \tau=0.1.
\]

参考值不从 validation 读取、不随训练更新，并写入 config/checkpoint。该实验固定训练
5 epochs，只评价最终 epoch 5；中间 checkpoint 仅用于故障恢复。本轮不同时使用
PCGrad，以确保能单独归因 minimax 的作用。

## 6. Reliability 监督

### 6.1 局部相对退化

在 SAR 有效区内生成 10%–40% 的软边界局部掩膜：

\[
\tilde S=(1-M_d)S+M_d(S\odot N).
\]

该过程称为 controlled multiplicative corruption，不称为严格物理 speckle。

### 6.2 核心损失

\[
L_{rank}=\frac{\|M_d\max(0,m+\tilde R-R)\|_1}{\sum M_d+\epsilon},
\]

\[
L_{unchanged}=\frac{\|(1-M_d)(R-\tilde R)\|_1}{\sum(1-M_d)+\epsilon}.
\]

固定结构教师只在前期使用并线性衰减：

\[
L_{teacher}=BCE(R,R_{fixed}).
\]

\[
L_{rel}=L_{rank}+0.5L_{unchanged}+\lambda_t(t)L_{teacher}.
\]

第一版不加入 `L_smooth`、`L_cons`、语义分支或通道级 Reliability。只有明确失败模式出现后，每次增加一项。

## 7. 实验阶段与闸门

### 阶段 0：结果与协议冻结（已完成）

- B0 clamp 正式重评；
- B1-v5 单 seed 完整训练和 1000 对测试；
- A1/A1′ pilot 与固定门对照；
- A1′ 70 组 Pareto 扫描；
- 旧 Detail residual 主线判定为不支持。

零样本历史结果缺少元数据且可能使用 `legacy_minmax`，必须重新用 clamp 评估后才能进入正式表格。

### 阶段 1：干净无 residual 基线

执行更新（2026-10-01）：固定 5 轮终点下 C0/C1/C2 均未通过；C0 明显优于 C1/C2，
因此停止删除原始 DetailFuse 和 Optical refiner 路线。下一次只运行 C0 的训练期参考
归一化 minimax 版本，仍固定 5 轮并只评价最终 checkpoint。

决策更新（2026-10-01）：C0-minimax 固定终点相对 B0 将 NAB/F 降低 60.95%，
QAB/F 保留 93.97%，但 VIF/FMI 未达到原 95% 硬门槛。该结果按 v5.0 原标准记录为
未通过；经人工审批，工程阶段暂定通过并允许启动完整 seed 42 训练。该决定不追溯
修改原闸门结论，最终论文结论仍需完整 validation/test、区域选择性指标和定性 ROI
共同确认。

实现 C1、C2，并从原 B0 checkpoint 重新初始化：

- 冻结 Encoder/Decoder；
- 训练 BaseFuse；
- C2 额外训练 `T_o`；
- 800 train / 200 validation；
- 5 epochs，每轮保存；
- 不加载 A1′ checkpoint；
- 不查看 test。

进入阶段 2 的条件：C2 相比 B1-v5 pilot 同时满足：

1. NAB/F 不高于 B1-v5 pilot 的 1.10 倍；
2. QAB/F、VIF、FMI 均不低于 B1-v5 pilot；
3. MS-SSIM 不低于 B1-v5 pilot 的 95%；
4. Output–Optical MAE ≥ 0.01；
5. 典型场景未出现明显亮度、颜色或纹理崩坏。

若 C2 不优于 C1，删除 `T_o`，后续使用 C1。

### 阶段 2：固定结构证据 C3

执行更新（2026-10-01）：C3a RoA-only pilot 相对 B0 保持约 95.4% FMI、99.1%
QAB/F，并将 NAB/F 降低约 24.1%；相对 C0-minimax 的 NAB/F 则明显反弹。
在固定结构图因果控制尚未补齐的情况下，经人工审批允许 C3a 作为工程主候选进入
完整 seed 42 训练。C3b 局部 FA 桥接只保留为 pilot 消融，不同步扩展。

先可视化 `R_fixed`，再训练 C3。必须检查：

- 建筑、道路和岸线是否形成连续响应；
- 船舶/角点是否被 point 分支保留；
- 水面、农田等平坦区响应是否较低；
- Optical 中无对应边缘的 SAR-only 目标是否被保留。

若 C3 不优于 C2，不允许训练 Learned Reliability；先修正结构证据或终止该路线。

### 阶段 3：Learned Reliability C4

先单独验证定位，再接入融合。最低定位条件：

- AUROC 建议 ≥0.75，最低不得低于 0.65；
- AUPRC 明显高于退化区域先验比例；
- 随退化强度单调下降；
- 未退化区漂移较小；
- SAR-only ROI 未被系统判低。

C4 必须优于 C2 与 C3，否则 Learned Reliability 没有必要性。

### 阶段 4：Base gating C5

仅当 C4 通过后实施。若 C5 与 C4 无显著差异，删除 Base gating，最终模型使用 loss-only Reliability。

### 阶段 5：完整训练与外部验证

1. seed 42、8,000 train / 1,000 validation；
2. 模型与超参数冻结后进行一次 1,000 test；
3. 通过后补 seed 3407、2026；
4. Dongying 或区域互斥外部测试；
5. 未见 Gaussian noise、blur、2 px shift；
6. 统计检验和效率评价。

## 8. 选模规则

不再使用仅基于训练损失的 minimax score，也不逐 epoch 计算融合指标后选择最优
checkpoint。训练轮数或优化步数必须在实验启动前冻结，正式比较只评价固定的最终
checkpoint；中间 checkpoint 仅用于崩溃恢复和训练动力学诊断。

固定终点模型仍需满足以下 validation 闸门：

1. QAB/F、VIF、FMI 均达到直接母基线的 95%；
2. MS-SSIM 达到母基线的 95%；
3. Output–Optical MAE ≥0.01；
4. 不允许通过扫描 epoch、事后缩短训练或选择中间 checkpoint 使模型通过。

validation 可用于选模，test 只做最终一次评价。探索阶段使用过的前 200 对 validation 不再作为独立确认集；冻结设计后使用剩余 validation 样本做确认。

## 9. 评价体系

### 9.1 全图融合指标

主指标：NAB/F、QAB/F、VIF、FMI、SSIM、MS-SSIM。其余 8 项完整报告但不用于“第一数量”叙事。AG、SF、EN 升高可能来自噪声，不能单独证明改进。

### 9.2 选择性结构指标

在预注册稳定结构区域和低可靠性区域分别计算：

\[
Retention_{stable}=Corr(\nabla F,\nabla S\mid R_{fixed}>0.7),
\]

\[
Transfer_{flat}=Corr(HP(F),HP(S)\mid R_{fixed}<0.2),
\]

\[
Selectivity=Retention_{stable}-Transfer_{flat}.
\]

目标是稳定结构保留率高、平坦区噪声传递率低，而不是统一降低 SAR 相关性。

### 9.3 Reliability 定位

- `1-R` 检测局部退化的 AUROC/AUPRC；
- 退化区平均可靠性下降；
- 未退化区平均漂移；
- 三档退化强度的单调性；
- Reliability 均值、标准差、低/高值占比。

### 9.4 定性与 ROI

预注册 24 对场景：建筑 6、港口/船舶 6、道路/岸线 4、农田/植被 4、水体/平坦区 4。展示 Optical、SAR、B0、B1-v5、C2、C3/C4、Reliability、局部放大图。

### 9.5 统计与效率

- 核心模型 3 seeds；
- paired bootstrap 95% CI；
- 六项主指标做配对 Wilcoxon 与 Holm 校正；
- 报告效应量、参数、FLOPs、GPU/CPU 推理时间和峰值显存。

## 10. 最低成功标准

最终模型相对正式 B0（clamp）必须满足：

1. NAB/F 至少下降 15%；
2. QAB/F、VIF、FMI 均保持 B0 的 95%以上；
3. SSIM 或 MS-SSIM 至少一项显著改善，另一项下降不超过 1%；
4. Selectivity 显著优于 B0 与 B1-v5；
5. 未见退化 robustness AUC 至少提高 10%；
6. 若保留 Reliability 主张，定位 AUROC ≥0.75。

前五条至少满足四条，其中第 1、2 条必须满足。第 6 条决定是否可以使用“Reliability Map”作为核心论文表述。

## 11. 判停与降级规则

- C2≈C1：删除 Optical aligner；
- C3≤C2：固定结构选择无效，不训练 Learned Reliability；
- C4≈C3：删除 ReliabilityNet，保留固定结构筛选；
- C4≈C2：删除全部 Reliability，研究收缩为频率职责解耦；
- C5≈C4：删除 Base gating；
- NAB/F 改善但 QAB/F、VIF、FMI任意两项下降超过 10%：判定为全局抑制，不通过；
- AUROC<0.65：只能称 degradation-aware gate；
- SAR-only ROI 被系统抑制：降低跨模态 cue 权重，不通过统一抬高 Reliability 修补。

## 12. 实施顺序与目标接口

根据项目约束，下列训练和大批量测试只提供 CMD 模板，不在代码实现阶段自动运行。

### 12.1 C1/C2 pilot（接口实现后）

```bat
cd /d D:\#imageInfusion\CDDFuse

.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant base_optical_detail_refine ^
  --training-policy fusion_only ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --init-checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
  --output-dir models\reliability_v5 ^
  --run-name C2_pilot ^
  --train-limit 800 ^
  --val-limit 200 ^
  --epochs 5 ^
  --lr 1e-5 ^
  --seed 42
```

### 12.2 C3 固定结构引导（接口实现后）

```bat
.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant fixed_structure_guided ^
  --structure-evidence roa,coherence,persistence,point ^
  --selection-reference-config models\reliability_v5\C2_pilot\config.json ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --init-checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
  --output-dir models\reliability_v5 ^
  --run-name C3_pilot ^
  --train-limit 800 ^
  --val-limit 200 ^
  --epochs 5 ^
  --lr 1e-5 ^
  --seed 42
```

以上命令定义目标接口；当前代码尚未实现这些 variant，不得直接运行。

## 13. 近期交付计划

| 顺序 | 工作 | 交付物 |
|---:|---|---|
| 1 | 实现 C1/C2 和频率职责损失 | 单元测试、dry-run、模型参数统计 |
| 2 | 运行 C1/C2 800/200 pilot | 每轮 checkpoint、指标与典型图 |
| 3 | 用剩余 validation 确认候选 | 独立确认表、Output–Optical MAE |
| 4 | 实现并可视化 `R_fixed` | line/point/evidence maps、ROI审查 |
| 5 | 运行 C3 pilot | 固定选择是否有效的因果结论 |
| 6 | 仅在 C3 通过后实现 C4 | Reliability定位与融合结果 |
| 7 | 仅在 C4 通过后实现 C5 | loss-only 与 Base-gating 对照 |
| 8 | 冻结模型后进行完整训练和 test | 主结果、多种子、泛化、统计 |

## 14. 论文叙事建议

方法章节按以下因果链组织：

```text
B0：无条件高频注入导致噪声扩散
  ↓
B1-v5：全局抑制降低噪声但删除有效结构
  ↓
C1/C2：按模态重新划分频率职责
  ↓
C3：固定结构证据证明选择性保留可行
  ↓
C4：学习型Reliability是否优于人工规则
  ↓
C5：Reliability是否还应控制Base特征
```

核心贡献不再是“双门控Detail注入”，而是：

> 以模态频率职责解耦为基础，通过可验证的局部可靠性对 SAR 中尺度/Base 证据进行折扣，在不直接注入 SAR Detail 的条件下实现稳定结构保留与 speckle 抑制。

## 15. 参考方向

- CDDFuse：Base/Detail 表征母框架，CVPR 2023；
- [SOSTF](https://ieeexplore.ieee.org/document/9936631/)：SAR结构与Optical纹理分工；
- [WEMFusion](https://www.mdpi.com/2072-4292/18/4/612)：Optical高频、低频跨模态协同；
- [SAMDFuse](https://www.sciencedirect.com/science/article/pii/S0924271626002303)：多尺度退化鲁棒与语义评价；
- [Joint medical fusion, denoising and enhancement](https://www.sciencedirect.com/science/article/pii/S0031320318300529)：联合去噪与融合；
- [Noise-robust fusion with low-rank sparse decomposition](https://www.sciencedirect.com/science/article/pii/S0020025520301833)：低秩、patch先验与结构/噪声分解；
- [Deep evidential fusion](https://www.sciencedirect.com/science/article/pii/S1566253524004263)：可靠性折扣与不确定性融合；
- [SAR ratio-based edge detection](https://dihana.cps.unizar.es/proceedings/ICASSP/2006/pdfs/0200837.pdf)：SAR专用局部比例边缘。
