# CDDFuse 面向 Optical–SAR 融合的可靠性改进：审批与分步实施计划

> **历史版本提示（2026-09-30）：** 本文档保留为 v4 决策与失败消融记录。
> B1-v5、A1/A1′及70组残差扫描已经否定“Reliability控制SAR Detail residual”
> 作为主线的前提。当前执行计划以
> `研究实验计划_可靠性驱动光学SAR融合.md` v5.0 为准。

版本：v4.0（核心贡献收敛版）  
日期：2026-09-25

## 1. 审批结论

**结论：有条件通过，但不批准把现有候选模块和损失一次性全部加入。**

方案抓住了当前结果中的真实矛盾：训练后的 CDDFuse 提升了 FMI、QAB/F、AG、SF、VIF 等信息与高频指标，同时 NAB/F、MS-SSIM、CC 等指标退化，视觉上又出现 SAR 散斑和强散射响应过量进入融合图的问题。把研究对象限定在 Detail 分支的“选择性注入”是正确方向，也比重做整个编码器更容易形成清晰贡献。

但是，现有方案的第一版同时加入四类 Reliability cue、双门控、残差注入、局部排序、未退化一致性、平滑、Fusion Consistency、Optical loss、SAR structure loss 和 Base correlation 对照。这样会带来三个问题：

1. 提升来源无法归因，论文会像多个技巧的叠加；
2. `R_s`、`A_s` 和损失权重之间可能互相代偿，难以证明 Reliability 是必要机制；
3. 当前代码的 Stage II 本身使用无条件 Detail 相加和最大梯度目标，若不增加同损失基线，改进可能只是来自损失替换，而非可靠性机制。

因此，本计划将方案收敛为一个中心命题，并按“共同损失控制 → 自适应注入 → 可靠性上界 → 必要时单项增强”的顺序实施。

## 2. 唯一中心贡献

论文主线建议统一为：

> **局部反事实可靠性约束的 SAR Detail 残差注入。** 网络从同一 SAR 局部区域的原始—退化对中学习相对稳定性分数，并将该分数作为内容驱动注入需求的上界，从而避免不可靠高频响应无条件进入融合结果。

英文可暂写为：

> **Locally Counterfactual Reliability-Bounded Detail Injection**

这是一项中心贡献，包含两个相互配合的技术环节，而不是两个互不相关的“创新点”：

1. **局部反事实可靠性学习**：通过同位置、不同受控退化强度的样本对学习 SAR Detail 的相对稳定性；
2. **可靠性有界注入**：Reliability 只规定 SAR 最多允许进入多少，内容门负责决定当前区域实际需要多少。

论文不应把“多尺度”“结构张量”“跨模态边缘”“平滑损失”分别写成创新。这些只能是服务于中心机制的候选实现或消融项。

## 3. 与原方法和近期工作的边界

CDDFuse 的原始贡献是相关性驱动的 Base/Detail 分解：Base 强调跨模态共享的低频全局信息，Detail 强调模态特有的高频局部信息。当前仓库在 Stage II 中仍采用：

\[
D_f^{base}=DetailFuse(D_o+D_s),
\]

并使用最大强度、最大梯度作为融合目标。对于 Optical–SAR，这两处都会促使 SAR 高频响应进入输出。因此，本研究不是重新证明或重新设计 Base/Detail 分解，而是补上 CDDFuse 未回答的问题：

> 模态特有的 SAR Detail 并不等于应该无条件保留的有效 Detail。

近期工作已经覆盖“抑制 speckle”“退化增强”“多尺度跨模态融合”“质量感知融合”等宽泛主题。因此不能把“考虑 SAR 噪声”或“加入可靠性门控”单独作为新颖性表述。差异化必须落在：

- Reliability 明确作用于 CDDFuse 的 Detail 注入，而非一般编码器去噪；
- Reliability 从局部反事实退化对中学习，不依赖融合真值或语义标签；
- Reliability 不是最终融合权重，而是对内容需求门的可验证上界；
- 同时验证 Reliability 定位、融合质量和未见退化鲁棒性。

## 4. 对原方案的逐项审批

| 原设计 | 审批意见 | 处理方式 |
|---|---|---|
| 保留 CDDFuse Base/Detail 表征 | 通过 | 作为母框架，不包装成新贡献 |
| 区分 `R_s` 与 `A_s` | 通过 | 保留，但必须 stop-gradient 并做 A-only/R-only 对照 |
| `W_s=R_s\odot A_s` | 有条件通过 | 解释为上界，因为 `W_s\le R_s`；必须验证两门不可互相替代 |
| `D_f=D_o+W_sD_s` | 修订后通过 | 不直接注入原始 `D_s`，改为注入联合预测的残差候选 `\Delta D_s` |
| 四类 cue 一次全部输入 | 不通过 | 核心版只使用 `D_s` 与 SAR 内部结构 cue；多尺度、跨模态 cue 后置 |
| `L_rank+L_unchanged` | 通过 | 作为核心 Reliability 监督 |
| `L_smooth` | 暂缓 | 只有 Reliability Map 出现椒盐跳变时再加 |
| `L_cons` | 暂缓 | 只有 Reliability 定位有效但输出鲁棒性不足时再加 |
| Base correlation 三套并行训练 | 暂缓 | 核心结果稳定后只做 `L_decomp=0` 对照 |
| 多类噪声同时训练 | 不通过 | 训练只使用一种受控乘性退化；其他退化只做测试 |
| 通道级 Reliability、语义分支、扩散模型 | 不通过 | 超出中心问题，首轮不进入方案 |

## 5. 核心方法的冻结版本

### 5.1 共同表征与 Base 分支

第一轮全部实验复用同一个已训练 QXS-SAROPT CDDFuse 检查点。Encoder、Decoder 和 BaseFuse 的初始化完全一致：

```text
CDDFuse/models/qxslab_saropt_from_ivf/
CDDFuse_OS_20260924_012025/best.pth
```

核心阶段保持：

\[
B_f=BaseFuse(B_o+B_s).
\]

不要在验证 Detail 机制前改 Base 分支。

### 5.2 稳定的残差候选，而非直接注入原始 SAR Detail

先对两模态 Detail 做轻量对齐：

\[
\bar D_o=T_o(D_o),\qquad \bar D_s=T_s(D_s).
\]

SAR 的可注入残差由两模态共同预测：

\[
\Delta D_s=\tanh\bigl(T_\Delta[\bar D_o,\bar D_s]\bigr).
\]

这样比直接使用 `T_s(D_s)` 更稳妥，因为 `D_s` 是为原始 INN 融合路径学习的特征，并不天然具有“可直接加到 Optical Detail 上”的尺度和分布。`T_o`、`T_s` 使用卷积加 GroupNorm/LayerNorm，`T_\Delta` 的末层零初始化。

### 5.3 内容需求门

\[
A_s=\sigma\bigl(H_A[\bar D_o,\bar D_s]\bigr).
\]

`A_s` 表示在当前内容条件下是否需要 SAR 补充，不承担噪声识别的论文叙事。

### 5.4 最小 ReliabilityNet

核心版只使用：

- 编码器输出的 `D_s`；
- 固定算子得到的 SAR 内部结构响应 `C_{str}`，包含高斯平滑后的梯度幅值和 structure-tensor coherence。

\[
R_s=H_R[D_s,C_{str}],\qquad R_s\in[0,1]^{H\times W\times1}.
\]

网络使用 `3×3 Conv → Depthwise Residual Block ×2 → 1×1 Conv → Sigmoid`。第一版不加入 Optical cue、多尺度 cue 或通道级输出。

### 5.5 可靠性有界残差注入

\[
W_s=\operatorname{sg}(R_s)\odot A_s,
\]

\[
D_f=\bar D_o+\gamma W_s\odot\Delta D_s,
\qquad \gamma=0.5.
\]

其中 `sg` 表示 stop-gradient。核心版将 `\gamma` 固定为 0.5，不再增加第三个可学习缩放因子；否则 `R_s`、`A_s` 与 `\gamma` 会产生额外的尺度代偿，使门控难以解释。只有在核心机制已经通过、且残差幅值被证明确实受限时，才允许把 `\gamma` 改为有界可学习标量。由于 `A_s\in[0,1]`，始终有：

\[
0\le W_s\le R_s,
\]

这使“Reliability 是注入上界”成为可检验而非仅口头描述的性质。

融合损失不得反向更新 `R_s`；Reliability 主要由局部退化监督学习，避免它为了降低融合损失而自行抬高或压低。

## 6. Reliability 监督与表述边界

### 6.1 局部受控退化

在训练 patch 内生成软边界局部掩膜 `M_d`，覆盖比例先固定为 10%–40%。构造：

\[
\tilde S=(1-M_d)\odot S+M_d\odot(S\odot N),
\qquad N\sim Gamma(L,1/L).
\]

当前数据加载代码把 8-bit PNG 灰度值直接除以 255，无法确认其是线性 SAR intensity。因此论文只能称其为 **controlled multiplicative corruption**，不能称为严格物理 speckle 仿真。

退化掩膜必须与现有 SAR 有效像素掩膜相交；无效/填充值区域不得参与 `L_rank`、`L_unchanged`、`L_str` 或 Reliability 定位评价。乘性退化后将数值裁剪到训练动态范围 `[0,1]`。

### 6.2 核心版只保留两个 Reliability 损失

局部排序：

\[
L_{rank}=
\frac{\|M_d\odot\max(0,m+\tilde R_s-R_s)\|_1}
{\sum M_d+\epsilon}.
\]

未退化区域一致性：

\[
L_{unchanged}=
\frac{\|(1-M_d)\odot(R_s-\tilde R_s)\|_1}
{\sum(1-M_d)+\epsilon}.
\]

\[
L_{rel}=L_{rank}+\lambda_uL_{unchanged}.
\]

核心版暂不加入 `L_smooth` 与 `L_cons`。这样可以先回答局部排序本身是否足以产生有用 Reliability，而不是让其他正则替它完成任务。

### 6.3 论文表述边界

`R_s` 首先只能称为“局部相对稳定性/可靠性分数”，不能称为校准概率。局部退化排序证明的是模型对合成退化的敏感性，不能自动证明其识别了所有真实 speckle 或物理不确定性。

只有在以下证据同时成立后，才可以使用“Reliability Map”作为主表述：

- 已知退化掩膜的 AUROC/AUPRC 明显高于随机；
- `R_s` 随退化强度单调下降；
- 在未见退化类型上仍有定位或鲁棒性收益；
- 定性上没有系统性压制 SAR 独有的建筑、船舶、岸线等结构。

否则应降级称为“degradation-aware stability gate”。

## 7. 共同融合损失与必做控制

当前 Stage II 使用：

\[
L_{old}=\|F-\max(O,S)\|_1
+10\|\nabla F-\max(\nabla O,\nabla S)\|_1.
\]

这会把 SAR 中的强响应和高频噪声一起视为目标。并且当前 `--gradient-weight` 只控制 Stage I，Stage II 的系数 `10` 是硬编码。实现新方法前必须增加 `--stage2-gradient-weight`。

所有新架构对照使用同一个共同损失，避免把损失收益误判为门控收益：

\[
L_{common}=L_{opt}+\lambda_{str}L_{str}.
\]

\[
L_{opt}=\|F-O\|_1+0.5(1-SSIM(F,O)).
\]

先定义滤波、截断后的 SAR 结构响应：

\[
S_{str}=clip_{p99}(|\nabla G_{\sigma}(S)|),\qquad \sigma=1.
\]

对无 Reliability 的方法令 `R=1`，对核心模型使用 `sg(R_s)`：

\[
G_{tar}=\max(|\nabla O|,\operatorname{sg}(R)\odot S_{str}),
\]

\[
L_{str}=\||\nabla F|-G_{tar}\|_1.
\]

核心模型的总损失先冻结为：

\[
L=L_{common}+\lambda_{rel}L_{rel}+\lambda_dL_{decomp}.
\]

冻结 Encoder 时 `\lambda_d=0`；联合微调时先用 `0.2`。核心版不加入 `L_cons` 和 `L_smooth`。

### 必做的损失匹配基线

必须有以下两条基线：

- `B0`：当前训练后 CDDFuse，使用原始最大强度/最大梯度损失；
- `B1`：架构不变的 CDDFuse，改用与新方法完全相同的 `L_common`。

如果 `B1` 已经解释了大部分 NAB/F 和 MS-SSIM 改善，论文就不能把该收益归因于 Reliability。

## 8. 分步实施计划与判停规则

### 阶段 0：冻结协议与修正控制接口

#### 目标

建立所有后续实验共同的可比基础，不加入新模型。

#### 只做以下工作

1. 固定现有 QXS-SAROPT train/validation/test 清单和基线检查点；
2. 固定 YCbCr、动态范围、14 项指标实现及六项主指标；
3. 新增可配置的 Stage II gradient weight；
4. 实现并训练 `B1` 同损失基线；
5. 记录每图指标、参数量、推理时间和典型场景 ID。

#### 六项主指标

`NAB/F↓、QAB/F↑、VIF↑、SSIM↑、MS-SSIM↑、FMI↑`。

AG、SF 只作为高频统计，不单独作为效果变好的证据。

#### 阶段结论

- 若 `B1` 已达到预期改进的 80% 以上，优先研究损失设计，暂缓复杂 Reliability；
- 若 `B1` 仍明显残留散斑和伪边缘，则进入阶段 1。

### 阶段 1：只验证内容门与残差注入

#### 模型

加入 `T_o/T_s/T_\Delta/A_s/\gamma`，令 `R_s=1`。不实现 ReliabilityNet，不加入任何 Reliability loss。

\[
D_f=\bar D_o+\gamma A_s\odot\Delta D_s.
\]

#### 训练

- 加载同一个基线检查点；
- 冻结 Encoder 和 BaseFuse；
- 训练新 Detail 头，Decoder 使用 0.1 倍学习率弱微调；
- 先做一次 800/200 小规模工程试跑；
- 工程正常后只训练 seed 42 的完整数据。

#### 对照

- `B1`：同损失、原 DetailFuse；
- `A1`：同损失、残差注入 + `A_s`。

#### 进入阶段 2 的条件

`A1` 相比 `B1` 至少满足：

1. NAB/F 有明确下降；
2. QAB/F 和 VIF 均保持在 `B1` 的 95% 以上；
3. 输出未退化为接近 Optical 的平凡解；
4. `A_s` 不是全 0、全 1 或几乎无方差。

若不满足，先修正残差候选的尺度、初始化或共同损失，不得通过添加 Reliability、多尺度和更多损失掩盖基础注入路径的问题。

### 阶段 2：加入最小 Reliability，形成核心模型

#### 模型

在阶段 1 上只增加最小 `H_R[D_s,C_{str}]`、局部退化、`L_rank` 和 `L_unchanged`：

\[
W_s=sg(R_s)\odot A_s.
\]

不加入 `C_scale`、`C_cross`、`L_smooth`、`L_cons`。

#### 核心消融

| 编号 | 变体 | 要回答的问题 |
|---|---|---|
| A1 | `A_s` only，`R_s=1` | 内容门是否已足够 |
| A2 | `R_s` only，`A_s=1` | 可靠性上界是否已足够 |
| A3 | `sg(R_s)×A_s` | 二者是否互补 |

三者使用相同初始化、共同损失、训练轮数和数据划分。

#### 训练顺序

1. 冻结 Encoder/BaseFuse 训练新模块 10–15 epochs；
2. 若验证稳定，再以十分之一学习率联合微调 15–25 epochs；
3. 首轮只做 seed 42；
4. 只有 A3 通过判定后才补 seed 3407 与 2026。

#### 核心通过条件

A3 必须同时满足：

- 相比 A1，NAB/F 或受控退化 robustness AUC 有实质改善；
- QAB/F、VIF 不出现超过 5% 的共同下降；
- Reliability 定位 AUROC 建议不低于 0.75，最低不得低于 0.65；
- A3 优于 A1 或 A2，而不是三者几乎等价；
- SAR 独有结构 ROI 未被系统压制。

#### 判停与降级

- 若 A1≈A3：删除 Reliability，研究收缩为自适应 Detail 注入；
- 若 A2≈A3：删除内容需求门，避免强行保留双门控；
- 若融合指标改善但 Reliability AUROC `<0.65`：改称 degradation-aware gate；
- 若仅 NAB/F 改善但 QAB/F、VIF 同时下降超过 10%：判定为简单抑制 SAR，不通过。

### 阶段 3：验证中心命题，不增加模型复杂度

阶段 3 的任务是补证据，不是继续加模块。

#### 3.1 Reliability 定位

在固定 200 对测试子集上，对每个样本生成三档强度、每档三个 realization，报告：

- `1-R_s` 检测 `M_d` 的 AUROC、AUPRC；
- 退化区 `mean(R_s-\tilde R_s)`；
- 未退化区 `mean(|R_s-\tilde R_s|)`；
- 不同强度下分数下降的单调性。

#### 3.2 未见退化鲁棒性

训练只使用乘性退化，测试额外使用：

- 加性噪声一个固定档位；
- Gaussian blur 一个固定档位；
- 2 px 平移作为扩展项。

报告相对 clean 的性能下降和 robustness AUC，不为每种退化重新调参。

#### 3.3 SAR 独有结构保护

预注册建筑强散射、船舶、岸线、道路等 ROI。检查低 Reliability 是否错误覆盖这些目标，并报告 A1/A2/A3 的局部放大图。

#### 3.4 三随机种子与统计

只有核心模型通过 seed 42 后，才对 `B1` 与 `A3` 补齐三个种子。报告均值±标准差、paired bootstrap 95% CI，并对六项主指标做配对检验和 Holm 校正。

### 阶段 4：按失败模式一次只加一项增强

任何增强都必须由阶段 2–3 的明确失败模式触发；每次只加一项，并与 A3 比较。没有收益就删除，不进入最终模型。

| 优先级 | 触发条件 | 唯一允许的增强 | 与中心贡献的关系 |
|---:|---|---|---|
| 1 | Reliability 对结构尺度变化敏感、稳定目标漏检 | 增加 `C_scale` | 直接增强“稳定性”证据 |
| 2 | 定位有效，但退化后融合输出仍变化大 | 增加 `L_cons`，权重中的 `R` 必须 stop-gradient | 强化可靠性约束的输出效果 |
| 3 | 平坦区 speckle 被高估，且内部结构 cue 无法区分 | 增加轻量 `C_cross` | 只作辅助，防止误判 |
| 4 | 定位正确但 map 存在明显椒盐跳变 | 增加联合边缘感知 `L_smooth` | 仅改善空间正则性 |
| 5 | Base 特征出现实证性 collapse | 做 `L_decomp=0/0.2/2.0` 对照 | 诊断母框架，不作为主创新 |

`C_cross` 的引入顺序必须晚于 `C_scale`，因为跨模态不一致既可能是噪声，也可能是 SAR 独有的有效结构。最终模型建议最多保留一到两项经证实有效的增强，而不是把表中所有项相加。

## 9. 最小实验矩阵

### 第一轮必须完成

| 编号 | 架构 | 损失 | 目的 |
|---|---|---|---|
| B0 | 原 CDDFuse | 原 max-intensity/max-gradient | 当前直接母基线 |
| B1 | 原 CDDFuse | `L_common` | 排除损失替换的影响 |
| A1 | 残差候选 + `A_s` | `L_common` | 验证自适应注入 |
| A2 | 残差候选 + `R_s` | `L_common+L_rel` | 验证可靠性上界 |
| A3 | 残差候选 + `sg(R_s)×A_s` | `L_common+L_rel` | 核心模型 |

### 第一轮不做

- 所有 cue 的全排列；
- 所有 loss 的全排列；
- 通道级 Reliability；
- 语义监督、扩散模型或新编码器；
- 多种 SAR 退化联合训练；
- 大规模超参数网格；
- 在核心机制尚未通过时复现大量 SOTA。

## 10. 数据、评价与结论边界

继续使用现有 QXS-SAROPT 固定划分进行直接比较，但其 patch 可能来自少数大场景且相邻区域重叠。它适合首轮方法验证，不足以单独证明地理泛化。最终论文至少增加以下一项：

- Dongying 跨数据集测试；
- 按原始场景或地理区域互斥的重新划分；
- 其他公开 Optical–SAR 数据集上的零样本测试。

颜色处理必须明确：当前网络只融合 Optical 的 Y 通道和灰度 SAR，输出再与 Optical Cb/Cr 合成。论文应称为亮度—结构融合，不能夸大为完整光谱融合。

### 最低成功标准

相对 `B1` 而非只相对 `B0`，A3 至少满足：

1. NAB/F 相对下降 10% 以上；
2. QAB/F 和 VIF 均保持 95% 以上；
3. SSIM 或 MS-SSIM 至少一项显著改善，另一项下降不超过 1%；
4. 受控退化 robustness AUC 相对提高 10% 以上；
5. Reliability AUROC 不低于 0.75，并呈现正确的退化强度单调性。

前四条至少满足三条，其中第 1 条必须满足；第 5 条决定是否可以保留“Reliability”主张。

## 11. 实现与训练 CMD 命令模板

以下命令只规定拟实现脚本的接口，不在计划阶段执行训练或大批量测试。

### 11.1 阶段 0：同损失基线 B1

```bat
cd /d D:\#imageInfusion\CDDFuse

.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant loss_matched_baseline ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --init-checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
  --output-dir models\reliability_v4\B1_seed42 ^
  --epochs 30 ^
  --seed 42 ^
  --amp
```

### 11.2 阶段 1：A-only 工程试跑

```bat
cd /d D:\#imageInfusion\CDDFuse

.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant demand_only ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --init-checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
  --output-dir models\reliability_v4\A1_pilot ^
  --train-limit 800 ^
  --val-limit 200 ^
  --epochs 10 ^
  --freeze-epochs 10 ^
  --seed 42 ^
  --amp
```

### 11.3 阶段 2：核心模型 A3

```bat
cd /d D:\#imageInfusion\CDDFuse

.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant reliability_demand ^
  --reliability-cues detail,structure ^
  --rel-loss rank,unchanged ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --init-checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
  --output-dir models\reliability_v4\A3_seed42 ^
  --epochs 40 ^
  --freeze-epochs 15 ^
  --head-lr 1e-4 ^
  --finetune-lr 1e-6 ^
  --decomp-weight 0.2 ^
  --rank-margin 0.2 ^
  --lambda-rel 1.0 ^
  --seed 42 ^
  --amp
```

### 11.4 阶段 3：Reliability 与鲁棒性评价

```bat
cd /d D:\#imageInfusion\CDDFuse

.venv\Scripts\python.exe evaluate_reliability_robustness.py ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --checkpoint models\reliability_v4\A3_seed42\best.pth ^
  --sample-count 200 ^
  --realizations 3 ^
  --corruptions multiplicative,gaussian,blur,shift2 ^
  --save-maps ^
  --output-dir test_result\reliability_v4\A3_seed42
```

## 12. 预计节奏与交付闸门

| 阶段 | 工作 | 交付物 | 未通过时怎么办 |
|---|---|---|---|
| 0 | 协议冻结、Stage II 权重接口、B1 | 可比基线与每图结果 | 若损失已解决问题，收缩研究问题 |
| 1 | A-only 残差注入 | A1 checkpoint、门控统计 | 只修注入路径，不加 Reliability |
| 2 | 最小 Reliability 与 A1/A2/A3 | 核心消融、seed42 结果 | 按判停规则删除无效组件 |
| 3 | 定位、鲁棒性、ROI、三种子 | 核心主张的完整证据 | 降级表述或停止复杂化 |
| 4 | 单项触发式增强 | 至多 1–2 个被保留增强 | 无明确收益立即删除 |

按周安排可设为：第 1 周完成阶段 0–1；第 2 周完成阶段 2；第 3 周完成阶段 3；第 4 周只做被触发的阶段 4 和跨数据集测试。不要预先承诺把所有增强做进最终模型。

## 13. 最终论文结构建议

方法章节只保留三段：

1. CDDFuse Detail 的无条件注入问题；
2. 局部反事实相对可靠性学习；
3. 可靠性有界的内容自适应残差注入。

实验章节按因果链组织：

```text
原损失/原架构
    ↓ 同损失控制
损失是否解释提升？
    ↓ A-only
自适应注入是否有效？
    ↓ R-only 与 R×A
Reliability 是否必要且不可被 A 替代？
    ↓ 定位/未见退化/ROI
Reliability 的语义是否成立？
```

这条链能够让每一步都对应中心贡献，也能在任一步失败时及时收缩结论，而不是继续堆模块。

## 14. 参考资料

- [CDDFuse，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zhao_CDDFuse_Correlation-Driven_Dual-Branch_Feature_Decomposition_for_Multi-Modality_Image_Fusion_CVPR_2023_paper.html)
- [CDDFuse 官方代码](https://github.com/Zhaozixiang1228/MMIF-CDDFuse)
- [SAMDFuse：退化鲁棒与语义感知 SAR–Optical 融合](https://www.sciencedirect.com/science/article/pii/S0924271626002303)
- [SAMDFuse 官方模块实现](https://github.com/christie6133/SAMDFuse)
- [CSP-MCIN：相关性与语义先验引导的多尺度跨模态融合](https://www.mdpi.com/2072-4292/18/7/975)
- [QDFNet：缺失模态场景下的质量感知 Optical–SAR 检测融合](https://arxiv.org/abs/2512.22447)
