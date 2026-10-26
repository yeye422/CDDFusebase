# CDDFuse 光学–SAR 融合最终执行方案 V3.1

日期：2026-10-22  
方法暂名：**PDCF（Predictability-Driven Decomposition and Complementary Fusion）**

## 1. 本次修订结论

保留 PDCF 的两个核心方向：

1. 用跨模态可预测性重新组织 CDDFuse 的 Base/Detail 特征；
2. 对可预测信息与预测残余采用不同的融合路径。

本版对 V3.0 做以下关键修正：

- **数据划分保持现状**：继续使用现有 QXS-SAROPT 随机 split，不改为场景或地区隔离，不重建数据划分；
- 现有历史 C0 与后续实验继续使用同一 split manifest、预处理和评价实现，以保持直接可比；
- 相关性损失消融降为解释性实验，不再作为进入 PDCF 的硬门槛；
- 先验证冻结特征上的跨模态可预测性，再开发可学习重分配；
- 补全 Predictable/Residual 的张量接口、消融定义、联合训练约束和通过条件；
- SPD 仅在主方法成立后作为可选辅助分支加入。

本方案不声称随机图片级 split 能证明跨地区泛化。若论文需要场景泛化结论，应另建扩展实验，但不得与当前 C0 指标直接混排。

## 2. 固定实验协议

### 2.1 基线与数据

- 正式基线：C0 frequency-minimax epoch-5 checkpoint；
- 数据：现有 QXS-SAROPT Optical–SAR 配对数据；
- split：`datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json`；
- 保持现有随机划分，不做场景、地区或相邻图块重新隔离；
- `train24/val24`：单元功能和 smoke 审计；
- `val200`：快速淘汰，不用于最终结论；
- 完整 `val1000`：候选选择和阶段验收；
- `test1000`：结构、超参数和 checkpoint 选择规则冻结后才运行；
- 所有正式实验必须写入新的唯一输出目录并登记到 `EXPERIMENT_REGISTRY.json`。

### 2.2 C0 的准确定位

当前 C0 使用 `fusion_only` 训练，`phase1_epochs=0`、`decomp_weight=0`，Encoder 和 Decoder 冻结。因此：

- 不能把 C0 描述为在当前 frequency-minimax 训练阶段继续使用相关性损失；
- 当前 Encoder 可能继承上游 CDDFuse 预训练产生的 Base/Detail 表示偏置；
- PDCF 的主要问题是冻结表示是否可被重新组织并改善融合，而不是先证明 C0 的当前损失仍受相关性项控制。

## 3. 明确的网络接口

冻结 C0 Encoder 得到：

\[
(B_m,D_m)=E_{C0}(I_m),\qquad m\in\{O,S\},
\]

其中当前实现中 `B_m`、`D_m` 均为 64 通道。

### 3.1 可逆特征重分配

\[
\begin{aligned}
B'_m &= B_m+f_\theta(D_m),\\
D'_m &= D_m+g_\theta(B'_m).
\end{aligned}
\]

`f`、`g` 的输入输出均为 64 通道，最后一层零初始化。逆变换为：

\[
D_m=D'_m-g_\theta(B'_m),\qquad
B_m=B'_m-f_\theta(D_m).
\]

恒等初始化、前向/逆向重建误差和混合精度数值稳定性必须有单元测试。

### 3.2 跨模态预测与残余

先分别归一化分支，再拼接：

\[
H_m=[N_B(B'_m),N_D(D'_m)]\in\mathbb R^{128\times H\times W}.
\]

双向预测器输出目标模态坐标系中的表示：

\[
C_S=P_{O\rightarrow S}(H_O),\qquad
C_O=P_{S\rightarrow O}(H_S),
\]

\[
R_S=H_S-C_S,\qquad R_O=H_O-C_O.
\]

术语严格限定：

- `C_O/C_S` 是跨模态可预测表示，不称为真实共享结构；
- `R_O/R_S` 是预测残余或互补候选，不称为纯互补信息，更不称为噪声；
- 两个预测输出位于不同目标模态坐标系，送入融合前必须经过各自的对齐投影，不能直接假定同构。

预测损失按分支等权计算，避免 Base 能量主导：

\[
L_{pred}=\frac12\sum_{X\in\{B,D\}}
\left[
\rho(\widehat X_{S|O},\operatorname{sg}(X_S))/s_X+
\rho(\widehat X_{O|S},\operatorname{sg}(X_O))/s_X
\right],
\]

其中 `s_X` 使用冻结训练集统计量或特征 RMS。第一版预测器使用受限感受野的轻量卷积；降采样只作为消融，不默认把它当作配准解决方案。

### 3.3 双路径融合和 Decoder 接口

先将各模态表示投影到统一的 64 通道工作空间：

\[
Z_C=\Phi_C(A_O(C_O),A_S(C_S),H_O,H_S),
\]

\[
Z_U=\Phi_U(U_O(R_O),U_S(R_S),H_O,H_S).
\]

联合输出头必须明确产生两个 64 通道修正量：

\[
(\Delta B,\Delta D)=Q([Z_C,Z_U]),
\]

\[
B_F=B_F^{C0}+\Delta B,\qquad
D_F=D_F^{C0}+\Delta D.
\]

`Q` 的最后一层零初始化，从而在初始化时严格复现 C0。输出继续送入原 CDDFuse Decoder。

## 4. 修订后的执行阶段

### P0：冻结协议和一致性验证

1. 固定 C0 checkpoint、SHA-256、split manifest 和评价 profile；
2. 验证加载后的 Encoder 与预期父 checkpoint 的参数一致性；
3. 验证 PDCF 全部新增模块在零初始化时与 C0 输出逐像素一致；
4. 保存 train24/val24 配对可视化、配置和运行元数据。

通过条件：最大输出绝对误差不超过预先设定的浮点容差，且所有正式来源信息可追踪。

### P1：冻结特征可预测性审计

本阶段不训练 refactor 和新融合模块，只训练容量受限的双向预测器。

必须报告：

- Base、Detail 分支各自的 train/val 预测误差；
- 与均值预测、同位置线性投影、单向预测的比较；
- Predictable/Residual 的能量比例；
- 对小位移、模态错配和跨样本置换的敏感性；
- 把 `R` 置零、空间打乱或换成等能量噪声后的 Decoder 输出变化。

继续条件：验证集预测效果稳定优于简单基线，且 Residual 对融合输出存在可测量、非随机替代能够复制的边际作用。

失败处理：停止可预测分解路线，转向参数量匹配的 Fusion-only 改造。

### P1-R：相关性损失解释性实验，可并行且非门槛

如需要解释上游 Base/Detail 表示来源，可训练：

| 编号 | 相关性目标 |
|---|---|
| D0 | 原始 `cc_D²/(1.01+cc_B)` |
| D1 | 不使用相关性目标 |
| D2 | 仅最小化 `cc_D²` |
| D3 | 使用有界的 `1-cc_B`，不使用倒数形式 |

四组必须从同一上游初始化开始，使用匹配的重建、融合预算和梯度尺度。每个新 Encoder 都必须重新训练其匹配 Fusion；这些结果不得直接代替当前 C0。

### P2：最小可预测分解

1. 冻结 C0 Encoder、Decoder 和原 Fusion；
2. 在原始 `H` 上训练预测器；
3. 用固定 Predictable/Residual 接参数量匹配的单路 concat-conv；
4. 暂不启用可逆 refactor；
5. 验证分解是否比直接拼接同等容量融合更有效。

继续条件：完整 val1000 上达到预注册门槛，并通过 Residual 置零、打乱和噪声替代检查。

### P3：可逆重分配与双路径融合

P2 通过后加入 refactor，并采用交替更新：

1. 固定 refactor，更新 predictor；
2. 固定或低学习率更新 predictor，更新 refactor 与 Fusion；
3. 联合阶段始终保留预测损失和身份约束。

联合目标：

\[
L=L_{C0\_fusion}
+\lambda_{pred}L_{pred}
+\lambda_{id}L_{identity}
+\lambda_{drift}L_{feature\_drift}.
\]

其中 `L_feature_drift` 约束新 Base/Detail 相对 C0 特征的归一化 RMS 偏移。必须记录四项损失、梯度范数、Predictable/Residual 能量和输出修正量，防止语义漂移。

### P4：SPD 辅助表征

仅在 P3 主方法成立后进入：

1. 冻结现有 SPD；
2. 明确选择 SPD 的具体中间层、通道数和预处理；
3. 用独立 Adapter 投影后，仅作为 Complementary Fusion 的条件输入；
4. 与无 SPD、原 SPD、同参数量无参考辅助分支比较；
5. 只有冻结 SPD 显示独立收益后，才考虑额外多时相 SAR 训练。

## 5. 可执行消融矩阵

| 实验 | 分解输入 | 融合结构 | Refactor | SPD | 目的 |
|---|---|---|---|---|---|
| E0 | 无 | 原 C0 Fusion | — | — | 正式基线 |
| E1 | Predictable/Residual | 参数匹配单路 concat-conv | — | — | 分解本身是否有效 |
| E2 | 固定 `C=(H_O+H_S)/2`、`R_m=H_m-C` | 双路径 | — | — | 双路径结构是否独立有效 |
| E3 | Predictable/Residual | 双路径 | — | — | 分解与融合能否协同 |
| E4 | 原始 `H_O/H_S` | 与 E3 参数量、深度匹配的 concat-conv | — | — | 排除容量收益 |
| E5 | Predictable | 仅 Correspondence | — | — | 对应路径贡献 |
| E6 | Residual | 仅 Complementary | — | — | 残余路径贡献 |
| E7 | Predictable/Residual | 双路径 | ✓ | — | 可逆重分配增益 |
| E8 | Predictable/Residual | 双路径 | ✓ | 原 SPD | SPD 辅助价值 |
| E9 | Predictable/Residual | 双路径 | ✓ | 增强 SPD | 额外 SPD 监督价值 |

所有容量对照必须匹配输入通道、输出通道、卷积深度、感受野、参数量和训练步数。

## 6. 评价与预注册规则

### 6.1 指标角色

主指标预先固定为：

- QAB/F：主要结构保留；
- VIF：主要信息保留；
- FMI：主要特征保留；
- NAB/F：伪影硬约束，越低越好。

MI、EN、SCD、SSIM、MS-SSIM、AG、SF、SD 为辅助指标，不允许以单个辅助指标上涨宣称通过。

### 6.2 统计规则

- 所有比较使用逐图配对差值；
- 报告均值、标准差、95% bootstrap 置信区间和胜率；
- 最终候选至少三个训练随机种子；
- 以 seed 为复现单位汇总，不把多张图片误当成独立训练重复；
- checkpoint 只按预先固定的 val1000 选择分数确定；
- 禁止从 test1000 反向选择 checkpoint、超参数或结构。

### 6.3 阶段通过原则

进入正式实验前，在配置中填写数值阈值；至少满足：

1. 三个主指标中至少两个改善，且其逐图差值 95% CI 不跨 0；
2. 另一个主指标不得超过预设退化容差；
3. NAB/F 不得超过预设恶化容差；
4. 改善必须跨至少三个 seed 保持同方向；
5. 相比参数量匹配对照仍有独立优势。

具体最小效应量不得在看到正式结果后再决定。

## 7. 代码组织

```text
models/
  pdcf/
    feature_refactor.py
    cross_modal_predictor.py
    decomposition.py
    correspondence_fusion.py
    complementary_fusion.py
    joint_projection.py
    spd_reference_adapter.py

training/
  train_predictability_audit.py
  train_pdcf_single_path.py
  train_pdcf_dual_path.py
  train_pdcf_spd.py

evaluation/
  evaluate_pdcf.py
  compare_pdcf_ablation.py

configs/
  pdcf_predictability/
  pdcf_single_path/
  pdcf_dual_path/
  pdcf_spd/
```

优先复用现有数据集、C0 加载、指标、元数据、checkpoint 和注册表逻辑，不建立第二套不可比较的评价实现。

## 8. 最终执行决策

执行顺序固定为：

```text
P0 协议与恒等性
  → P1 冻结特征可预测性
  → P2 最小分解 + 单路融合
  → P3 可逆重分配 + 双路径融合
  → P4 SPD 辅助
```

相关性损失实验 `P1-R` 是解释性支线，不阻断 PDCF。数据继续保持现有随机 split；所有新结果只与使用同一 manifest、预处理、checkpoint 口径和评价 profile 的实验直接比较。

本文件是开发计划修订，尚未修改模型代码，也未执行训练或批量测试。
