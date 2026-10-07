# SRR DetailFuse 可靠性路由方案收尾总结

## 1. 文档目的

本文档对斑点感知结构可靠性路由方案（SRR DetailFuse）进行正式收尾，记录其研究动机、方法架构、版本演进、实验结果、成因干预、失败结论和可复用资产。

该方案现已停止继续开发，不进入正式8000对独立完整训练，也不作为当前最终融合模型。当前综合基线仍为 C0 frequency-minimax。

需要特别说明：V6 第6–10轮续训命令遗漏了 `--train-limit 800 --val-limit 200`，因此该阶段实际使用8000对训练和1000对内部验证。该结果只能用于探索性诊断，不能作为与C0公平比较的正式完整训练结果。

---

## 2. 研究动机

SAR图像中的高频响应同时包含真实结构、散射信息、斑点噪声和局部伪边缘，因此：

\[
\text{SAR High Frequency}
\neq
\text{Reliable SAR Structure}
\]

同时，光学与SAR不一致也不必然表示噪声：

\[
\text{Cross-modal Inconsistency}
\neq
\text{SAR Noise}
\]

方案最初希望联合跨模态对应证据和SAR内部固定结构证据，在进入细节融合前估计SAR可靠性，实现：

\[
\boxed{\text{Reliability Before Fusion}}
\]

前期C3a实验表明，固定结构图能够恢复部分SAR结构，但只将其用于损失加权无法保证推理阶段的空间选择。因此SRR进一步把固定结构先验引入前向细节融合路径。

---

## 3. 最终方案架构

方案保留CDDFuse的Encoder、BaseFuse和Decoder，仅替换原DetailFuse。

原C0路径：

\[
D_f=\operatorname{DetailFuse}(D_o+D_s)
\]

SRR路径：

\[
D_f=\operatorname{SRRDetailFuse}(D_o,D_s,R_0)
\]

其中：

- \(D_o\)：光学细节特征；
- \(D_s\)：SAR细节特征；
- \(R_0\)：FSE-v2a固定结构先验；
- \(D_f\)：路由后的融合细节。

基础路径保持：

\[
B_f=\operatorname{BaseFuse}(B_o+B_s)
\]

### 3.1 跨模态证据

光学和SAR细节分别经独立投影，在局部 \(3\times3\) 邻域内计算余弦软匹配，获得：

- 跨模态支持度 \(E_x\)；
- 匹配确定性；
- 光学/SAR特征能量。

### 3.2 SAR内部证据

使用已完成训练集校准的FSE-v2a结构图：

\[
E_s=R_0
\]

固定图不直接生成输出，而作为Router输入，并在后续版本中用于生成稀疏Anchor。

### 3.3 可靠性与双属性

Router预测：

\[
g_s=\sigma\Phi_r(E_x,E_s,\text{energy})
\]

\[
a_x=\sigma\Phi_x(\cdot)
\]

\[
a_m=\sigma\Phi_m(\cdot)
\]

其中：

- \(g_s\)：SAR细节可靠性；
- \(a_x\)：Cross-Supported属性；
- \(a_m\)：SAR-Complementary属性。

两个属性非互斥。

### 3.4 候选分支与融合

\[
D_x=\Phi_{cross}(D_o,D_s)
\]

\[
D_m=\Phi_{comp}(D_s)
\]

\[
\widetilde D_x=g_sa_xD_x
\]

\[
\widetilde D_m=g_sa_mD_m
\]

随后拼接 \(D_o,\widetilde D_x,\widetilde D_m\)，通过可学习的 \(1\times1\) 路由融合层和继承自原模型的DetailTransform生成最终细节。

### 3.5 训练目标

主目标沿用C0频率解耦目标及固定训练期参考：

\[
L_{main}
=
\operatorname{SmoothMax}
\left(
\frac{L_O}{R_O},
\frac{L_S}{R_S}
\right)
\]

后续加入稀疏可靠性Anchor：

\[
M^+=[R_0\ge0.5]
\]

\[
M^-=[R_0\le0.05]\land[E_x\le Q_{0.25}(E_x)]
\]

并采用平衡BCE和正负排序约束：

\[
L_{rank}
=
\max\left(0,m-(\bar g_{pos}-\bar g_{neg})\right)
\]

最终：

\[
L_{total}=L_{main}+0.05(L_{BCE}+L_{rank})
\]

---

## 4. 版本演进

| 版本 | 主要修改 | 结果 |
|---|---|---|
| V1 | 相似度、匹配熵和能量相乘；双候选归一化平均 | 跨模态证据约 \(2.8\times10^{-5}\)，Router停留在初始化 |
| V2 | 匹配确定性独立输入；两个候选独立通道融合 | 跨模态证据恢复至约0.016，但Router仍接近常数 |
| V3 | 增加可靠正负Anchor BCE | Anchor有效但学习率过低，Router几乎不变化 |
| V4 | 整个SRR模块学习率提高到 \(10^{-4}\) | 候选分支和DetailTransform漂移，指标明显崩溃 |
| V5 | Evidence/Router使用高学习率，其余融合参数保持 \(10^{-5}\) | 避免特征放大，Router开始学习，但主要整体下移 |
| V6 | 增加正负Anchor排序约束 | 第5轮形成轻微分离；后续训练形成强分离，但融合质量仍下降 |

最终采用的优化分组为：

| 参数组 | 学习率 | 梯度裁剪 |
|---|---:|---:|
| Cross Evidence与Router Head | \(10^{-4}\) | 0.1 |
| 候选分支、Route Fusion、DetailTransform | \(10^{-5}\) | 0.01 |
| BaseFuse | \(10^{-5}\) | 0.01 |

---

## 5. 200对验证集结果

以下C0、V1–V6第5轮均使用相同val前200对和paper/clamp评价协议。V6第10轮由于续训数据规模混杂，仅列作诊断参考。

| 方法 | MI | FMI | QAB/F | NAB/F↓ | SSIM | MS-SSIM | VIF |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 3.1129 | **0.6571** | **0.7387** | 0.03679 | **0.5303** | 0.5775 | **0.3269** |
| V1 | 2.9703 | 0.6272 | 0.7190 | **0.03649** | 0.5291 | 0.5777 | 0.3064 |
| V2 | 3.0052 | 0.6326 | 0.7248 | 0.03655 | 0.5298 | **0.5778** | 0.3112 |
| V3 Anchor | 3.0053 | 0.6325 | 0.7248 | 0.0365 | 0.5298 | **0.5778** | 0.3112 |
| V4整体高学习率 | 2.3262 | 0.4911 | 0.5965 | 0.0614 | 0.4910 | 0.5742 | 0.2220 |
| V5分组学习率 | 3.0094 | 0.6336 | 0.7253 | 0.0366 | 0.5298 | **0.5778** | 0.3117 |
| V6第5轮 | 3.0086 | 0.6322 | 0.7252 | 0.0365 | 0.5298 | **0.5778** | 0.3116 |
| V6第10轮（诊断） | 2.9987 | 0.6289 | 0.7084 | 0.0448 | 0.5273 | 0.5654 | 0.2983 |

V6第5轮相对C0：

- FMI下降约3.79%；
- QAB/F下降约1.83%；
- NAB/F轻微改善；
- VIF下降约4.67%；
- SSIM基本持平。

所有正式Pilot均未达到预注册的FMI提升目标和VIF保持目标。

---

## 6. Router学习情况

V6第5轮验证：

\[
\bar g_{pos}=0.8981
\]

\[
\bar g_{neg}=0.8821
\]

\[
\Delta g=0.0159
\]

在第6–10轮混合规模续训后：

\[
\bar g_{pos}=0.9881
\]

\[
\bar g_{neg}=0.1119
\]

\[
\Delta g=0.8762
\]

同时整体门控标准差达到约0.2268。由此可以确认：

1. Evidence与Router具备学习能力；
2. Anchor梯度路径有效；
3. 方案失败不是单纯因为没有收敛；
4. 学会Anchor规则没有带来融合质量提升。

---

## 7. 成因干预检查

使用同一V6第10轮checkpoint，在val200中固定选择6张等间隔图像，进行无训练干预。该检查用于定位成因，不用于正式性能报告。

### 7.1 门控空间布局

| 条件 | FMI | QAB/F | NAB/F↓ | VIF |
|---|---:|---:|---:|---:|
| 原门控 | 0.6244 | 0.7182 | 0.0455 | 0.3045 |
| 同均值常数门控 | 0.6157 | 0.7145 | 0.0451 | 0.3014 |
| 同直方图随机打乱 | 0.6040–0.6086 | 约0.7115 | 约0.0461 | 约0.2978 |
| 反转门控 | 0.5790 | 0.6752 | 0.0505 | 0.2698 |
| 平滑门控 | 0.6180 | 0.7156 | 0.0449 | 0.3027 |

结论：门控空间位置并非随机；学到的布局对信息保留有一定正作用。但这一收益不足以弥补整个新Detail路径相对C0的损失。

### 7.2 候选分支

| 条件 | FMI | QAB/F | NAB/F↓ | VIF |
|---|---:|---:|---:|---:|
| 完整模型 | 0.6244 | 0.7182 | 0.0455 | 0.3045 |
| 关闭Cross | 0.6122 | 0.7253 | 0.0407 | 0.3120 |
| 关闭Complementary | **0.6400** | 0.7197 | 0.0448 | 0.3062 |
| 两分支都关闭 | 0.6155 | **0.7261** | **0.0404** | **0.3127** |

Cross分支提高FMI，但伴随QAB/F、NAB/F和VIF代价。Complementary分支未显示独立正收益，关闭后四项指标均改善。

两个候选特征余弦相似度为0.9045–0.9245，且Complementary投影后RMS仅为Cross的大约1/4，说明双属性分工没有得到验证。

### 7.3 固定结构先验

固定图置零、平移或随机打乱会显著改变门控，但最终输出变化约为0.0022–0.0029。固定先验可以被Router读取，但未表现出明确的独立推理净收益。

跨模态证据与确定性同时置零后，FMI、QAB/F、NAB/F和VIF明显恶化，说明当前有效空间调节更多依赖学习到的跨模态证据。

### 7.4 Base/Detail协同适配

| Base / Detail | FMI | QAB/F | NAB/F↓ | VIF |
|---|---:|---:|---:|---:|
| C0 / C0 | **0.7056** | **0.7490** | 0.0325 | **0.3379** |
| V6 / V6 | 0.6244 | 0.7182 | 0.0455 | 0.3045 |
| C0 / V6 | 0.4666 | 0.6887 | 0.0779 | 0.2633 |
| V6 / C0 | 0.6626 | 0.7128 | **0.0180** | 0.3002 |

C0完整模型与V6完整模型输出MAE约为0.0149，但交换Base或Detail后输出MAE达到约0.12–0.13，证明BaseFuse与Detail路径形成了强协同表示，不能作为独立贡献相加解释。

---

## 8. 最终失败原因

### 8.1 结构可靠性不等于融合效用

固定图高响应只说明某位置具有稳定线结构证据，不说明把对应SAR特征加入融合一定改善结果。

### 8.2 门控没有独占控制SAR信息

SAR仍可经BaseFuse进入输出；Base/Detail还会共同适配。门控值不能解释为整个模型的SAR贡献率。

### 8.3 新模块替换了已验证的C0细节表示

即使Router近似常数，新路径也不严格等于C0，导致早期版本已经出现FMI和VIF下降。

### 8.4 门控与候选幅值不可辨识

\(g_s,a_x,a_m\)、候选特征幅值、Route Fusion和DetailTransform可以互相补偿。同一输出可由多种参数组合产生。

### 8.5 双属性缺少功能语义

Cross和Complementary没有独立目标、正交约束或专家级任务，两个名称没有转化为可验证分工。

### 8.6 Anchor存在输入—标签捷径

固定图既是Router输入，又生成正Anchor。网络可以直接学习固定图阈值规则，而无需学习融合效用。

### 8.7 负Anchor定义不可靠且会变化

低固定图和低跨模态证据不必然等于噪声；跨模态证据由可学习投影产生，使负Anchor集合随训练变化。

### 8.8 C0结构目标与Anchor可能冲突

C0中频目标可能要求输出保留某处SAR梯度，而Anchor同时要求降低该处SAR Detail门控，促使网络通过其他路径补偿。

### 8.9 标量空间门控与深层特征语义不匹配

单通道像素图同时调制64个具有较大感受野的特征通道，不能等价为只删除该像素的speckle。

### 8.10 训练和整图推理尺度不一致

训练/内部验证使用128裁剪，正式评价使用256整图；强非线性Router对这种分布差异较敏感。

---

## 9. 已排除的解释

当前证据不支持继续将失败归因于：

- Router完全无法训练；
- Anchor损失未生效；
- 单纯学习率过低；
- 仅仅训练轮数不足；
- 仅仅Pilot样本太少。

Router最终能够高度拟合Anchor，但仍不能改善融合，说明失败主要来自架构与监督定义，而非普通优化问题。

---

## 10. 仍可保留的研究资产

1. C0频率解耦与固定参考Smooth Maximum基线；
2. FSE-v2a固定结构证据及其校准流程；
3. 局部跨模态Soft Matching实现；
4. Fast/Slow参数组训练机制；
5. 路由图、分支图和Anchor诊断体系；
6. 成因干预脚本；
7. “结构可靠性不等于融合效用”的完整负实验链；
8. 门控空间布局具有一定信息价值的干预证据。

固定图不应被表述为无效。现有结果只能说明：它不适合在当前架构中充当SAR Detail贡献的直接标签和强门控依据。

---

## 11. 正式终止决定

停止以下工作：

- 继续调整SRR Anchor阈值、权重或排序间隔；
- 增加训练轮数；
- 按当前结构启动正式完整训练；
- 继续扩展Complementary分支；
- 将Speckle一致性叠加到当前模型；
- 用现有V6结果作为正式论文主表方法。

当前C0继续作为综合基线。后续研究若继续选择性融合方向，应重新定义问题为“融合效用估计”，直接比较使用SAR与不使用SAR的候选融合结果，而不是继续预测缺少真值的结构可靠性标签。

---

## 12. 实验资料索引

- C0 Pilot：`models/reliability_v5/frequency_minimax_baseline_pilot/`
- SRR V1：`models/reliability_v5/srr_detail_pilot/`
- SRR V2：`models/reliability_v5/srr_detail_v2_pilot/`
- SRR V3：`models/reliability_v5/srr_detail_v3_anchor_pilot/`
- SRR V4：`models/reliability_v5/srr_detail_v4_router_lr_pilot/`
- SRR V5：`models/reliability_v5/srr_detail_v5_split_lr_pilot/`
- SRR V6：`models/reliability_v5/srr_detail_v6_ranking_pilot/`
- 第一轮成因干预：`test_result/reliability_v5/srr_v6_e010_interventions_final/`
- 门控布局干预：`test_result/reliability_v5/srr_v6_e010_gate_layout_diagnosis_6/`
- C0/V6路径混合：`test_result/reliability_v5/srr_v6_e010_hybrid_diagnosis_6_final/`
- Anchor四象限语义审计：`test_result/reliability_v5/srr_anchor_quadrant_audit_24_center_checked/`
- FSE-v2a先验独立审计：`test_result/reliability_v5/fse_v2a_prior_audit_24/`
- 干预脚本：`diagnose_srr_interventions.py`、`diagnose_srr_hybrids.py`

现有test划分已经在此前研究过程中多次参与方法判断，不能继续表述为完全独立的最终确认集。若开展新方法，最终确认应从尚未使用的10000对数据中重新划分并冻结。
