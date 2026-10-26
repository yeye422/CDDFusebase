# SRR Anchor 四象限语义审计

## 审计目的

检查当前Anchor规则是否能够作为SAR fusion reliability的监督，重点审查负Anchor：

\[
M^-=[R_0\le0.05]\land[E_x\le Q_{0.25}(E_x)]
\]

审计使用val200中24个等间隔位置，只执行前向推理。每幅图按以下四象限选择一个局部支持度较高且中心像素严格满足条件的64×64裁剪，最后每类保留6个样本：

1. \(R_0高,E_x高\)；
2. \(R_0高,E_x低\)；
3. \(R_0低,E_x高\)；
4. \(R_0低,E_x低\)。

每个样本同时展示Optical、SAR、\(R_0\)、\(E_x\)、当前门控 \(g\)、C0输出和SRR输出。

本审计是定性语义检查，不估计各类区域在数据集中的比例，也不能替代具有专家标注的可靠性真值。

## Router主要复制了固定先验

四类样本的中心门控表现非常明确：

| 象限 | 中心门控的典型范围 |
|---|---:|
| \(R_0高,E_x高\) | 约0.9992–1.0000 |
| \(R_0高,E_x低\) | 约0.99997–1.0000 |
| \(R_0低,E_x高\) | 约0.052–0.213 |
| \(R_0低,E_x低\) | 约0.080–0.166 |

尤其值得注意：

- 当 \(R_0\) 高、\(E_x\) 几乎为零时，\(g\)仍接近1；
- 当 \(R_0\) 低、\(E_x\)处于高分位时，\(g\)仍保持低值。

这说明后期Router的主要决策近似为：

\[
g\approx h(R_0)
\]

而不是：

\[
g=f(D_o,D_s,E_x,R_0)\approx\text{fusion utility}
\]

这与当前监督捷径一致：\(R_0\)既作为Router输入，又直接生成正Anchor。

## 四象限的视觉语义

### 1. \(R_0高,E_x高\)

该组多数表现为强边缘、道路/田块边界和明显SAR结构。作为“存在显著SAR结构证据”的样本基本合理。

但它仍只能证明结构显著，不能证明加入这些信息一定改善融合。

### 2. \(R_0高,E_x低\)

该组包含：

- 光学中不明显的孤立强散射点；
- SAR中清晰但光学对应较弱的结构；
- 可能存在位置偏差的长边缘；
- 强局部散射簇。

它证明“跨模态不一致不等于噪声”。其中部分样本确实可能是SAR补充信息，也可能包含错位或强散射伪影，仅靠 \(R_0\)不能区分。

### 3. \(R_0低,E_x高\)

该组出现了清晰道路边界、区域分界和两模态共同支持的结构，但固定线结构先验中心响应低。

这说明 \(R_0\)对某些可比较结构存在漏检。当前规则没有把它们作为正Anchor，而训练后的Router在这些位置仍给出低门控，表明门控语义受 \(R_0\)主导。

### 4. \(R_0低,E_x低\)

这是当前负Anchor来源。6个代表裁剪显示它并非统一的“纯speckle”类别，而是混合了：

- 暗区或均匀区域中的随机颗粒，较符合speckle负样本；
- 光学中可见的道路、田块或区域边界，但SAR响应弱或位置不一致；
- SAR中的局部亮点和非线性散射响应；
- 普通弱纹理、低对比结构和可能的固定图漏检；
- 光学/SAR差异较大的区域。

因此：

\[
M^-
\not\equiv
\text{Speckle}
\]

更准确地说：

\[
M^-
=
\{\text{speckle},\text{weak structure},\text{misregistration},
\text{SAR-specific response},\text{prior miss},\ldots\}
\]

人工视觉检查无法给出每个SAR亮点的物理真值，但已经足以否定“该规则产生语义纯净负标签”的假设。

## 正负Anchor不在同一语义轴上

正Anchor定义：

\[
R_0高
\]

它主要表示SAR内部线结构证据强。

负Anchor定义：

\[
R_0低\land E_x低
\]

它同时混合SAR内部证据弱和跨模态对应弱。

所以模型实际学习的是一个复合规则：

\[
g=f(\text{line evidence},\text{cross-modal agreement})
\]

而监督和论文表述却将其解释为统一的：

\[
\text{SAR reliability}
\]

该语义没有闭合。

## 对V6结果的修正解释

V6后期达到：

\[
\bar g_{pos}\approx0.99,
\qquad
\bar g_{neg}\approx0.11
\]

不能解释为“模型准确识别了可靠SAR结构”。结合本审计，更准确的解释是：

> Router越来越准确地拟合了由固定先验和动态跨模态分位数定义的Anchor划分。

因此以下两件事可以同时发生：

\[
\text{Anchor classification quality}\uparrow
\]

\[
\text{Fusion utility estimation quality}\downarrow
\]

这正是V6中Anchor间隔快速增长、融合指标不升反降的合理解释。

## 结论

本次审计支持以下结论：

1. 当前负Anchor标签噪声较大，不能作为SAR fusion reliability的可靠监督；
2. 当前正负Anchor不是同一语义轴的两个端点；
3. \(R_0\)输入与正标签同源，Router存在明显监督捷径；
4. V6证明了Router能够拟合Anchor，未证明Router思想本身无效；
5. 应终止当前V6 Anchor路线，包括继续调整阈值、BCE权重和ranking margin；
6. 是否保留Router，需要用融合效用监督或独立人工审查标签重新验证。

## 文件索引

- 四象限总图：`Rhigh_Ehigh.png`、`Rhigh_Elow.png`、`Rlow_Ehigh.png`、`Rlow_Elow.png`
- 放大样本：各同名子目录；
- 样本坐标和数值：`samples.csv`；
- 抽样规则：`audit_config.json`。

注意：先前未强制候选中心属于目标象限的两个审计目录仅用于调试，不应引用。本目录 `srr_anchor_quadrant_audit_24_center_checked` 是修正后的有效版本。
