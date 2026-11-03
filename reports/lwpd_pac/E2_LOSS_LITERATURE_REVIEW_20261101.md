# E2 损失改进：核心论文调研与过冲惩罚审查

日期：2026-11-01。PRL保持暂停。本轮仅调研和代码核对，不修改损失、不训练、不产生新的性能评价。

## 结论

此前提出的梯度过冲hinge惩罚在数学上成立，也能表达“更不接受新增强边缘”的偏好，但不足以成为已经论证有效的SAR–光学融合目标。它适合作为一个简单、可解释的消融项，不宜立即认定为最优方向或论文核心创新。

主要问题有三个：与现有最大梯度L1在监督内容上大量重叠；原始SAR最大梯度仍可能包含斑点噪声；只约束梯度幅值上界不能保证结构与互补信息保持。现有单样本PRL干预也没有验证损失改动的效果。

应先确认E2当前梯度权重的敏感性，再用单因素实验比较非对称过冲代价；若继续发展方法，应优先研究“哪些源信息值得保留”的可靠性与结构目标，而不是不断叠加梯度惩罚。

## 核心论文及可迁移结论

以下只把原文明确说明的内容列为论文结论；对本项目的建议是另行推断，不能当成原论文在QXS上的验证结果。不同数据集和指标实现的数值不作横向排名。

### 1. PMGI，AAAI 2020

Hao Zhang等，*Rethinking the Image Fusion: A Fast Unified Image Fusion Network based on Proportional Maintenance of Gradient and Intensity*。

论文把不同融合任务组织为强度与梯度的比例保持问题，为平衡两类监督提供依据，但没有证明本文提出的过冲hinge能改善SAR融合。

来源：[AAAI原论文](https://ojs.aaai.org/index.php/AAAI/article/download/6975/6829)，DOI：10.1609/aaai.v34i07.6975。

### 2. CDDFuse，CVPR 2023

Zixiang Zhao等，*CDDFuse: Correlation-Driven Dual-Branch Feature Decomposition for Multi-Modality Image Fusion*。

原文第5页式(10)采用最大强度L1、最大梯度L1及分解相关性损失。我们的E2沿用了最大强度/梯度思路，但还使用单模态重建项，不能将整个E2损失等同于原始CDDFuse。该文主要验证红外–可见光及医学融合，不能直接证明原始SAR最大梯度可靠。

来源：[作者arXiv全文](https://arxiv.org/pdf/2211.14461)。

### 3. U2Fusion，TPAMI，2020在线／2022卷期

Han Xu等，*U2Fusion: A Unified Unsupervised Image Fusion Network*。

第3.2节式(4)–(6)使用自适应源权重的SSIM与MSE；第5.5节说明，相比FusionDN，移除梯度损失可缓解假边缘，加入MSE可缓解亮度偏差。这是“梯度监督并非越强越好”的直接经验依据，但不等于应在E2中直接删掉梯度损失。

来源：[作者全文](https://hanna-xu.github.io/files/TPAMI_U2Fusion.pdf)，DOI：10.1109/TPAMI.2020.3012548。注意不要把FusionDN的梯度目标当成最终U2Fusion的目标。

### 4. SwinFusion，IEEE/CAA JAS 2022

Jiayi Ma等，*SwinFusion: Cross-domain Long-range Learning for General Image Fusion via Swin Transformer*。

期刊摘要明确列出SSIM、纹理、强度三类损失，支持结构与细节分别约束。本轮只核实该损失组成，未据摘要推断具体公式或SAR效果。

来源：[期刊页面](https://www.ieee-jas.net/article/doi/10.1109/JAS.2022.105686)，DOI：10.1109/JAS.2022.105686。

### 5. SOSTF，IEEE GRSL，2022在线接受版本

Yuanxin Ye等，*An Unsupervised SAR and Optical Image Fusion Network Based on Structure-Texture Decomposition*。

作者稿第3页式(4)–(7)组合光学像素L2、光学纹理L1和SAR结构L1；SAR结构经分解后受约束，以减少将斑点当作有效信息。它直接对应SAR–光学任务，比红外融合经验更贴近本项目，但其结构提取、SAR拉伸和训练流程均有差异，不能直接照抄权重。

来源：[作者全文](https://ktqizhi.github.io/papers/ye2022unsupervised.pdf)，DOI：10.1109/LGRS.2022.3219341。年份在此按已读作者稿标注，不混同最终卷期。

### 6. VSFF，ISPRS Annals 2023

Jiacheng Zhang等，*Optical and SAR Image Fusion Based on Visual Saliency Features*。

该文是传统优化方法而非训练损失论文。其方法区分主要结构与细节纹理，并针对SAR噪声和显著信息设计融合，说明“细节多”与“有效信息多”应分开处理。不能据此给E2新增项赋予确定权重。

来源：[ISPRS原文](https://isprs-annals.copernicus.org/articles/X-1-W1-2023/747/2023/isprs-annals-X-1-W1-2023-747-2023.pdf)，DOI：10.5194/isprs-annals-X-1-W1-2023-747-2023。

## 近期及相邻研究的边界

- **SAMDFuse，ISPRS JPRS 2026**：出版社摘要强调SAR斑点、光学退化与可靠信息选择。本轮未取得全文损失公式，不能作为所提hinge的公式依据。[出版社页面](https://www.sciencedirect.com/science/article/pii/S0924271626002303)，DOI：10.1016/j.isprsjprs.2026.04.049。
- **HS2P，Information Fusion 2023**：目标是SAR辅助的光学去云，具有光学重建目标。可参考光谱/结构协同思路，但与无真值融合任务不同，不应移植其监督目标或报告其结果为本任务证据。[作者全文](https://skyearth.org/publication/papers/2023_hs2p.pdf)，DOI：10.1016/j.inffus.2023.02.002。

## 对原过冲惩罚的数学审查

设相同梯度定义下，`g_F=G(F)`，`t=max(G(O),G(S))`，`r=g_F-t`。忽略容忍量epsilon时：

\[
c|r|+\lambda[r]_+
=\begin{cases}
c(-r),&r<0,\\
(c+\lambda)r,&r\ge0.
\end{cases}
\]

因此它等价于非对称绝对误差：欠梯度代价为c，过梯度代价为c+lambda。只考虑该像素项，其最优点仍为r=0；联合其他损失时才会通过不同代价改变折中。它不是错误设计，也不是加入了一份独立的结构真值。epsilon>0时仅在超过容忍量后提高斜率。

该推导针对“相同梯度算子”。当前代码用`|Sx|+|Sy|`，而NAB/F使用`√(Sx²+Sy²)`；若新项使用后者，它还引入了梯度范数差异，不能再声称整个组合严格等同非对称L1。为避免混淆，范数改变与非对称惩罚应单独对照。

### 局限1：并未纠正有噪目标

假设某位置光学梯度为0.02，SAR原始梯度为0.20且主要来自斑点，输出为0.18。新项为0，但模型仍然保留较强的噪声响应。仅限制“不超过两源最大值”无法区分有效SAR目标与斑点。

### 局限2：不约束方向及空间结构

梯度幅值符合上界仍可能存在方向错误、错位、双边缘或局部结构改变。它无法独立保证QAB/F、SSIM、MS-SSIM，更不保证MI、FMI、VIF。

### 局限3：与NAB/F仅部分对应

本地`utils/Evaluator.py`除过冲掩码外，还按边缘质量与源梯度加权归一化。简单平均幅值hinge不是可微NAB/F，也不能保证NAB/F单调改善。若仅追逐阈值，可通过保守输出降低该指标，却损失有用细节。

### 局限4：现有因果证据不够

PRL失败不能证明E2的梯度损失有错；PRL核置零单样本结果只能定位当前PRL的差距。损失项数值占85%也不能推出其参数梯度占85%，需要实际测量各项在共享参数上的梯度范数和夹角。此前“梯度项占比高”的描述应限于标量损失组成。

## 更稳妥的研究顺序

1. **建立权重对照**：固定E2全部结构、数据与训练预算，保留原始梯度定义，只改变梯度权重。可以预先限定10、5两个初始候选；5是工程试验值，不是论文结论。比较全部保护指标，不能假定降权必然有效。
2. **检验非对称代价**：保留相同算子，用hinge提高过冲代价；同时记录有效区域过冲分布、边缘保持及模态信息。它的研究问题是“过冲比欠冲是否应该付出更高代价”。
3. **独立检查梯度范数**：若需要与评价幅值定义一致，将G1改成G2作为独立因素，不与新惩罚、结构项一次性混改。
4. **有证据后改监督目标**：借鉴SOSTF的模态分工，研究光学外观/纹理与SAR可靠结构的分离约束。结构提取器本身可能漏掉SAR小目标，必须保存SAR独有结构；不能把光学当作全图唯一真值，也不能盲目平均约束两源SSIM。
5. **后续结构项属于另一轮实验**：可以考察固定、源图像决定的局部可靠性权重与结构相似性，不让模型通过降低置信度逃避约束；任何新平滑、结构提取或预处理都要记录参数。其形式属于待验证的设计，不是上述论文共同认可的统一标准。

所有训练和批量验证只提供CMD命令由用户执行。本轮没有生成新训练/评价结果，因此没有新增实验注册条目。

## 本地代码证据

- `train_direct_fusion.py:137`：最大强度与最大梯度L1。
- `train_direct_fusion.py:123`：SSIM位于单模态重建分支。
- `train_lwpd.py:224`：融合项与0.1倍重建项组合。
- `utils/loss.py:24`：训练Sobel幅值为绝对值和。
- `utils/Evaluator.py:183`、`:405`：评价梯度幅值与NAB/F定义。

本报告没有发现足以证明“该精确hinge形式已在同设置SAR–光学融合中有效”的证据；这不等于断言相关文献中不存在该形式。
