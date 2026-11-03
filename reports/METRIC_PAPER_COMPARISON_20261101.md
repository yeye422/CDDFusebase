# 当前14项指标计算公式与融合论文公开代码对照

日期：2026-11-01。本文只描述当前正式路径 `test_OS.py --metric-profile paper`，不将 `reference_v2_candidate` 当成已启用算法。未运行训练、模型推理或批量评价。

## 比较对象与边界

- 当前实现：`utils/Evaluator.py` 的 `PaperEvaluator`。
- CDDFuse，CVPR 2023：[论文](https://arxiv.org/pdf/2211.14461)、[作者评价代码](https://github.com/Zhaozixiang1228/MMIF-CDDFuse/blob/main/utils/Evaluator.py)。论文正文列出8项：EN、SD、SF、MI、SCD、VIF、QAB/F、SSIM；不能说其论文采用了本项目全部14项。
- SeAFusion，Information Fusion 2022：[作者公开评价目录](https://github.com/Linfeng-Tang/SeAFusion/tree/main/Evaluation)。以下核对公开文件内容；目录中有某指标函数，并不证明论文每张表都使用该函数及其当前默认参数。
- 指标原作者代码：Haghighat FAST-FMI、Wang SSIM/MS-SSIM、LIVE VIFp、Kumar modified NAB/F、Aslantas/Bendes SCD。

论文定义、作者公开代码、当前实现三者可能不同。这里把“公式一致”“只有聚合或单位差异”“算法不同”分别注明，不把某论文发布过的所有代码都视为无缺陷标准。

## 符号与输入

光学亮度为A，SAR灰度为B，融合灰度为F，尺寸H×W，N=HW。使用当前OpenCV YCrCb亮度通道，F经过clamp与四舍五入量化到uint8；计算时使用float64。以下E表示空间均值，rho表示Pearson相关，I2表示以2为底的离散互信息。源均值为二分之一，两源之和不除2。

## 当前全部14项公式

| 指标 | 当前paper计算方式 | 对照判断 |
|---|---|---|
| EN | `-sum(p_F(k)*log2(p_F(k))), k=0..255` | 与CDDFuse公式及MATLAB entropy的8位离散定义一致 |
| MI | `I2(A,F)+I2(B,F)`，256×256联合直方图 | 与CDDFuse同属双源MI之和，但后者sklearn默认自然对数，单位不同；与SeAFusion作者署名MI函数的理论log2公式一致，输入dtype仍须核对 |
| FMI | `I2(Q_A(DCT(A)),Q_FA(DCT(F))) + I2(Q_B(DCT(B)),Q_FB(DCT(F)))` | 每对DCT图共同线性量化为256箱；不是SeAFusion及Haghighat的局部归一化FAST-FMI |
| PSNR | `10*log10(255² / ((E[(F-A)²]+E[(F-B)²])/2))` | 固定峰值；与CDDFuse的max(F)峰值不同。与标准PSNR一致，但双源MSE聚合是另一个必须说明的约定 |
| QAB/F | `sum(Q_AF*g_A+Q_BF*g_B)/sum(g_A+g_B)` | 与CDDFuse和SeAFusion的Hong移植代码属于同一旧版本；不等于XP原论文全部参数/分支约定 |
| NAB/F | `sum(M*((1-Q_AF)*g_A+(1-Q_BF)*g_B))/sum(g_A+g_B)`；`M=1[g_F>max(g_A,g_B)]` | 与Kumar修改版聚合形式相似，但当前复用旧QAB/F质量与权重，不等于SeAFusion目录中的Kumar完整算法 |
| AG | 在`i<H,j<W`上平均`sqrt(((F[i+1,j]-F[i,j])²+(F[i,j+1]-F[i,j])²)/2)` | 当前为前向差分；CDDFuse为中心差分加单边边界，SeAFusion使用MATLAB gradient，归一化又不同 |
| SF | `sqrt((sum(diff_x(F)²)+sum(diff_y(F)²))/N)` | CDDFuse/SeAFusion单函数分别按各自差分数组元素数取均值，归一化不同 |
| SD | `sqrt(E[(F-E[F])²])` | 与CDDFuse及SeAFusion总体标准差一致，ddof=0 |
| CC | `(rho(A,F)+rho(B,F))/2` | 非零方差时与两论文公开函数的数学结果一致；当前零方差特殊返回0/1不属于Pearson公式 |
| SCD | `rho(A,F-B)+rho(B,F-A)` | 与CDDFuse、SeAFusion及Aslantas/Bendes作者公式一致；常量残差的未定义情况需单独处理 |
| SSIM | `(SSIM_Wang(A,F)+SSIM_Wang(B,F))/2`，11×11高斯、sigma1.5、总体协方差、范围255 | 与Wang单尺度公式吻合；CDDFuse公开代码用默认skimage SSIM再求和；SeAFusion当前wrapper调用MEF-SSIM，均不等同当前算法 |
| MS-SSIM | `(MS(A,F)+MS(B,F))/2`，最多5尺度，权重归一化，边界全图池化 | 同属Wang式多尺度结构乘积，但细节不同于原msssim.m；SeAFusion同名文件实际为多尺度MEF-SSIM，算法不同 |
| VIF | `(VIFp(A,F)+VIFp(B,F))/2`，四尺度、噪声参数2 | 像素域核心与CDDFuse/LIVE代码一致；CDDFuse和SeAFusion wrapper用两源求和，当前为均值；当前NaN转1还需修正 |

## 关键公式进一步展开

### 互信息与FMI不是同一个“换个变换”问题

离散互信息为：

\[
I_2(X,Y)=\sum_{x,y:p(x,y)>0}p(x,y)\log_2\frac{p(x,y)}{p_X(x)p_Y(y)}.
\]

当前FMI只是将X、Y替换为共同量化的全图正交DCT系数，再求两源之和。每个源与F共享该对系数的最小值/最大值，所以F在两次MI中可能使用不同量化。

Haghighat的FAST-FMI在特征域用3×3局部窗口估计边际和联合分布，计算归一化互信息，并对位置和两源取均值。示意为 `mean_windows((2*I_local(A,F)/(H_local(A)+H_local(F)) + 2*I_local(B,F)/(H_local(B)+H_local(F)))/2)`，这里局部PDF/CDF与联合分布的构造才是算法关键，不能用普通局部联合直方图替代。[原作者代码](https://github.com/mhaghighat/fmi)

同一灰度图作为两源和F的既有合成检查中，当前FMI为2.80428247，标准归一化候选为1。这是算法差别，不能靠统一log底或乘一个系数对齐。

### QAB/F的局部质量

三图均用补零Sobel得到g=sqrt(Sx²+Sy²)，当前方向为atan(Sy/Sx)，Sx为0时置pi/2。当前局部强度比实际为：

\[
G_{AF}=\begin{cases}
g_F/g_A,&g_A>g_F,\\
g_F,&g_A=g_F,\\
g_A/g_F,&g_A<g_F.
\end{cases}
\]

当前方向相似度为 `a=1-|theta_A-theta_F|/(pi/2)`；质量为：

\[
Q_{AF}=\frac{0.9994}{1+e^{-15(G_{AF}-0.5)}}\frac{0.9879}{1+e^{-22(a-0.8)}}.
\]

这与CDDFuse公开代码和SeAFusion目录中的Hong代码一致，包括相等分支和未折叠方向的问题。XP论文式(3)的相等正梯度比应为1，方向按无向边缘折叠，源权重可以取g^L，论文实验L=1.5；不能把“旧公开代码一致”等同“原论文完全一致”。[XP原文](https://www.researchgate.net/profile/C-Xydeas/publication/3381966_Objective_image_fusion_performance_measure/links/5475cccd0cf2778985af11a0/Objective-image-fusion-performance-measure.pdf)

### NAB/F为什么同一聚合形式仍不一致

当前NAB/F使用上面的旧Q和w=g。Kumar参考函数另算反射边界、除8的Sobel，采用折叠方向、另一组sigmoid参数、sqrt(Qg*Qa)，并设 `w=g^1.5 if g>=2 else .001`。这些内容都会改变分值，不能只检查最后一行聚合公式。[SeAFusion中的Kumar函数](https://github.com/Linfeng-Tang/SeAFusion/blob/main/Evaluation/analysis_nabf.m)

SeAFusion wrapper还把输入转为[0,1]再传入该函数，而固定阈值Td=2与尺度有关。因此即使复制单函数，也必须复核调用时的范围。[wrapper](https://github.com/Linfeng-Tang/SeAFusion/blob/main/Evaluation/analysis_Reference.m)

### SSIM与MS-SSIM

当前每个源与F的SSIM为：

\[
\operatorname{mean}_{valid}\frac{(2\mu_X\mu_F+C_1)(2\sigma_{XF}+C_2)}{(\mu_X^2+\mu_F^2+C_1)(\sigma_X^2+\sigma_F^2+C_2)},
\]

其中C1=(.01*255)^2，C2=(.03*255)^2。当前256像素场景不自动预缩小，对照作者ssim_index.m合理；不能说与带自动缩小的ssim.m在所有尺寸上等价。[作者页面](https://ece.uwaterloo.ca/~z70wang/research/ssim/)

当前MS-SSIM每对图像计算：

\[
MS=\left(\prod_{j=1}^{L-1}\max(CS_j,\epsilon)^{w_j}\right)\max(SSIM_L,\epsilon)^{w_L}.
\]

256像素时L=5；原权重为[.0448,.2856,.3001,.2363,.1333]，当前再除总和。每尺度高斯统计使用反射边界并在全图取均值；尺度间用2×2 uniform_filter再隔点抽样。原作者msssim.m的valid池化和降采样对齐不同，不是完全相同实现。[作者MS-SSIM代码](https://ece.uwaterloo.ca/~z70wang/research/iwssim/msssim.zip)

MEF-SSIM是以源序列构造结构参考的融合质量模型，不是两个普通SSIM结果的平均。SeAFusion目录把该模型存为analysis_ms_ssim.m；因此不能根据文件名把其输出当成Wang 2003两图MS-SSIM。

### VIFp

当前每个源到F的像素域多尺度核心可概括为：

\[
VIFp(X,F)=\frac{\sum_{s,p}\log(1+g_{s,p}^{2}\sigma_{X,s,p}^{2}/(\sigma_{v,s,p}^{2}+2))}{\sum_{s,p}\log(1+\sigma_{X,s,p}^{2}/2)}.
\]

采用4尺度高斯统计并处理负方差、负增益。对正常输入，当前两源均值与CDDFuse/SeAFusion两源之和相差2倍；不应称为不同质量提升。零参考信息量时原比值未定义，当前改成1有问题。[LIVE作者说明](https://live.ece.utexas.edu/research/quality/VIF.htm)

## 与CDDFuse逐项对照的可换算部分

以下只针对同一图像、相同预处理、正常输入下，当前抓取的公开代码，不断言能直接换算论文表格：

- MI：当前bit数值=CDDFuse自然对数数值/ln(2)，约乘1.442695。
- VIF：当前均值=CDDFuse双源和/2。
- PSNR：两者差为`20*log10(255/max(F))`，所以不是所有图像共用的固定偏移。
- SF：H=W=256且相同[0,255]输入时，当前SF=CDDFuse SF*sqrt(255/256)，约低0.1955%；非方图要分别处理两个方向。
- SSIM：既有求和/均值差异，又有高斯/默认窗口与协方差差异，不能简单除2。
- AG、FMI、NAB/F、MS-SSIM：不存在可靠的单个固定换算系数。

## 对论文报告的建议

1. 论文明确给出采用的metric profile及源码版本，不仅列缩写。
2. 当前14项不是任何一篇对照论文的原样评价集合，应称为本项目固定评价协议。
3. 比较模型时统一重评输出图，不能把各篇论文表格里的不同单位、聚合、窗口分数拼到一张表中。
4. 修正后的FMI/NAB/F等属于新协议，旧门槛和旧结果保留但不可直接迁移。
5. 原作者MATLAB认证仍待完成，本轮没有生成认证通过文件，也没有继续启动MATLAB或批量评价。
