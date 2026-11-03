# 14项融合指标：论文、作者实现与当前代码审查

日期：2026-11-01。范围为 `test_OS.py --metric-profile paper` 使用的全部14项指标，并说明legacy兼容范围。未训练、未批量验证、未访问test1000，没有修改历史CSV或默认评价路径。

## 审查结论与交付状态

当前paper评价器不能整体宣称与标准作者实现一致。8项在声明聚合方式、输入范围和非退化条件后可保留；FMI、QAB/F、NAB/F、MS-SSIM有实质差异；AG和SF有需要明确选择的离散化约定；VIF及相关性还存在未定义情况处理问题。

已新增隔离的 `utils/reference_metrics.py`，标识为 `reference_v2_candidate`，未接入默认正式推理。它是按下列选定来源实现的候选版本，不宣称所有指标只有唯一权威定义。Python旧/新合计16项测试已通过。原作者MATLAB跨语言数值对照尚未完成：本机MATLAB R2024a两次启动均在执行任何脚本前报错 `failed to load settings errors_warnings plugin`，更换工作区prefdir后仍失败。这是运行环境问题，不能将未执行的认证报告为通过。

## 输入和多源聚合协议

- 单通道、相同尺寸、固定[0,255]整数灰度级；计算前统一float64，避免uint8减法回绕。
- RGB融合仍使用现有OpenCV YCrCb亮度策略；不能在重评时静默换成MATLAB rgb2gray。RGB到亮度本身是评价协议的一部分，不同策略必须分组。
- MI采用两源MI之和，单位bit；CC、SSIM、MS-SSIM、VIFp采用两源结果均值；SCD采用原定义两项之和。
- PSNR使用两源MSE的均值再取对数，峰值255。它不是“两项PSNR的平均”，也不是有真值融合图像的重建PSNR。
- FMI明确为FAST-FMI的DCT特征、3×3局部窗口、归一化版本。
- VIF明确为LIVE的像素域多尺度VIFp，两源均值。它不是波形域原始VIF或Han等人的独立融合VIF指标。
- 候选版固定5尺度MS-SSIM，输入至少176×176；不因图小而自动删尺度。当前QXS 256×256适用。
- 未定义的相关性/VIF、负数CS的实数分数幂记为NaN或拒绝，不返回“完美分数1”，不默默剔除后计算均值。正式报告应另报有效样本数、原因，并阻止未解释的无效值混入排名。

## 全部14项逐项审查

| 指标 | 当前paper实现 | 权威来源／明确的参考约定 | 结论与候选版处理 |
|---|---|---|---|
| EN | 256灰度概率，log2 | Shannon离散熵；MATLAB entropy的uint8约定 | 保留；单位bit，拒绝隐式改变灰度范围 |
| MI | 256×256联合直方图，MI(A,F)+MI(B,F)，log2 | 离散互信息公式；Qu作者署名参考代码 | 保留；不能与legacy的自然对数结果混比 |
| FMI | 全图DCT后，将系数联合量化为256箱，再求未归一化MI之和 | Haghighat原作者FAST-FMI：局部PDF/CDF、联合分布估计、归一化、源均值 | 必须区分；候选版改为FMI_dct/w3，旧值不应当成标准归一化FMI |
| PSNR | 固定255峰值，双源MSE均值后取对数 | MATLAB psnr公式，并明确本项目多源聚合 | 保留；不能照搬参考库的归一化后仍用峰值255错误 |
| QAB/F | 旧Hong代码的等幅分支、未折叠方向差、源权重g^1 | Xydeas–Petrovic 2000式(3)(4)(7)(8)，实验参数L=1.5 | 候选版正比值、折叠方向、显式L=1.5；保留旧实现作兼容 |
| NAB/F | 用旧QAB/F局部质量和g权重拼modified聚合 | Kumar modified NAB/F的独立边缘、质量和权重实现 | 聚合主体正确，完整算法不一致；候选版独立实现，不再复用QAB/F参数 |
| AG | 邻接前向差分，在(H-1)×(W-1)区域取均值 | Tang参考AG_evaluation：MATLAB gradient中心差分/单边边界，全文像素求和再除(H-1)(W-1) | 两种离散版本不同；候选明确选后者，并非宣称前向差分公式数学错误 |
| SF | 横/纵差平方和均除MN | Tang参考SF_evaluation：分别除各自差分数组的元素数 | 归一化约定不同；候选明确选择差分样本数版本，不能混表 |
| SD | sqrt(mean((F-mean(F))²)) | 总体标准差；参考SD_evaluation同式 | 保留ddof=0；不要直接换成默认N-1的std2 |
| CC | 两源Pearson相关均值；零方差自定义0/1 | Pearson相关、corr2 | 非退化保留；零方差未定义，候选返回NaN |
| SCD | corr(A,F-B)+corr(B,F-A)；零方差自定义0/1 | Aslantas与Bendes 2015作者署名实现 | 非退化保留；常量残差未定义，候选不伪造数值 |
| SSIM | sigma1.5高斯、总体协方差、data_range255、源均值 | Wang作者ssim_index.m（不自动预缩小） | 保留单尺度公式；256像素下成立，不能笼统声称等同带自动预缩小的ssim.m |
| MS-SSIM | 全边界均值、可减少尺度、归一化权重、uniform_filter降采样、负CS截成epsilon | Wang作者msssim.zip：valid窗口、固定5尺度、2×2 symmetric低通、原权重乘积 | 候选按作者代码规则实现；不是某库同名的多尺度MEF-SSIM |
| VIF | LIVE像素域核心，两源均值；NaN统一返回1 | LIVE vifp_mscale.m；分母为参考信息量 | 非退化核心保留；未定义值处理必须修正，候选返回NaN |

“8项可保留”指EN、MI、PSNR、SD、CC、SCD、SSIM、VIF的主要非退化公式；CC、SCD、VIF的异常处理仍需要改变。

## 最重要的实证检查

### FMI已经足以确认不是同一算法

固定seed42的256×256随机灰度图，以同图作为两源和融合图：当前FMI=2.80428247；作者归一化FAST-FMI对相同特征窗口应为1，候选结果为1。当前全图DCT直方图MI不是局部归一化FMI。不能通过除以一个常数修复。

### QAB/F不能只复制流行代码

已经取得原作者上传的论文全文。式(3)的相等正梯度比为1，式(4)为折叠后的无向边缘方向相似度，式(8)允许g^L权重；论文实验列L=1.5。流行Hong移植代码的相等分支直接赋gF，方向未折叠，权重为g。这些内容即使在参考仓库中出现，也不能因此替代论文定义。

候选版保留明确的XP参数；不会将QAB/F的sigmoid最大值硬归一化成1。恒等图得分略低于1可以由sigmoid参数解释，不应把它当作必须修正的问题。补零边界在XP兼容路径中仍明确保留，与Kumar NAB/F的镜像边界分开；论文没有规定的边界约定必须通过spec披露。

### NAB/F的问题不来自欧氏幅值

欧氏幅值和“严格超过两源”的掩码均保留。需要改变的是专用质量函数、方向折叠、源权重及边界。参见此前NABF_IMPLEMENTATION_AUDIT_20261101.md；本次对所有指标使用独立版本，不改旧值。

### SSIM已通过独立公式检查

合成图上旧paper SSIM=0.6814181293627254，直接11×11 valid高斯公式=0.681418129362719，差约6e-15；与官方scikit-image推荐的Wang参数一致。没有证据需要更改该256像素单尺度实现。

### VIF不能把无参考信息解释成完美

源图均为全128、融合输出为随机噪声时，旧VIF返回1，因为0/0产生NaN后被改成1。候选返回NaN。数学上参考信息量为零时该比值未定义；既不能直接记1，也不宜私自改0后混入平均分。

### 合成全指标对照（不代表模型性能）

已保存一组固定随机合成输入和14项旧/新输出。EN、MI、PSNR、SD、CC、SCD、SSIM、正常VIF一致或仅有浮点尾差；FMI、QAB/F、NAB/F、AG、SF、MS-SSIM不同。最终源码锁定及MATLAB待认证输入在 `reports/metric_audit_20261101_v2/`，数值表为其下synthetic_comparison.csv。v1是先前导出的合成审查快照。两者均不能用于模型排名或新门槛选取。

## 参考实现本身的陷阱

本轮没有直接调用流行仓库的analysis_Reference整体包装器：

- 它调用analysis_nabf时的参数顺序与函数签名不一致。
- PSNR_evaluation先除255，又继续以255作为峰值，等价于额外偏高20log10(255)，约48.1308 dB。
- MI的Hab若直接对uint8执行+1，MATLAB饱和算术会使255级与254级索引混叠；测试oracle采用double直方图索引。
- analysis_ms_ssim注释和源码实际是多尺度MEF质量算法，不是2003两图MS-SSIM，不能因为文件名相似而替换。
- CC_evaluation中B项减的是A均值，但由于F已去均值，普通非退化情况下额外常数在求和中抵消；不能将这一表面问题误报成所有CC数值错误。

## 来源与锁定

- **XP QAB/F论文**：[原作者上传全文](https://www.researchgate.net/profile/C-Xydeas/publication/3381966_Objective_image_fusion_performance_measure/links/5475cccd0cf2778985af11a0/Objective-image-fusion-performance-measure.pdf)，DOI 10.1049/el:20000267。
- **FMI**：[原作者mhaghighat/fmi](https://github.com/mhaghighat/fmi)，2011原论文DOI 10.1016/j.compeleceng.2011.07.012；FAST-FMI 2014 DOI 10.1109/ICAICT.2014.7036000。已保存原fmi.m与BSD许可证，blob SHA在SOURCES.json。
- **SSIM**：[作者网站](https://ece.uwaterloo.ca/~z70wang/research/ssim/)，原ssim_index.m已保存，含原版权说明。
- **MS-SSIM**：[作者MS-SSIM压缩包](https://ece.uwaterloo.ca/~z70wang/research/iwssim/msssim.zip)，已保存msssim.m、ssim_index_new.m；[作者论文页面](https://www.cns.nyu.edu/~lcv/pubs/makeAbs.php?loc=Wang03b)，DOI 10.1109/ACSSC.2003.1292216。
- **VIFp**：[LIVE作者网站](https://live.ece.utexas.edu/research/quality/VIF.htm)，参考vifp_mscale保留UT Austin版权说明；与2006完整波形域算法的区别已注明。
- **NAB/F**：[Kumar作者发布页](https://www.mathworks.com/matlabcentral/fileexchange/43053-objective-fusion-performance-parameters-modified-fusion-artifacts-measure)，对应2012在线／2013卷期论文DOI 10.1007/s11760-012-0361-x。
- **SCD**：作者Bendes署名analysis_SCD.m，[出版方论文页面](https://www.sciencedirect.com/science/article/pii/S1434841115002691)，DOI 10.1016/j.aeue.2015.09.004，已核对题名、作者和DOI。
- **公式明确的基础统计与作者代码汇集**：[Tang评价实现目录](https://github.com/Linfeng-Tang/Image-Fusion/tree/main/General%20Evaluation%20Metric/Evaluation)。选用AG/SF只是固定可复现版本，不把它们泛称唯一标准。
- **PSNR**：[MathWorks官方定义](https://www.mathworks.com/help/images/ref/psnr.html)；**SSIM参数**：[scikit-image官方说明](https://scikit-image.org/docs/stable/api/skimage.metrics.html)。

引用文件的git blob锁定见 `tests/metric_reference/SOURCES.json`，本地实际文件SHA-256和候选实现SHA-256见合成审查的audit_metadata.json。没有下载完整模型或训练数据。

## 验证与下一步使用要求

已执行：

```cmd
.venv\Scripts\python.exe -m unittest tests.test_reference_metrics tests.test_paper_metrics
```

16项通过，覆盖14项计算、局部FMI独立标量公式、Kumar边缘独立标量对照、SSIM官方参数对照、输入及未定义值行为。它们不是16份原作者MATLAB黄金数值。

修好本机MATLAB启动后，执行已准备的合成对照：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_metric_reference_check.cmd
```

此命令仅检查合成数据，不训练、不加载checkpoint或验证集。MATLAB端会直接调用作者FMI、SSIM、MS-SSIM、VIFp、Kumar及SCD等函数；基础统计与XP修正以显式公式计算，避免已知包装器缺陷。输出matlab_oracle.csv，只有全部14项误差通过才创建MATLAB_PARITY_PASSED.txt。存在oracle输出即拒绝覆盖。

认证通过前不把候选版作为正式论文指标、不替换test_OS.py默认paper、不比较候选版与历史表。通过后仍需预先固定新协议，对所有方法在相同图像上统一重评；批量重评由用户执行，写入新唯一目录、比较组与注册条目。旧NAB/F门槛、FMI涨幅门槛及QAB/F门槛不能自动迁移。

本轮没有产生新模型评价，因此未新增EXPERIMENT_REGISTRY条目。PRL仍暂停，损失改动也暂不实施，优先完成评价口径认证。
