# NAB/F 实现审查：是否需要修正

日期：2026-11-01。本轮为源码、文献与合成单元级检查，不训练、不运行val1000或test1000，不修改生产评价代码。

## 判断

需要澄清评价版本，并准备独立的新实现以对齐指定参考。不能为了与训练一致，把NAB/F的欧氏梯度幅值改成绝对值和。当前函数有合理的modified NAB/F聚合形式，但不是已核对的Kumar参考代码的完整复现，存在方向处理和边界行为问题。修正后必须作为新metric profile保留，不能覆盖历史结果或沿用旧阈值。

代码自述是依据王浩龙论文第4.3.2节、式4-32/33实现的paper版本；工作区未找到该论文PDF，故本轮不能证明它违反该论文所有细节。以下对比对象是明确可追溯的Kumar modified NAB/F代码，不宣称所有文献只存在一个NAB/F版本。

## 参考来源与证据边界

1. Petrovic与Xydeas，ICCV 2005，*Objective Image Fusion Performance Characterisation*，DOI：10.1109/ICCV.2005.175。[作者单位书目](https://research.lancaster-university.uk/en/publications/objective-image-fusion-performance-characterisation/)；该来源提供摘要及书目，未取得原文完整公式。
2. B. K. Shreyamsha Kumar，*Multifocus and Multispectral Image Fusion based on Pixel Significance using Discrete Cosine Harmonic Wavelet Transform*，2012在线、2013卷期，DOI：10.1007/s11760-012-0361-x。[作者主页](https://sites.google.com/view/shreyamsha/research/image-fusion/image-fusion-using-DCHWT)及[作者发布的MATLAB工具](https://www.mathworks.com/matlabcentral/fileexchange/43053-objective-fusion-performance-parameters-modified-fusion-artifacts-measure)明确说明modified NAB/F是对Petrovic版本的修改。作者自存档与代码Drive链接本轮仅返回登录或加载页面，未据此声称已经读取原始包。
3. 图像融合研究者Linfeng Tang维护的参考实现保留Kumar署名、函数注释和日期，本轮读取了完整`analysis_nabf.m`及其两个Sobel依赖文件。它是本轮逐项代码比较的直接来源：
   - [analysis_nabf.m](https://github.com/Linfeng-Tang/Image-Fusion/blob/main/General%20Evaluation%20Metric/Evaluation/analysis_nabf.m)，blob SHA `11b5267db905f630ba3e9eb797ec6ac938a58161`。
   - [sobel_fn.m](https://github.com/Linfeng-Tang/Image-Fusion/blob/main/General%20Evaluation%20Metric/Evaluation/sobel_fn.m)，blob SHA `3ec01336dffa0a5a4bccd979ff4629a1c64da163`。
   - [per_extn_im_fn.m](https://github.com/Linfeng-Tang/Image-Fusion/blob/main/General%20Evaluation%20Metric/Evaluation/per_extn_im_fn.m)，blob SHA `91eb7b567e4db80dc933356b8ee4a83de896979e`。

不采用将NAB/F写成噪声方差比例的网络博客代码，也不将第三方转载代码当作作者原始发布包。

## 逐项核对

| 项目 | 当前PaperEvaluator | 所核对的Kumar参考实现 | 判断 |
|---|---|---|---|
| 梯度幅值 | sqrt(Sx²+Sy²) | sqrt(Sx²+Sy²) | 相同思路，保留欧氏幅值 |
| 伪影掩码 | gF严格大于gA及gB | 相同 | 不应改成大于任一源或改成梯度差均值 |
| 聚合 | 掩码内分别用(1-QAF)wA及(1-QBF)wB，除总权重 | modified版本相同形式 | 聚合主体合理，但Q和w不同会改变值 |
| Sobel尺度 | 未除8 | 系数除8 | 对带固定阈值的参考权重必须同步处理 |
| 边界 | scipy.signal.convolve2d默认补零 | 扩展函数实际为镜像邻行/邻列 | 当前产生图像外框梯度；参考函数虽命名periodic，实际不应移植为wrap |
| 方向相似度 | 1-abs(aA-aF)/(pi/2) | abs(abs(aA-aF)-pi/2)*2/pi | 角度跨越±pi/2时不同；当前值可为负 |
| 强度sigmoid参数 | Tg=.9994、kg=15、中心=.5 | Nrg=.9999、kg=19、中心=.5 | 两套参数不可默认为同一版本 |
| 方向sigmoid参数 | Ta=.9879、ka=22、中心=.8 | Nra=.9995、ka=22、中心=.5 | 明显不同 |
| 局部质量组合 | Qg乘Qa | sqrt(Qg乘Qa) | 当前复用旧QAB/F函数，不是参考NAB/F的Q |
| 源权重 | w=g | g>=2时g^1.5，否则.001 | 弱边缘与强边缘的相对贡献不同 |

参考代码还同时计算Petrovic原版和Kumar修改版；不能把两者的分子混用后统称同一指标。上表比较的是修改版。

## 已执行的合成检查

### A. 等幅值分支存在独立缺陷，但不解释当前NAB/F退化

`Qabf_getQabf`在gA==gF时把比值设为gF，而正的相等梯度比应为1。例如方向相同、两边梯度同为0.5时当前Q为0.48766638；梯度同为1时Q为0.97479362。这是共用QAB/F函数的问题。

但当前NAB/F严格只累计gF>gA且gF>gB的位置，这些位置不可能进入gA==gF或gB==gF分支。合成随机数组的断言也确认掩码排除了这些分支。因此修正该等值分支会影响QAB/F，但不能用它解释NAB/F的主要变化。

### B. 方向跨界时会错误地给予极低保留质量

对aA=+89度、aF=-89度，当前方向差按178度计算；按无向边缘的pi周期，两者应只差2度。当前gA=1、gF=2时Q约5.10e-18。参考代码的折叠表达处理了该周期问题。

实现时不能只把atan换成atan2就宣布修好；必须同时明确pi周期方向差与零梯度方向的约定。

### C. 补零边界会把纯亮度变化计为新增边缘

构造两张全128的源图，融合图为全140，无内部空间结构变化：

- 当前32×32、64×64和256×256三种尺寸，NAB/F均约0.02661463。
- 全128的32×32图被当前Sobel检测出124个非零梯度位置，恰为全部外圈像素。
- 仅将边界改成镜像，其余当前NAB/F参数和公式不变，该平坦亮度变化的NAB/F为0。

这说明默认补零边界确实引入外框效应。该合成例不说明真实图像的NAB/F全部来自边界，也不说明应该忽略真实边缘亮度变化。

### D. 已有测试通过，但不足以证明公式对齐

执行：`.venv\Scripts\python.exe -m unittest tests.test_paper_metrics`，5项通过。已有测试检查有限性、MI、SSIM、PSNR和全零图，未覆盖方向周期、非零平坦图、等梯度比或Kumar参考数值。

## 应如何修正

1. 保留当前评价器及历史CSV，明确命名为旧paper/thesis约定；不静默替换。
2. 新增独立版本，例如`paper_kumar_nabf_v2`，为NAB/F单独实现并固定Sob​​el尺度、镜像边界、方向周期、sigmoid参数、平方根组合、权重阈值和输入[0,255]范围；不要直接修改旧QAB/F辅助函数来间接改变所有指标。
3. 等幅值比属于QAB/F的独立问题，单独版本化修正；不得把其修复收益混作NAB/F修复收益。
4. 先与已固定的参考代码核对合成样例、边界像素和零梯度。要求源图交换对称、取值有限且在预期范围内、平坦图无边界伪影；方向折叠需有明确用例。
5. 再对固定单张真实验证图作旧/新口径对照，定位差异来自方向、权重还是边界。单图只能诊断。
6. 若采用新口径，后续才由用户批量重评所有比较方法，使用新的唯一目录和比较组、更新注册表；优先评价已保存融合图，无需因指标修复重新训练。正式test1000仍遵守已选模型与既定测试协议。
7. 旧NAB/F阈值0.0660仅适用于旧版本。新版本不能沿用旧阈值，也不能根据候选输赢反向挑选评价器或新阈值。

本轮只执行单元级合成检查，没有产生新的正式模型评价结果，不新增实验注册条目。

## 对已有结论的影响

E2与PRL旧结果确实使用同一实现，因此此前数字和旧门槛筛选记录不应删除。但同一旧实现不等于外部标准可比，也不能保证换成参考实现后排名不变。当前结论应限定为“PRL未通过旧协议”；不能将相对NAB/F恶化48.47%直接解释为噪声能量或真实伪影增加48.47%。SSIM和MS-SSIM的下降仍是独立证据，不会因只修NAB/F自动消失。
