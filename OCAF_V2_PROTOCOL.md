# OCAF-v2：独立增量门与显式灰度幅度

## 修改原因

OCAF-v1保持了光学外观，但退化为光学复制。Exclusive Prior均值0.358%，SAR Gate均值0.0235%，Exclusive增量RMS仅7e-6。主要原因是三路Softmax竞争、五项置信度连乘、候选/门控/尺度三重衰减，以及路由与稀疏损失主导总损失。

## OCAF-v2变化

1. 光学画布永久存在，不再参与Softmax。
2. Common和SAR使用两个互不竞争的Sigmoid门。
3. Exclusive结构只将SAR相对光学的新颖性作为必要条件；可靠性、SPD删除、连贯性、跨尺度持续性和扰动稳定性进入logit加减，不再全部相乘。
4. 使用原SAR与固定乘性扰动SAR的两条SPD Clean路径构造稳定SAR平均图和扰动稳定图。
5. 候选增量在门控区域内做RMS归一化，再由具有明确灰度意义的幅度控制。
6. 最大Common幅度为2/255；Exclusive中频为1.5/255；Exclusive高频为1/255。
7. 去除稀疏损失和三路路由交叉熵，门控软监督权重降至0.01。
8. 分别约束Common与Exclusive输出RMS预算。

## 严格不变项

- 输出仍为`Optical + Common delta + Exclusive delta`。
- 初始化输出逐值等于光学图像。
- 低频只来自光学。
- B0、SPD、Denoising Head和原融合网络全部冻结。
- 数据划分与测试数据不变。

## Validation1000输出

五联图：

`Optical / Raw SAR / Stable SPD SAR / Detail-only SPD / OCAF-v2`

七联证据图：

`SPD Removed / SAR Reliability / Perturbation Stability / Common Score / Exclusive Score / Common Gate / SAR Gate`

## 验收条件

- OCAF-v2必须保持可辨的光学区域外观；
- 相对Optical-only，FMI、QAB/F或SCD至少有一项稳定提高，而不是只有AG/SF微小变化；
- Exclusive Score与SAR Gate不能全零；
- Exclusive增量RMS必须达到可测水平，不能重现7e-6塌缩；
- SAR结构增量应集中于连贯目标，而不是全图散斑；
- 光学低频MAE保持接近零；
- 不得仅凭NAB/F或VIF判断成功。

验证失败时不得进入正式test1000。

## 命令

- 训练：`run_ocaf_v2_training_seed42.cmd`
- 验证：`run_ocaf_v2_validation.cmd`
- 正式测试：仅验证通过后执行`run_ocaf_v2_test1000.cmd`
