# OCAF-v1：去噪反事实驱动的光学恒等融合

## 研究问题

现有B0、Detail-only SPD和CSR输出在视觉上均由SAR主导。OCAF-v1不再修补B0输出，而将最终结果定义为光学图像与可归因SAR增量之和：

`Y = Optical + Common SAR increment + Exclusive stable SAR increment`

## 严格结构性质

1. 三个增量尺度初始为零，初始输出逐值等于光学图像。
2. 低频只来自光学；SAR不能修改区域亮度和整体外观。
3. Common专家仅沿光学中频结构方向增强，不替换光学结构。
4. Exclusive专家只增加Clean SAR相对光学的正超额中高频能量。
5. SAR可靠性同时使用SPD Raw/Clean/Removed状态、方向连贯性和多尺度持续性。

## 冻结模块

- B0 Encoder、BaseFuse、DetailFuse、Decoder；
- Detail-only SPD；
- SPD Denoising Head。

训练模块只有OCAF结构路由器和三个有界注入尺度。

## 去噪反事实训练

每个训练批次同时输入原SAR和轻微乘性扰动SAR。两条路径经过同一冻结SPD和OCAF，最终输出应保持一致。该约束用于降低随机斑点对SAR增量的影响，不将SPD删除残差直接视作噪声真值。

## Validation1000输出

五联图顺序：

`Optical / Raw SAR / SPD Clean SAR / Detail-only SPD / OCAF-v1`

证据图顺序：

`SPD Removed / SAR Reliability / Common Prior / Exclusive Prior / Common Gate / SAR Gate`

## 验收条件

- OCAF输出必须明显保留光学区域外观、道路、田界与纹理；
- 光学低频MAE应接近零，不能通过改变低频获得融合指标；
- OCAF相对Optical-only必须提高至少一项SAR结构相关指标，如QAB/F、FMI或SCD；
- Stable SAR gradient gain必须为正，Exclusive Gate不能全零；
- SAR增量应集中于共同结构和稳定独占结构，而不是散斑全图覆盖；
- NAB/F不应明显高于Detail-only SPD；
- 必须同时报告Optical-only、Detail-only SPD和OCAF-v1，不能只选择有利基线。

任何纯光学复制、SAR重新主导或证据图散斑化都不得进入正式test1000。

## 命令

- 训练：`run_ocaf_v1_training_seed42.cmd`
- 验证：`run_ocaf_v1_validation.cmd`
- 正式测试：仅验证通过后执行`run_ocaf_v1_test1000.cmd`
