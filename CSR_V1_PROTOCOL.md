# CSR-v1：SPD引导的竞争式可见结构路由

## 目的

在不修改 B0、Detail-only SPD 与原解码器的前提下，恢复被 SAR 高频覆盖的光学道路、田界和建筑轮廓，同时避免退回到“降低 NAB/F 但同步损失 FMI、QAB/F、VIF”的平滑方向。

## 固定锚点

- 数据集：QXS-SAROPT。
- 划分：`cddfuse_8000_1000_1000_seed20260924.json`。
- B0：`models/qxslab_saropt_from_ivf/CDDFuse_OS_20260924_012025/best.pth`。
- SPD：`models/v21_softmask/spd_detail_only_seed42/best.pth`。
- Encoder、BaseFuse、DetailFuse、SPD、Decoder 全部冻结。
- 新模块从严格等同 Detail-only SPD 输出的状态开始。

## 架构

1. 计算 `R_spd = Raw SAR Detail - Clean SAR Detail`，分别得到删除证据与可靠 SAR 证据。
2. 根据光学与基线输出的局部梯度缺口生成光学缺失图。
3. 路由器输出 Anchor、Optical、SAR 三路 Softmax 概率。
4. Optical 路由同时受到光学缺失图和 SAR 可靠度抑制。
5. 低频完全沿用 Detail-only SPD；只对中频和高频执行有界光学替换。
6. 中频最大替换系数为 0.10，高频为 0.03；实际系数从零开始学习。

## 训练目标

- Optical visibility：在光学缺失区域匹配最终输出与光学梯度。
- SAR retention：在可靠 SAR 区域保持 Detail-only SPD 梯度。
- Unsupported edge：惩罚两输入依据均不支持的新生边缘。
- Anchor：非救援区域保持原输出。
- Low frequency：低频保持原输出。
- Routing：以缺失图和 SPD 可靠性构造三路软监督。
- Negative scale：禁止通过反向替换学习新的平滑捷径。

## Validation1000 预设判断

相对于同一次评估中的 `detail_spd`：

- NAB/F 相对下降至少 5%；
- FMI 下降不超过 0.2%；
- QAB/F 下降不超过 0.1%；
- VIF 下降不超过 0.2%；
- SSIM 不出现系统性下降；
- 保存的光学缺失图应集中于道路、田界和建筑轮廓；
- Optical gate 不应覆盖整幅图或退化为全零；
- 四联图最后两幅必须出现局部可辨的光学结构恢复，而非全局锐化。

任一信息保持指标越界，或视觉变化不可辨，均不得进入正式 test1000。

## 命令

训练：`run_csr_v1_training_seed42.cmd`

验证：`run_csr_v1_validation.cmd`

正式测试仅在验证通过后运行：`run_csr_v1_test1000.cmd`
