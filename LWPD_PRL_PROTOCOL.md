# LWPD E2-PRL 完美重建可学习滤波器验证协议

## 研究问题

在保持 E2 方向上下文门控、融合器、解码器和训练预算不变时，将固定 Haar 分解替换为 Haar 初始化、严格可逆且小幅受限的 lifting 分解，能否在不损害 QAB/F 与 NAB/F 的前提下改善 MI、FMI 或 VIF？

## 唯一架构变化

- 新变体：`lwpd_e2_prl`。
- 固定 `HaarDWT2D` 替换为 `LearnableLiftingDWT2D`。
- 水平和垂直轴分别具有共享的 Predict/Update 三抽头滤波器。
- 光学、SAR和全部特征通道共享同一套滤波器。
- Predict核和Update核分别保持固定和为1与0.5，高频保持零直流。
- 可学习残差由 `0.1 * tanh(delta)` 限幅，初始 `delta=0`，初始输出严格等于 E2 Haar。
- lifting逆变换由相同Predict/Update算子解析构造，完美重建不依赖训练损失。

不加入：

- SAR高频恢复；
- PS或线结构证据；
- 额外Detail旁路；
- Common/Private路由；
- 多尺度或SWT；
- 新融合器或新Decoder。

## 工程闸门

单元测试必须确认：

1. Haar初始化时四个子带与原 `HaarDWT2D` 逐值一致；
2. 偶数和奇数尺寸均能完美重建；
3. lifting参数扰动后仍能完美重建；
4. E2-PRL初始输出与E2一致；
5. lifting参数获得有限非零梯度；
6. 新变体可通过完整CDDFuse前向和自动checkpoint识别。

运行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
.venv\Scripts\python.exe tests\test_lwpd.py
```

## 训练协议

使用与E1–E4相同的QXS-SAROPT协议：

- 固定train8000/val1000/test1000清单；
- seed 42；
- 2 warmup + 4 local + 10 joint，共16 epoch；
- batch size 2，梯度累积4；
- 初始化为相同R3 checkpoint；
- 门控、融合器、Decoder和编码器学习率与E2一致；
- lifting学习率为门控学习率的0.1倍。

训练和val1000评价由用户运行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_lwpd_prl.cmd train-val
```

该命令不访问test1000，并自动生成：

- `metrics.csv`；
- 与E2逐图配对的 `paired_comparison.csv`；
- `comparison_vs_e2/SUMMARY.md`。

## 单种子筛选门槛

相对E2：

- MI提高至少0.5%，或FMI提高至少0.3%；
- QAB/F不低于0.7320；
- NAB/F不高于0.0660；
- VIF不下降；
- SSIM不下降；
- MS-SSIM不下降。

单种子通过只允许进入多种子复核，不直接证明方法有效。只有验证集选择完成后，才能执行一次正式test1000：

```cmd
run_lwpd_prl.cmd test
```

正式test1000结果产生后，必须先更新 `EXPERIMENT_REGISTRY.json`，再运行实验索引维护脚本。

## 停止条件

出现以下任一情况即停止该方向：

1. 完美重建或E2初始等价测试失败；
2. NAB/F超过0.0660；
3. QAB/F低于0.7320；
4. 只提高AG/SF而MI、FMI、VIF均无改善；
5. 滤波器直流约束、方向响应或子带能量出现明显退化。

