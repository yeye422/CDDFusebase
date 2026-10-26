# P1：来源保持的稳定SAR Detail注入

## 固定内容

- SPD不参与P1；
- Encoder、BaseFuse、原DetailFuse和Decoder全部冻结；
- Base路径保持原B0；
- 不使用局部Warp、Deformable Convolution、Mamba或FFT分支；
- 只训练轻量Detail Router。

## 两个冻结专家

```text
Optical expert = DetailFuse(Optical Detail)
Joint expert   = DetailFuse(Optical Detail + SAR Detail)
```

Router预测空间×通道Gate：

```text
Fused Detail = Optical expert + Gate * (Joint expert - Optical expert)
```

- `Gate=1`：恢复原B0 Detail路径；
- `Gate=0`：只使用光学Detail专家；
- 初始Gate均值为0.995，近似复现B0；
- 稳定性教师只监督可靠性排序，不直接规定SAR注入比例；
- 独立的可学习抑制上限初始为1%，由融合损失决定实际抑制强度；
- 训练时用两次独立SAR乘性扰动构造稳定性教师；
- 推理时Router只需一次Optical/SAR特征。

## 训练

训练属于大批量任务，由用户在Windows CMD中运行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_fusion_p1_training_seed42.cmd
```

第一轮仅用于验证：Gate是否从接近1形成空间和通道差异，且validation loss稳定下降。正式指标评价脚本在训练接口确认后单独接入，禁止直接使用test1000选择P1结构。

训练完成后的validation1000：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_fusion_p1_validation.cmd
```

输出完整14项逐图指标、均值、标准差、相对B0的逐图改善比例，以及固定8张Optical/SAR/B0/P1横向预览。该阶段不执行正式test1000。
