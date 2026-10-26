# P2-A：零初始化来源保持Detail残差融合

## 目标

停止P1专家插值与稳定性Gate路线。在严格冻结B0主干的前提下，直接在B0融合Detail坐标中学习小幅、有界的来源保持残差。

## 架构

```text
B0 Detail = frozen DetailFuse(Optical Detail + SAR Detail)

Evidence = [norm(Optical), norm(SAR), abs difference, product]
Correction = bounded zero-initialized Adapter(Evidence)
P2-A Detail = B0 Detail + Correction
```

修正上限为B0融合Detail逐通道RMS的3%。初始化严格复现B0。

## 非对称伪感知

- Optical Head重建完整Optical Detail；
- SAR Head只在双扰动稳定度高的区域重建两次SAR Detail均值；
- Head先独立校准1 epoch；
- 随后冻结Head，只训练Adapter，防止Head吸收全部伪感知监督。

## 固定模块

- Encoder冻结；
- BaseFuse冻结并保持B0路径；
- DetailFuse冻结；
- Decoder冻结；
- SPD不使用；
- 只训练伪感知Head和来源保持Adapter。

## 执行

训练和validation1000均由用户在Windows CMD中运行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_fusion_p2a_training_seed42.cmd
run_fusion_p2a_validation.cmd
```

未通过validation闸门时禁止进入正式test1000。

当前全强度P2-A形成了稳定但不满足非劣约束的折中，因此先冻结checkpoint执行validation残差强度扫描：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_fusion_p2a_scale_sweep_val1000.cmd
```

扫描仅使用validation：`alpha={0,0.25,0.5,0.75,1.0}`。若不存在同时满足NAB/F改善与FMI/QAB/F/VIF非劣限制的强度，则停止P2-A，不进入P2-B或test1000。
