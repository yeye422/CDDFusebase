# DCFD-V1 双源感知双路径协同融合解码器

DCFD-V1 从 CF1 第 4 轮初始化。共享 Restormer Encoder、LWPD-E2、光学 Base 锚定融合、多感受野 Detail Fusion、解码器输入投影及前两层 Transformer 在首轮冻结。原第 3、4 层分别深拷贝到结构流和细节流，基础辐射输出头和 CF1 原位增益头参与微调。DSBI 从双源 Base 和两源差值注入结构信息，DSDI 在原分辨率注入双源 Detail。BPC 在两条路径各自的第 3 层输出之间同步交互。四个新增残差投影的权重和偏置均为零初始化。

成图沿用 CF1 的 5×5 二项式平滑、replicate 边界及原位细节加权，最终对基础与细节之和裁剪到 `[0, 1]`。教师从同一份 CF1 权重独立复制未训练的后两层和成图头，复用冻结前端产生的共享上下文与融合 Detail。每轮 checkpoint 包含学生参数、AdamW 状态、scaler 状态和历史。推理只构建学生。

## 固定训练口径

首轮锁定 QXS-SAROPT 的 `train8000 / val1000 / test1000`，切分种子 `20260924`，训练种子 `42`。单轮 4,000 个 batch，batch size 2，每 4 个 batch 更新一次，四轮共 4,000 次优化器更新尝试。AdamW 的继承层及成图头学习率为 `2.5e-5`，新增 DSBI、DSDI、BPC 为 `1e-4`，weight decay 为 `1e-4`，全局梯度裁剪阈值为 `1.0`。

损失使用 FP32，权重固定为 `1.0 / 0.5 / 2.0 / 0.15`，依次对应二倍和四倍低频结构、全分辨率结构、原位细节及有符号梯度保持、基础辐射软锚定。SSIM 使用 11×11 高斯窗、标准差 1.5 和 replicate 边界。只有窗口依赖的 SAR 像素全部满足 `SAR > 0` 时，才纳入 SAR SSIM。低频路径依次传播 5×5 平滑支持、非重叠平均池化支持和 SSIM 窗口支持。单图单尺度没有有效 SAR 窗口时，使用完整光学 SSIM 损失。正式指标继续使用仓库既定评价器。

## Windows CMD

当前 CUDA 解释器为 `D:\anaconda\envs\lightglue_env\python.exe`。依次运行以下阶段：

```cmd
cd /d "D:\#imageInfusion\CDDFuse"
set "DCFD_PYTHON=D:\anaconda\envs\lightglue_env\python.exe"
"%DCFD_PYTHON%" -u audit_dcfd_preflight.py --device cuda
"%DCFD_PYTHON%" -u run_dcfd_frozen.py train --device cuda --seed 42 --epochs 4
"%DCFD_PYTHON%" -u run_dcfd_frozen.py trend --device cuda --epochs 1 2 4 --offset 200 --limit 200
"%DCFD_PYTHON%" -u run_dcfd_frozen.py validate --device cuda --epoch 4 --limit 1000
```

单景工程检查可在预检前运行 `"%DCFD_PYTHON%" -u audit_dcfd_preflight.py --device cuda --smoke`。正式 P1 使用早于首轮训练的临时实例，按 FP16、硬件支持的 BF16、FP32 的顺序执行梯度累积验收，首个合格精度写入 `models/e2_dcfd_v1_frozen/seed42/preflight.json`。每种尝试包含 128 个微批次和 32 次优化器更新尝试，99% 的门槛在该样本数下要求 32 次全部有效。报告记录 CF1 FP32 等价性、分支及教师存储独立性、三类 SAR 支持、首个非零梯度与首个权重更新、冻结参数一致性、耗时及 CUDA 峰值显存。

训练从同一 CF1 权重重新初始化，锁定预检选定的精度。相同命令可从最新完整轮次恢复。每轮生成 `epoch_001.pth` 至 `epoch_004.pth`、`last.pth` 和 `history.csv`，并记录各项损失、教师偏差、路径分化、三路细节权重、裁剪前越界率、更新与跳过计数、梯度范数、MiB 和秒数。

## 趋势和综合验证

| 阶段 | 固定比较组 | 输出目录 |
| --- | --- | --- |
| P3 · val200 | 第 201–400 景，CF1-E04、DCFD-E01/E02/E04 | `test_result/e2_dcfd_v1_frozen/val200_trend/offset200` |
| P4 · val1000 | 锁定验证集，CF0、CF1-E04、DCFD-E04 | `test_result/e2_dcfd_v1_frozen/val1000/epoch_004` |

两阶段均使用 FP32 推理、`reference_v2_candidate_with_paper_fmi` 和 `clamp_then_uint8`。权重、验证清单、评价器及输出口径匹配时，CF0/CF1 复用归档的逐图指标。各组的 `metrics.csv` 保留 14 项逐景分数和共同有限样本均值、标准差。`paired_comparison.csv` 给出共同样本数及 95% 配对 bootstrap 区间。`coverage.json` 保留固定 CF1 有效场景列表、各臂损失和新增的有效场景、固定样本均值差及置信区间。DCFD 缺失任一 CF1 有效场景时，MS-SSIM 项无法用于达标判定。

P4 另外保存 `comparison_14_metrics.md` 与 `efficiency.json`，固定取 val1000 中索引为 0、125、250、375、500、625、750、875 的八景。三组对这八景保存灰度、RGB 和三联图，效率比较在同一设备上测量八景的 FP32 前向平均耗时、峰值显存和参数量。训练及评测完成后登记实验并刷新索引。

`screening.json` 根据同批 CF1 检查弱项及保真指标。SSIM、MS-SSIM、CC、SCD、PSNR 的最小增量依次为 `+0.003 / +0.003 / +0.020 / +0.050 / +0.30 dB`，至少三项达标且包含一个 SSIM 指标。MI、FMI、QAB/F、VIF 的最大回撤依次为 `5% / 3% / 0.005 / 5%`。NAB/F 采用优先持平或降低的条件，单独报告。继续完整训练的决定结合以上数值、配对区间、MS-SSIM 覆盖与固定场景图像。可使用下列命令交互检查数值门槛：

```cmd
"%DCFD_PYTHON%" run_dcfd_frozen.py screen
```
