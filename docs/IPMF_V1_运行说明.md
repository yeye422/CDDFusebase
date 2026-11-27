# IPMF‑V1 冻结实验

IPMF‑V1 从完整的 CF1 第 4 轮 checkpoint 初始化。冻结共享 Encoder、LWPD‑E2、原 Base Fusion、原 Detail Fusion、Restormer Decoder 和 CF1 重建头。新增分支保留全分辨率流，并用 PixelUnshuffle 构造半分辨率双源流，执行一次双向交换。零初始化的 Base、Detail 修正头使初始成图与同精度的 CF1 一致。Decoder 和 CF1 重建头在评估模式下接收更新后的特征，融合损失沿这条路径回传至新增模块。

## P0 和 P1 工程检查

在 Windows CMD 中执行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
set "IPMF_PY=D:\anaconda\envs\lightglue_env\python.exe"
"%IPMF_PY%" -u audit_ipmf_preflight.py --device cuda
```

检查读取固定 val 划分的前 8 张图。报告记录 Detail 原始、池化与重排的 RMS、元素数和逆重排误差，同时使用已有 ASCD‑V1 第 4 轮权重测量两路半尺度输入在 Detail 置零前后的变化。

掩码检查记录 SAR 零值比例、接触零值的有效 Sobel 中心，以及这些中心由 SAR 梯度主导的数量。数据加载器将零值标为无效，当前输入清单和加载器没有给出逐像素的原始 NoData 标注，零值的遥感含义需结合数据来源确认。若这些位置明显影响目标，应在独立协议下评估边界修正，并为 CF1 使用相同损失。本轮训练沿用原有 `L_I + 10 L_G`。

精度检查用相同初始权重和微批顺序运行 4 次更新尝试，记录裁剪前后梯度范数、逐模块梯度、参数更新量和 scaler 跳步。FP32 用于建立工程基准。FP16 至少需要 3 次有效更新、最后一次更新成功、各子模块获得有限的非零梯度并发生更新。FP16 未通过时检查 BF16，最终可选 FP32。训练绑定报告选出的精度和报告的 SHA‑256。查看 `test_result\ipmf_preflight_v1\report.json` 中的 `inputs`、`mask_summary`、`precision` 与 `selected_precision`。

## P2 冻结训练和第 4 轮筛选

```cmd
"%IPMF_PY%" -u run_ipmf_frozen.py train --epoch 4 --device cuda
"%IPMF_PY%" -u run_ipmf_frozen.py val --epoch 4 --device cuda
```

训练使用固定 train8000、split seed 20260924、训练 seed 42、256×256、batch 2、累积 4、AdamW `1e-4`、weight decay `1e-4`、梯度裁剪 1.0，以及 CF1 原有数据增强。每轮保存 checkpoint。重跑同一命令从最近完整轮次恢复模型、优化器和 scaler。日志记录两种修正的 RMS 比、逐模块梯度范数、有效更新数、三路 CF1 权重、裁剪前越界比例以及冻结权重一致性。

val1000 使用 `reference_v2_candidate_with_paper_fmi` 和 `clamp_then_uint8`。同口径的 CF0、CF1 第 4 轮逐图结果优先复用，有匹配归档时纳入 CF1 第 8 轮。训练和评测完成后自动更新实验登记与索引。第 4 轮结果位于 `test_result\e2_ipmf_v1_frozen\val1000\epoch_004`。查看 `paired_comparison.csv` 的 14 项指标和共同有效样本数、`screening.json` 的数值筛选结果，以及 `IPMF-V1\preview` 的三联图。

判为通过还需要人工检查平滑退化、散斑回灌、重影和异常对比度。数值筛选要求至少一项结构指标相对 CF1 提升 0.003，另一项达到 CF0 减 0.0005 的下限，并通过 MI、FMI、QAB/F、VIF 和 NAB/F 门槛。有效样本集变化时，用共同有效样本重算配对基准。CC、PSNR、SCD 作为观察项。候选筛选使用 val1000。

## 条件续训至第 8 轮

仅当第 4 轮通过信息保留与 NAB/F 门槛、一项结构指标较 CF1 至少提升 0.0015 且另一项未下降时，再补第 2 轮 val1000 来检查趋势。续训入口会核验两轮的来源和共同有效样本上的结构改善趋势。

```cmd
"%IPMF_PY%" -u run_ipmf_frozen.py val --epoch 2 --device cuda
"%IPMF_PY%" -u run_ipmf_frozen.py train --epoch 8 --device cuda
"%IPMF_PY%" -u run_ipmf_frozen.py val --epoch 8 --device cuda
```

第 8 轮验证写入单独的 `epoch_008` 目录。只有冻结实验出现综合收益后，再决定是否开展联合训练。
