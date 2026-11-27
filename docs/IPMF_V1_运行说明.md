# IPMF‑V1 冻结实验

IPMF‑V1 从完整的 CF1 第 4 轮 checkpoint 初始化。冻结共享 Encoder、LWPD‑E2、原 Base Fusion、原 Detail Fusion、Restormer Decoder 和 CF1 重建头。新增分支保留全分辨率流，并用 PixelUnshuffle 构造半分辨率双源流，执行一次双向交换。零初始化的 Base、Detail 修正头使初始成图与同精度的 CF1 一致。Decoder 和 CF1 重建头在评估模式下接收更新后的特征，融合损失沿这条路径回传至新增模块。

## P0 和 P1 工程检查

在 Windows CMD 中执行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
set "IPMF_PY=D:\anaconda\envs\lightglue_env\python.exe"
"%IPMF_PY%" -u audit_ipmf_preflight.py --device cuda
```

频带检查使用固定 val 划分的前 8 张图。报告记录 Detail 原始、池化与重排的 RMS、元素数和逆重排误差，并分别测量已有 ASCD‑V1 半尺度输入和新 IPMF 的 `512→32` 投影在 Detail 置零前后的变化。IPMF 的光学和 SAR 响应都须高于浮点噪声下限。预检同时核对 CF0 checkpoint 的 SHA‑256 与 CF1 训练配置，并逐项比对 CF1 继承的 CF0 权重和缓冲区。

掩码检查覆盖 256 张训练图和 8 张验证图，记录 SAR 零值比例、接触零值的有效 Sobel 中心，以及这些中心由 SAR 梯度主导的数量。数据加载器将零值标为无效，当前输入清单和加载器没有给出逐像素的原始 NoData 标注，零值的遥感含义需结合数据来源确认。若这些位置明显影响目标，应在独立协议下评估边界修正，并为 CF1 使用相同损失。可接受当前掩码口径时再开始训练，沿用原有 `L_I + 10 L_G`。

精度检查对 264 张图逐张验证 IPMF 与 CF1 的初始化输出等价。FP32、FP16 和设备支持时的 BF16 使用相同初始权重及相同的 256 张增强训练图，执行 32 次优化器更新尝试。前 8 次和后 24 次分别统计跳步，后段有效更新率至少 99%。各子模块须收到有限的非零梯度并发生更新。FP32 基准通过后按 FP16、BF16、FP32 的顺序选择首个合格精度。查看 `test_result\ipmf_preflight_v2\report.json` 中的 `inputs`、`mask_summary`、`mask_scenes`、`precision` 与 `selected_precision`。

## P2 冻结训练和第 4 轮筛选

```cmd
set /p "IPMF_MASK_SHA=Enter the reviewed report SHA-256: "
"%IPMF_PY%" -u run_ipmf_frozen.py train --epoch 4 --device cuda --reviewed-mask-sha256 "%IPMF_MASK_SHA%"
"%IPMF_PY%" -u run_ipmf_frozen.py val --epoch 4 --device cuda
```

人工确认掩码可用于本轮后，将预检输出的 SHA‑256 粘贴到 CMD 提示处。训练配置记录已确认的报告，后续续训可沿用。训练使用固定 train8000、split seed 20260924、训练 seed 42、256×256、batch 2、累积 4、AdamW `1e-4`、weight decay `1e-4`、梯度裁剪 1.0，以及 CF1 原有数据增强。

FP16 第 1 轮的前 8 次更新计入初始化调整期，其余更新逐轮要求有效率至少 99%。低于门槛会停止并保存失败诊断。梯度缺失会列出模块及参数名称。每个合格轮次保存 checkpoint，重跑可恢复最近完整轮次的模型、优化器和 scaler。日志记录两种修正的 RMS 比、逐模块梯度范数、分段有效更新和跳步、三路 CF1 权重、裁剪前越界比例以及冻结权重一致性。

val1000 使用 `reference_v2_candidate_with_paper_fmi` 和 `clamp_then_uint8`。主筛选固定采用 CF0、CF1 第 4 轮和 IPMF。存在同口径的 CF1 第 8 轮归档时，另写入 `supplemental_cf1_e08`。训练和评测完成后自动更新实验登记与索引。第 4 轮结果位于 `test_result\e2_ipmf_v1_frozen\val1000\epoch_004`。查看 `paired_comparison.csv` 的 14 项指标和共同有效样本数、`screening.json` 的数值筛选结果，以及 `IPMF-V1\preview` 的三联图。

判为通过还需要人工检查平滑退化、散斑回灌、重影和异常对比度。数值筛选要求至少一项结构指标相对 CF1 提升 0.003，另一项达到 CF0 减 0.0005 的下限，并通过 MI、FMI、QAB/F、VIF 和 NAB/F 门槛。MS‑SSIM 要覆盖 CF0、CF1 的全部共同有效场景，当前参照为 766 张。其他核心指标须覆盖 1,000 张。报告单独列出每项指标新增及丢失的有效场景。CC、PSNR、SCD 作为观察项。候选筛选使用 val1000。

## 免训练趋势诊断

```cmd
"%IPMF_PY%" -u run_ipmf_val200_trend.py --device cuda
```

从锁定验证清单按原顺序取前 200 张图，比较 CF1 与 IPMF 第 1、2、4 轮。复用匹配的 CF1 和 IPMF val1000 逐图档案，仅对缺少档案的轮次推理。结果写入 `test_result\e2_ipmf_v1_frozen\val200_trend\first200_epochs_001_002_004`。

`trend.csv` 给出 14 项指标的共同有效样本均值，`paired_comparison.csv` 给出各轮相对 CF1 的差值和相邻检查点的配对差值及区间，`coverage.json` 记录各轮在 CF1 有效场景中的缺失。结合 `training_diagnostics.csv` 的 Base、Detail 修正 RMS 比、损失及三路权重判断趋势。正式续训门槛仍使用 val1000。

## 条件续训至第 8 轮

仅当第 4 轮通过信息保留与 NAB/F 门槛、一项结构指标较 CF1 至少提升 0.0015 且另一项未下降时，再补第 2 轮 val1000 来检查趋势。续训入口会核验两轮的来源和共同有效样本上的结构改善趋势。

```cmd
"%IPMF_PY%" -u run_ipmf_frozen.py val --epoch 2 --device cuda
"%IPMF_PY%" -u run_ipmf_frozen.py train --epoch 8 --device cuda
"%IPMF_PY%" -u run_ipmf_frozen.py val --epoch 8 --device cuda
```

第 8 轮验证写入单独的 `epoch_008` 目录。只有冻结实验出现综合收益后，再决定是否开展联合训练。
