# 阶段 0：B1 同损失基线冻结协议

> **执行状态更新（2026-09-30）：** 本文档中的 B0、B1-v1～v5、A1 和 A1′
> 结果继续作为阶段 0–1 的可复现记录；后续 A2/A3 Detail residual 路线已停止。
> 新实验从 C1/C2 无 residual 基线开始，具体以工作区根目录
> `研究实验计划_可靠性驱动光学SAR融合.md` v5.0 为准。

版本：1.0  
日期：2026-09-25

## 0. 已知无效运行

`models/reliability_v4/B1_seed42` 的首轮结果不得作为正式 B1。该运行的
`L_str` 错误地混用了未归一化 Optical Sobel 与归一化 SAR/输出 Sobel，导致
结构目标尺度不可达，并退化为接近 Optical 的平凡解。2026-09-28 已统一为
`Sobelxy / 8`，配置中以 `common_loss_version=v2_normalized_sobel` 标识，并增加
回归测试。修正后的实验必须使用新输出目录，不得覆盖或续训无效 checkpoint。

## 1. 实验目的

B1 只替换 CDDFuse 的 Stage II 融合损失，不修改 Encoder、Decoder、BaseFuse
或 DetailFuse 的结构。它用于区分“共同损失带来的收益”和后续 A1/A2/A3
结构带来的收益。

## 2. 冻结输入

- 数据清单：`../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json`
- 划分内容 SHA-256：`5199869db0d72c04edffe037dee80ecf55dcb49f6c14e429fd8b543ec687d2a7`
- 当前 JSON 文件 SHA-256：`0d540ce16f7fc1a31c16e227ac48100edd6e9a37fd4a01dbf928c46a07954fbd`
- 初始化检查点：`models/qxslab_saropt_from_ivf/CDDFuse_OS_20260924_012025/best.pth`
- Optical：OpenCV BGR 转 YCrCb 后的 Y 通道，除以 255 映射到 `[0,1]`
- SAR：8-bit 灰度，除以 255 映射到 `[0,1]`
- SAR 无效区：像素值等于 0；无效区不参与 `L_str`

加载器会验证三个划分没有重复，并核对划分内容 SHA-256。训练器还会把 JSON
文件和初始化检查点的 SHA-256、参数量及完整参数写入 `config.json`。

## 3. B1 模型与损失

B1 保持原始融合路径：

```text
B_f = BaseFuse(B_o + B_s)
D_f = DetailFuse(D_o + D_s)
F   = Decoder(O, B_f, D_f)
```

共同损失为：

```text
L_common = L_opt + lambda_str * L_str
L_opt    = L1(F, O) + DSSIM(F, O)
```

代码中的 Kornia `SSIM` loss 即 `0.5 * (1 - SSIM)`，对应方案中的 DSSIM 项。

SAR 结构目标按以下固定协议计算：

1. 使用有效像素归一化的 Gaussian blur，`sigma=1.0`；
2. 使用 `Sobelxy / 8` 得到归一化梯度响应；
3. 在每个样本的 SAR 有效区内进行 p99 截断；
4. `G_tar=max(grad(O), S_str)`；
5. 仅在 SAR 有效区计算 `L1(grad(F), G_tar)`。

第一轮固定：

- `lambda_str=1.0`；
- `lambda_decomp=0.0`；
- `phase1_epochs=0`；
- seed 为 42；
- 完整验证集，不限制 validation batch 数；
- 初始化后联合微调原 CDDFuse 四个模块，不增加任何新参数。

## 4. 评估协议

- 正式结果统一使用 `--output-range-policy clamp`，即把网络输出截断到 `[0,1]`；
- `legacy_minmax` 只用于复现旧结果，不得与正式 B0/B1/A1/A2/A3 表格混用；
- 使用论文口径 14 项指标，六项主指标为 NAB/F、QAB/F、VIF、SSIM、
  MS-SSIM、FMI；
- `metrics.csv` 保存逐图指标、均值和标准差；
- `run_metadata.json` 保存 checkpoint/清单哈希、输出范围策略、参数量、
  推理时间、测试图 ID 和预注册典型场景 ID。

现有 `test_result/QXSLAB_SAROPT_1000` 使用逐图 min-max，属于历史结果。
正式比较前必须用 `clamp` 对同一 B0 checkpoint 重新评估，不能直接把旧 CSV
与 B1 比较。

## 5. 阶段闸门

B1 完成后先与重新评估的 B0 比较：

- 如果 B1 已解释预期 NAB/F、SSIM、MS-SSIM 改善的 80% 以上，研究优先收缩为
  损失设计，不进入复杂 Reliability；
- 如果 B1 仍有明显散斑和伪边缘，再进入阶段 1 的 A-only 残差注入；
- 在 B1 结果完成前，不实现或训练 A1/A2/A3。

## 6. CMD 命令

以下命令由实验执行者运行；代码收束阶段不自动启动训练或 1000 对正式测试。

### 6.1 B1 配置检查（不训练）

```bat
cd /d D:\#imageInfusion\CDDFuse

.venv\Scripts\python.exe train_reliability_OS.py ^
  --variant loss_matched_baseline ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --init-checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
  --output-dir tmp ^
  --run-name B1_seed42_dry_run ^
  --epochs 30 ^
  --seed 42 ^
  --amp ^
  --dry-run
```

### 6.2 重新评估正式 B0（1000 对）

```bat
cd /d D:\#imageInfusion\CDDFuse

.venv\Scripts\python.exe test_OS.py ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
  --metric-profile paper ^
  --output-range-policy clamp ^
  --output-dir test_result\reliability_v4\B0_clamp
```

### 6.3 训练 B1

```bat
cd /d D:\#imageInfusion\CDDFuse

.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant loss_matched_baseline ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --init-checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
  --output-dir models\reliability_v4 ^
  --run-name B1_seed42 ^
  --epochs 30 ^
  --phase1-epochs 0 ^
  --lr 1e-5 ^
  --decomp-weight 0.0 ^
  --structure-weight 1.0 ^
  --structure-sigma 1.0 ^
  --structure-percentile 99 ^
  --max-val-batches 0 ^
  --seed 42 ^
  --amp
```

### 6.4 评估 B1（1000 对）

```bat
cd /d D:\#imageInfusion\CDDFuse

.venv\Scripts\python.exe test_OS.py ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --checkpoint models\reliability_v4\B1_seed42\best.pth ^
  --metric-profile paper ^
  --output-range-policy clamp ^
  --output-dir test_result\reliability_v4\B1_seed42
```

## 7. B1-v1 失败结论与 B1-v2 修订

B1-v1 的 1000 对结果确认发生 Optical identity collapse：输出与 Optical Y 的
平均 MAE 约为 `0.41/255`，99.09% 像素相差不超过 2 灰度级。其 NAB/F 下降来自
整体删除 SAR 高频，同时 QAB/F、FMI、AG 和 SF 大幅下降，因此 B1-v1 不通过。

B1-v2 保留原 CDDFuse 架构，但修订损失与训练自由度：

1. Optical 项只约束 `sigma=2` 的低频图像，不再逐像素惩罚 SAR 高频注入；
2. `lambda_opt=0.2`，根据实测梯度范数把 Optical 与结构项的初始参数梯度调至
   同一量级；
3. SAR 梯度保留方向与符号，不再只比较绝对梯度幅值；
4. SAR 候选只取有效区 p85 以上、且比 Optical 梯度至少高 0.02 的位置，候选比例
   上限约 15%；
5. SAR 梯度仍先经过 `sigma=1` 平滑与 p99 截断；
6. 非 SAR 候选位置使用 Optical 的有方向梯度；
7. 默认冻结 Encoder 和 Decoder，只训练 BaseFuse 与 DetailFuse，避免 Decoder
   利用 Optical skip 学习恒等映射；
8. 若验证输出与 Optical 的 MAE 低于 0.01，该轮判定为 identity collapse，不能
   保存为 `best.pth`。

B1-v2 损失为：

```text
L_B1v2 = 0.2 * L_opt_low + 1.0 * L_signed_sparse_structure
```

### 7.1 B1-v2 小规模工程试跑

这是训练命令，由实验执行者运行：

```bat
cd /d D:\#imageInfusion\CDDFuse

.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant loss_matched_baseline_v2 ^
  --training-policy fusion_only ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --init-checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
  --output-dir models\reliability_v4 ^
  --run-name B1_v2_pilot ^
  --train-limit 800 ^
  --val-limit 200 ^
  --epochs 5 ^
  --phase1-epochs 0 ^
  --lr 1e-5 ^
  --optical-weight 0.2 ^
  --optical-lowpass-sigma 2.0 ^
  --structure-weight 1.0 ^
  --structure-sigma 1.0 ^
  --structure-percentile 99 ^
  --sar-edge-percentile 85 ^
  --sar-edge-margin 0.02 ^
  --collapse-mae-threshold 0.01 ^
  --max-val-batches 0 ^
  --seed 42 ^
  --amp
```

试跑必须同时满足：`val_structure` 有下降、`val_output_optical_mae >= 0.01`、
`val_sar_edge_fraction` 约为 0.05–0.15，才允许启动完整 30 轮训练。

## 8. B1-v3：区域平衡与 Optical 容差约束

B1-v2 pilot 虽未完全塌缩，但 SAR 候选只占约 15%，统一全图平均使结构损失仍由
约 85% 的 Optical 背景控制。B1-v3 保留 V1/V2 供复现，采用以下冻结损失：

```text
L_opt_constraint = relu(L1_low - 0.02) + relu(DSSIM_low - 0.03)
L_structure = 0.25 * L_opt_gradient(non-SAR region)
            + 1.00 * L_sar_gradient(SAR candidate region)
L_total = 0.20 * L_opt_constraint + L_structure
```

两个梯度区域分别按各自像素数归一化，SAR 候选不再因面积较小而被背景稀释。
当 Optical 低频误差进入允许范围后，`L_opt_constraint` 自动归零，不再继续推动
输出复制 Optical。

新增监控项：

- `optical_constraint`：容差外的低频 Optical 误差；
- `optical_gradient`：非 SAR 候选区的 Optical 梯度误差；
- `sar_gradient`：SAR 候选区的梯度误差；
- `sar_gradient_retention`：候选区输出对 SAR 梯度幅值的保留率；
- `sar_gradient_alignment`：候选区输出与 SAR 梯度的有符号方向一致性。

### 8.1 B1-v3 小规模试跑

```bat
cd /d D:\#imageInfusion\CDDFuse

.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant loss_matched_baseline_v3 ^
  --training-policy fusion_only ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --init-checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
  --output-dir models\reliability_v4 ^
  --run-name B1_v3_pilot ^
  --train-limit 800 ^
  --val-limit 200 ^
  --epochs 5 ^
  --phase1-epochs 0 ^
  --lr 1e-5 ^
  --optical-weight 0.2 ^
  --optical-lowpass-sigma 2.0 ^
  --optical-l1-tolerance 0.02 ^
  --optical-dssim-tolerance 0.03 ^
  --optical-gradient-weight 0.25 ^
  --structure-weight 1.0 ^
  --structure-sigma 1.0 ^
  --structure-percentile 99 ^
  --sar-edge-percentile 85 ^
  --sar-edge-margin 0.02 ^
  --collapse-mae-threshold 0.01 ^
  --max-val-batches 0 ^
  --seed 42 ^
  --amp
```

不得从 B1-v2 pilot 续训；B1-v3 必须重新从原 B0 checkpoint 初始化。

## 9. B1-v4：计划对齐的最终共同损失候选

B1-v4 删除 V3 的 p85、SAR margin、Optical 容差、区域权重和硬候选门，仅保留：

```text
L_app = L1(Gaussian_sigma2(F), Gaussian_sigma2(O))
      + DSSIM(Gaussian_sigma2(F), Gaussian_sigma2(O))

S_str = clip_p99(abs(normalized_sobel(Gaussian_sigma1(S))))
G_tar = max(abs(normalized_sobel(O)), stop_gradient(R) * S_str)
L_str = Charbonnier(abs(normalized_sobel(F)) - G_tar, epsilon=1e-3)

L_common = L_app + lambda_str * L_str
```

B1/A1 使用 `R=1`；后续 A2/A3 使用 `stop_gradient(R_s)`。Structure Tensor
只作为 ReliabilityNet cue，不进入共同损失。

在原 B0 checkpoint、冻结 Encoder/Decoder、16 个固定 validation batch 上进行一次
梯度量纲校准：

```text
lambda_str = median(||grad L_app|| / ||grad L_str||) = 7.369028
```

均值为 `7.374808`，与中位数一致，因此阶段 0 固定使用 `lambda_str=7.37`，后续
B1/A1/A2/A3 不再重新调节。

### 9.1 B1-v4 小规模试跑

```bat
cd /d D:\#imageInfusion\CDDFuse

.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant loss_matched_baseline_v4 ^
  --training-policy fusion_only ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --init-checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
  --output-dir models\reliability_v4 ^
  --run-name B1_v4_pilot ^
  --train-limit 800 ^
  --val-limit 200 ^
  --epochs 5 ^
  --phase1-epochs 0 ^
  --lr 1e-5 ^
  --decomp-weight 0.0 ^
  --optical-lowpass-sigma 2.0 ^
  --structure-weight 7.37 ^
  --structure-sigma 1.0 ^
  --structure-percentile 99 ^
  --structure-charbonnier-epsilon 0.001 ^
  --collapse-mae-threshold 0.01 ^
  --max-val-batches 0 ^
  --seed 42 ^
  --amp
```

B1-v4 必须从原 B0 初始化，不得从 V1/V2/V3 checkpoint 续训。

## 10. B1-v5：双目标梯度协调与 minimax 选模

B1-v5 完全复用 V4 的 `L_app` 与 `L_str`，不再使用固定 `lambda_str`。每个训练
step 分别计算两个损失对 BaseFuse/DetailFuse 的梯度，将其归一化；余弦为负时
使用 PCGrad 去除冲突分量，再平均更新。该优化规则对后续 B1/A1/A2/A3 统一使用，
不作为论文创新点。

训练前在同一 validation split 记录 B0 参考值：

```text
L_app_0, L_str_0
```

每轮验证计算：

```text
r_app = L_app / L_app_0
r_str = L_str / L_str_0
selection_score = max(r_app, r_str)
```

`best.pth` 按最小 `selection_score` 选择，不再按两个损失的加权和选择。Pilot 每轮
均保存 checkpoint，便于检查 Pareto 折中。

### 10.1 B1-v5 小规模试跑

```bat
cd /d D:\#imageInfusion\CDDFuse

.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant loss_matched_baseline_v5 ^
  --training-policy fusion_only ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --init-checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
  --output-dir models\reliability_v4 ^
  --run-name B1_v5_pilot ^
  --train-limit 800 ^
  --val-limit 200 ^
  --epochs 5 ^
  --phase1-epochs 0 ^
  --lr 1e-5 ^
  --decomp-weight 0.0 ^
  --optical-lowpass-sigma 2.0 ^
  --structure-sigma 1.0 ^
  --structure-percentile 99 ^
  --structure-charbonnier-epsilon 0.001 ^
  --collapse-mae-threshold 0.01 ^
  --max-val-batches 0 ^
  --save-every 1 ^
  --seed 42
```

V5 的 PCGrad 路径固定使用 FP32，因此命令不加 `--amp`。必须从原 B0 初始化，
不得从 V4 或更早版本续训。

## 11. 阶段 1：A1 内容门控残差注入

A1 使用 `demand_only` 变体，保持 Encoder、Decoder 和 BaseFuse 结构不变，将原始
`DetailFuse(D_o+D_s)` 替换为：

```text
D_o_bar = D_o + T_o(D_o)
D_s_bar = D_s + T_s(D_s)
delta_s = tanh(T_delta([D_o_bar, D_s_bar]))
A_s = sigmoid(H_A([D_o_bar, D_s_bar]))
D_f = D_o_bar + 0.5 * A_s * delta_s
```

`T_o/T_s` 与 `T_delta` 末层零初始化；`H_A` 初始输出 0.5，因此初始
`D_f=D_o`。A1 不包含 ReliabilityNet、局部退化或 Reliability loss。

训练协议与 B1-v5 一致：冻结 Encoder/Decoder，训练 BaseFuse 与 A1 Detail 注入头；
继续使用 V5 两目标 normalized PCGrad。Pilot 复用 B1-v5 pilot 的同一 validation
参考值，确保 minimax 分数可直接比较。

### 11.1 A1 pilot

```bat
cd /d D:\#imageInfusion\CDDFuse

.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant demand_only ^
  --training-policy fusion_only ^
  --selection-reference-config models\reliability_v4\B1_v5_pilot\config.json ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --init-checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
  --output-dir models\reliability_v4 ^
  --run-name A1_pilot ^
  --train-limit 800 ^
  --val-limit 200 ^
  --epochs 10 ^
  --phase1-epochs 0 ^
  --lr 1e-5 ^
  --decomp-weight 0.0 ^
  --optical-lowpass-sigma 2.0 ^
  --structure-sigma 1.0 ^
  --structure-percentile 99 ^
  --structure-charbonnier-epsilon 0.001 ^
  --collapse-mae-threshold 0.01 ^
  --max-val-batches 0 ^
  --save-every 1 ^
  --seed 42
```

不得添加 `--amp`，不得从 B1-v5 checkpoint 续训。通过条件：minimax 优于 B1-v5
pilot 的 `0.4049`；`demand_std`、`delta_norm` 和 `injection_norm` 明显非零；门控
不出现大面积全 0/全 1；Output–Optical MAE 不低于 0.01；验证定性图未恢复 B0
级别的密集散斑。

### 11.2 A1 validation 指标与固定门对照

评估器支持 `--split val`、A1 checkpoint 自动识别、门控统计/热力图，以及无训练的
`--force-demand-value 0.5` 对照。三条命令必须使用相同的前200个 validation ID。

```bat
cd /d D:\#imageInfusion\CDDFuse

.venv\Scripts\python.exe test_OS.py ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --split val --limit 200 ^
  --checkpoint models\reliability_v4\B1_v5_pilot\best.pth ^
  --variant auto --metric-profile paper --output-range-policy clamp ^
  --output-dir test_result\reliability_v4\validation\B1_v5_pilot_200

.venv\Scripts\python.exe test_OS.py ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --split val --limit 200 ^
  --checkpoint models\reliability_v4\A1_pilot\best.pth ^
  --variant auto --metric-profile paper --output-range-policy clamp ^
  --save-gate-maps ^
  --output-dir test_result\reliability_v4\validation\A1_pilot_learned_200

.venv\Scripts\python.exe test_OS.py ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --split val --limit 200 ^
  --checkpoint models\reliability_v4\A1_pilot\best.pth ^
  --variant auto --force-demand-value 0.5 ^
  --metric-profile paper --output-range-policy clamp ^
  --output-dir test_result\reliability_v4\validation\A1_pilot_fixed05_200
```

## 12. A1′：Residual-only Detail Injection

旧 A1 的 learned DemandGate 在200对 validation 上系统性弱于固定 `A=0.5`，因此
保留 `demand_only` 供失败消融，主线改为 `residual_only`：

```text
D_o_bar = D_o + T_o(D_o)
D_s_bar = D_s + T_s(D_s)
delta_s = tanh(T_delta([D_o_bar, D_s_bar]))
D_f = D_o_bar + 0.25 * delta_s
```

该变体完全不实例化 DemandGate。Detail 注入头参数为32,256，总参数1,202,352，
可训练参数395,272。其余训练、PCGrad、minimax和参考协议与 B1-v5 相同。

### 12.1 A1′ pilot

```bat
cd /d D:\#imageInfusion\CDDFuse

.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant residual_only ^
  --training-policy fusion_only ^
  --selection-reference-config models\reliability_v4\B1_v5_pilot\config.json ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --init-checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
  --output-dir models\reliability_v4 ^
  --run-name A1prime_pilot ^
  --train-limit 800 ^
  --val-limit 200 ^
  --epochs 10 ^
  --phase1-epochs 0 ^
  --lr 1e-5 ^
  --decomp-weight 0.0 ^
  --optical-lowpass-sigma 2.0 ^
  --structure-sigma 1.0 ^
  --structure-percentile 99 ^
  --structure-charbonnier-epsilon 0.001 ^
  --collapse-mae-threshold 0.01 ^
  --max-val-batches 0 ^
  --save-every 1 ^
  --seed 42
```

命令不加 `--amp`，必须从原 B0 初始化，不得从旧 A1 续训。

### 12.2 A1′ validation 评估

```bat
.venv\Scripts\python.exe test_OS.py ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --split val --limit 200 ^
  --checkpoint models\reliability_v4\A1prime_pilot\best.pth ^
  --variant auto --metric-profile paper --output-range-policy clamp ^
  --output-dir test_result\reliability_v4\validation\A1prime_pilot_200
```
