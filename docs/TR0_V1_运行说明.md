# TR0 V1 运行说明

所有实验使用固定 train1024、种子 42 和光学 Base 锚定。C00 与 C01 使用原 E2 的 Detail Fusion 和 Decoder。C10 与 C11 使用 PJD。C00 与 C10 在第 5–8 轮重建重复视图，C01 与 C11 合成互补视图。

PJD-Early 将第二个 SCS 单元移到第一个 Restormer Block 之前。EARLY00 对照 C10，EARLY01 对照 C11。R8 延长来源重建至 8 轮，再进行 12 轮真实融合。B0 复用原损失 E2 的 train1024、4+12 checkpoint 及匹配当前量化与指标协议的 val64 逐图结果。确认集上的 B0 使用同一 checkpoint 评测。

训练默认使用 CUDA BF16 和激活检查点，并记录每轮的实际优化步数、梯度范数、光学及 SAR 重建误差、退化统计、耗时和显存峰值。历史 B0 使用原运行的 FP16 配置。每组从头训练，除 B0 外统一使用第 20 轮权重。代理诊断使用第 8 轮权重。

## 主实验

在 Windows CMD 中执行下列命令。`lock` 将历史 val64 用于筛选，从原验证集中排除这 64 景后等距选出 val128 用于确认。确认集来自已用于历史研究的验证集。

```cmd
cd /d D:\#imageInfusion\CDDFuse
set "TR0_PY=D:\anaconda\envs\lightglue_env\python.exe"
"%TR0_PY%" run_tr0_factorial.py lock
"%TR0_PY%" -u run_tr0_factorial.py train --device cuda --resume
"%TR0_PY%" -u run_tr0_factorial.py screen --device cuda
"%TR0_PY%" -u run_tr0_factorial.py proxy-check --device cuda
```

`--resume` 接续最近一个完整的 epoch，数据顺序、配对增强及代理视图都按 epoch 重新设种子。已完成的组直接复用。在修改配置、训练实现或确认划分后，用新的 `--run-id` 运行。

## 确认与机制对照

T1 第一阶段已完成。使用已锁定的 val128 和 B0、C00、C01 的现有权重，分别固定第 16、20、20 轮。已有 B0 val936 档案覆盖确认集，脚本核对 checkpoint、模型与指标实现哈希、数据目录和场景覆盖后，提取对应的 128 景逐图结果。

```cmd
"%TR0_PY%" -u run_tr0_factorial.py confirm --device cuda --arms B0 C00 C01
```

该命令输出到 `test_result\tr0_v1\seed42\confirm-t1-paired-v1`。按每项指标在三组共同有限的场景上计算配对差，对场景做 10,000 次有放回重采样，报告 95% 百分位区间。区间和逐图差都按收益方向报告，NAB/F 降低记为正。区间反映锁定场景的抽样变化，当前训练种子只有 42。

T1 保护条件以同场景、同评价器的 B0 为参照。QAB/F、VIF、SSIM 和 MS-SSIM 不低于 B0，NAB/F 不高于 B0。继续使用 MI 至少提高 0.5% 或 FMI 至少提高 0.3% 的相对增益门槛。对于 C01 相比 C00 的 SSIM、MI、FMI，分别报告点估计的方向与区间是否完全位于收益方向。相同三组用于后续 `screen` 时也采用 T1 协议。

`models\tr0_v1\seed42` 保存 split manifest、权重和训练日志。评测沿用 `reference_v2_candidate_with_paper_fmi` 和 `clamp_then_uint8`。`paired_scene_differences.csv` 保存逐图配对差，`paired_comparisons.csv` 保存平均变化和置信区间，`common_scenes.csv` 保存各指标的共同样本数，`summary.json` 保存完整指标和保护条件判定。历史四组筛选目录保存原协议结果及 `factorial_effects.csv`。

训练与评测完成后会登记结果并校验索引。如果历史条目指向不可访问的目录，索引校验会报错。恢复历史目录访问后可重执行同一命令登记已有结果。

## C02 延长来源重建

C02 复用 C00 第 4 轮的 E2 模型权重和 AdamW 状态，以相同的 seed42、train1024、逐轮数据顺序、精度和损失继续。第 5–8 轮做普通来源重建，第 9–20 轮做真实融合。第 20 轮用于评测。2,560 次优化更新中，512 次来自复用阶段，实际新增 2,048 次。训练命令可从最近保存的 C02 epoch 接续。

```cmd
cd /d D:\#imageInfusion\CDDFuse
set "TR0_PY=D:\anaconda\envs\lightglue_env\python.exe"
"%TR0_PY%" -u run_tr0_c02.py train --device cuda
"%TR0_PY%" -u run_tr0_c02.py screen --device cuda
"%TR0_PY%" -u run_tr0_c02.py confirm --device cuda
```

两次评测只对 C02 推理，复用相同划分的 B0、C00、C01 逐图指标。新比较组分别写入 `screen-c02-paired-v1` 和 `confirm-c02-paired-v1`，使用四组共同有限的场景计算配对差和区间。比较组记录 C00、C01、C02 的训练耗时、显存峰值、更新步数及第 5–8 轮耗时。C02 的 `training_cost.json` 另外记录继承与实际新增开销。历史 B0 日志缺少同口径成本字段。

C00 与 C01 的第 4 轮模型及优化器状态实际不同。C02 与 C00 共享起点，可比较从相同状态延长重建和重复视图训练的差异。解读 C02 与 C01 的差值时需考虑起点差异。B0 使用历史 FP16 配置，跨这两套训练记录的差异也包含精度和其他运行差异。

## C03-H 外观锚定异构代理学习

C03-H 从 C01 第 4 轮恢复原 E2 权重和 AdamW 状态。第 5–8 轮每个大小为 2 的批次中随机选一个样本保留 V1，另一个使用 V2。V2 继续使用 V1 的模糊尺度和软掩码生成思路，调整清晰区域重叠至约 30%、30%、30%、10%，分别对应两路各自独有清晰、共同清晰和共同退化。光学分支收到锚定视图，SAR 分支收到 `a × V2 + b`。`a` 均匀采样于 `[0.7, 1]`，`b` 均匀采样于 `[0, 1−a]`。参考目标为原始光学图像。

第 9–20 轮保持 C01 的真实双源融合设置，固定第 20 轮权重评测。C03-H 与 C01 的比较检验 V2 区域分布和非对称外观变换的联合作用。C03-H 配置只有训练入口读取，推理沿用原 E2。

```cmd
cd /d D:\#imageInfusion\CDDFuse
set "TR0_PY=D:\anaconda\envs\lightglue_env\python.exe"
"%TR0_PY%" -u run_tr0_hcsl.py smoke --device cuda
"%TR0_PY%" -u run_tr0_hcsl.py train --device cuda
"%TR0_PY%" -u run_tr0_hcsl.py screen --device cuda
"%TR0_PY%" -u run_tr0_hcsl.py confirm --device cuda
```

两次正式评测只对 C03-H 推理，分别写入 `screen-hcsl-paired-v1` 和 `confirm-hcsl-paired-v1`。五组按各指标共同有限的场景计算结果，输出 14 项指标、95% 配对 bootstrap 区间、训练开销和推理耗时。以 B0 执行既有保护条件。相对 C01 的保持门槛预先固定为零退化，MI、VIF、QAB/F、SSIM 的点估计应至少持平，FMI、NAB/F、MS-SSIM 应严格朝收益方向变化。区间用于判断逐图差的不确定性。
