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

如果筛选结果支持继续验证，对锁定的 val128 执行全图确认。按需训练普通延长重建和 Early 对照，再在同一批场景上比较。

```cmd
"%TR0_PY%" -u run_tr0_factorial.py confirm --device cuda
"%TR0_PY%" -u run_tr0_factorial.py train --device cuda --resume --arms R8 EARLY00 EARLY01
"%TR0_PY%" -u run_tr0_factorial.py confirm --device cuda --arms B0 C00 C01 C10 C11 R8 EARLY00 EARLY01
```

`models\tr0_v1\seed42` 保存 split manifest、权重和训练日志。`test_result\tr0_v1\seed42` 保存逐图指标、评测元数据和配对汇总。评测沿用 `reference_v2_candidate_with_paper_fmi` 和 `clamp_then_uint8`，每项对同组方法共同有限的场景计算均值和效应。`factorial_effects.csv` 提供原始尺度的 ΔT、ΔA、ΔTA 及按指标方向翻转的增量。`summary.json` 中的筛选条件沿用仓库原有 MI/FMI、QAB/F、NAB/F、VIF、SSIM、MS-SSIM 门槛。代理诊断对相同场景和随机视图比较互补输入、重复第一视图、重复第二视图。

训练与评测完成后会登记结果并校验索引。如果历史条目指向不可访问的目录，索引校验会报错。恢复历史目录访问后可重执行同一命令登记已有结果。
