# E2 + PAC-Mamba 实施与执行说明

日期：2026-11-04。按本次要求保留原E2损失，仅接入已有PAC-Mamba模块。用户所写PCA-mamba在本次实现中按仓库已有PAC-Mamba理解，没有引入PCA降维。

## 已实现

- 新模型 `PACMambaE2CDDFuse`，独立variant：`pac_mamba_e2`。
- E2骨干：固定Haar、方向上下文门控、原Base融合及原Detail融合保留路径；LWPD mode为gated，不添加E3恢复或E4相位对称恢复分支。
- 复用现有PAC模块：光学/SAR相位反对称PA局部匹配、学习可靠度、半分辨率双轴正反向选择性SSM、零初始化残差更新。
- PAC只添加特征残差，不替换原E2的保留路径。初始化输出与原E2逐值一致；PA证据用于残差交互门控，不用作新的损失目标。
- 使用现有纯PyTorch选择性扫描实现，不新增编译依赖。
- `test_OS.py --variant auto`支持新模型、14项指标、灰度/RGB/三联图及PAC门控图；旧 `pac_mamba` 仍表示E4骨干，避免加载错误。

## 损失与初始化

保留：

`L = L_intensity + 10 * L_gradient + 0.1 * L_reconstruction`。

强度、梯度、重建定义复用原E2代码，不引入SSIM融合项、PA方向损失或BGC1目标。原有重建项本身包含的结构计算保持原样。

从已有历史E2初始化：

`models/lwpd_qxs_ablation_v1/e2_seed42/epoch_016.pth`

SHA-256：`c2af2a8182b85276405f2a0892bb4a07e28fa4649429105444e573d985928444`。

比较基线复用：`test_result/lwpd_qxs_ablation_v1/val1000/e2_seed42`。不混用匹配损失实验E2-B的均值，也不默认重训任何E2基线。

## 训练安排

延续已有PAC训练安排：2 warmup + 3 local + 7 joint，共12轮，seed42，batch2、累积4，固定train8000/val1000。

- Warmup：编码器冻结，PAC以1e-4学习，旧LWPD/融合器/Decoder以1e-5更新。
- Local：开放编码器后段；PAC为1e-4，旧模块2e-5，编码器后段5e-6。
- Joint：开放全编码器；PAC为5e-5，旧模块1e-5，编码器后段5e-6、前段1e-6。
- 梯度裁剪1、weight_decay=1e-4，完整训练延续既有AMP设置。
- 固定最终 `epoch_012.pth` 评价，不根据val结果事后选择epoch；最低val loss checkpoint仅诊断保存。

这是在训练好的E2上再训练12轮。与原E2的比较包含额外训练与模块的共同收益，不能单独证明PAC模块的因果增益。当前按要求复用原E2，不自动安排相同额外预算的无PAC微调对照；如后续做严格模块归因，应另立该对照，不能把它暗中加入默认命令。

## 已完成的工程验证

2026-11-04已执行：

1. 5项新增单元/集成检查通过：初始化与E2输出逐值一致；原损失各分项及所检查Decoder参数梯度一致；新checkpoint评价加载与门控图路径；拒绝错误骨干checkpoint；PAC输出投影及SSM路径可反传。
2. 原PAC-Mamba全部7个脚本检查通过，覆盖旧E4初始化等路径。
3. 真实单样本FP32前向反传通过，无优化器更新：loss=0.855303，interaction_mean=0.077888，梯度有限。

单样本输出：`models/e2_pac_mamba_smoke_v1/seed42/dry_run_diagnostics.json`。已登记为 `E2-PAC-MAMBA-V1-SEED42-SMOKE`，实验注册检查通过，共70条。

初始残差为零，日志中的relative_update_rms约0.000030来自原监控公式的epsilon下限，不代表已有非零PAC更新。零投影使上游SSM/可靠度首步梯度为零属于该初始化行为；投影开始学习后，上游路径应获得梯度，单元检查已覆盖其可达性。

没有运行完整训练、val1000或test1000，也没有修改历史结果。

## Windows CMD执行

推荐训练与验证一次完成：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_e2_pac_mamba.cmd train-val
```

也可分开执行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_e2_pac_mamba.cmd train
run_e2_pac_mamba.cmd val
```

仅运行单元检查：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_e2_pac_mamba.cmd check
```

smoke已运行且目录非空，再次执行smoke会拒绝覆盖；不需要重复执行。

## 输出与复用行为

- 新模型目录：`models/e2_pac_mamba_v1/seed42/`。
- 训练源码/参数/初始化哈希锁：`models/e2_pac_mamba_v1/workflow_protocol.json`，首次训练前创建。
- 新验证目录：`test_result/e2_pac_mamba_v1/val1000/seed42/`。
- 与已有E2配对摘要：上述目录内 `comparison_vs_e2/SUMMARY.md`；包含14项变化、逐图改善比例及配对正态近似区间。
- 完成训练或验证后自动登记并执行实验索引检查；验证比较组沿用相同口径的 `QXS-VAL1000-DIRECT-LWPD-v1`，脚本核对组、清单、全部scene ID、评价口径及输出策略。
- 默认只运行缺失候选；已完成训练跳过，已有验证先核对而非覆盖，已有checkpoint时可继续训练；不完整验证目录拒绝覆盖。
- 源码或协议变化必须新建版本及唯一目录，不能强行复用旧协议锁。
- 默认不包含test1000，不因val loss降低自动推进正式测试。先联合查看QAB/F、NAB/F、SSIM/MS-SSIM、FMI/SCD、MI/VIF以及源图结构；配对图区间不代表训练种子方差。

## 代码入口

- 模型：[pac_mamba.py](./pac_mamba.py)。
- 训练：[train_pac_mamba.py](./train_pac_mamba.py)。
- 流程：[run_e2_pac_mamba.py](./run_e2_pac_mamba.py)、[CMD入口](./run_e2_pac_mamba.cmd)。
- 验证：[test_OS.py](./test_OS.py)。
- 新检查：[test_pac_mamba_e2.py](./tests/test_pac_mamba_e2.py)。
