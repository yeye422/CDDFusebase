# E2-Adapt 融合适配协议

## 目的

让 C0 的 BaseFuse/DetailFuse 适应 SPD 净化后的 SAR 特征，并用严格匹配的 C0-Adapt 排除“额外训练轮数”带来的收益。

## 两组实验

### C0-Adapt

```text
Optical/SAR -> 冻结 Encoder -> 原始 Base/Detail
            -> 训练 BaseFuse/DetailFuse
            -> 冻结 Decoder -> F
```

### E2-Adapt

```text
Optical -> 冻结 Encoder -> Optical Base/Detail
SAR     -> 冻结 Encoder -> 冻结 SPD -> 净化 Base/Detail
        -> 训练 BaseFuse/DetailFuse
        -> 冻结 Decoder -> F
```

两组使用完全相同的：

- 正式 C0 epoch-5 checkpoint；
- QXS train/val 样本；
- 随机种子和数据增强；
- 5 epochs、学习率和优化器；
- C0 固定 frequency reference；
- checkpoint 选择规则。

## 可训练与冻结模块

| 模块 | C0-Adapt | E2-Adapt |
|---|---:|---:|
| Encoder | 冻结 | 冻结 |
| SPD | 不使用 | 冻结 |
| Denoising Head | 不使用 | 冻结 |
| BaseFuse | 训练 | 训练 |
| DetailFuse | 训练 | 训练 |
| Decoder | 冻结 | 冻结 |

## 损失

继续使用正式 C0 的 reference-normalized smooth minimax：

```text
r_opt = L_optical / 0.38274383956938984
r_sar = L_sar_mid / 0.003583242081978824
L = 0.1 * logsumexp([r_opt, r_sar] / 0.1)
```

不得为 E2 单独重新校准 reference。

## Pilot 命令

该命令属于训练任务，由用户在 Windows CMD 中执行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_e2_adapt_pilots.cmd
```

输出：

```text
models\v21_softmask\c0_adapt_pilot_seed42
models\v21_softmask\e2_adapt_pilot_seed42
```

## 完整 8000/1000 训练

用户决定跳过 Pilot 数据规模，直接使用完整 QXS 划分：

- train：8000 对；
- validation：1000 对；
- test：1000 对保持不参与训练和选模；
- epoch：5；
- batch size：8；
- patch：128；
- seed：42。

执行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_e2_adapt_full_seed42.cmd
```

输出：

```text
models\v21_softmask\c0_adapt_full_seed42
models\v21_softmask\e2_adapt_full_seed42
```

完整训练仍然必须同时运行 C0-Adapt。不能只训练 E2-Adapt，否则无法排除额外五轮 Fusion 微调的收益。

## 判定

E2-Adapt 必须同时与以下方法比较：

1. 原始 C0；
2. E2 Zero-shot；
3. C0-Adapt；
4. E2-Adapt。

只有 E2-Adapt 优于 C0-Adapt，才能把收益归因于 SPD；如果两者相近，改善主要来自额外 Fusion 训练。

Pilot 通过建议：

- NAB/F 不高于 E2 Zero-shot；
- QAB/F、VIF 不低于 E2 Zero-shot 的 99%；
- FMI 高于 E2 Zero-shot；
- Output–Optical MAE 不低于 0.01；
- 相对 C0-Adapt 至少在 FMI、QAB/F、NAB/F、VIF 中三项更优。

## 完整 8000/1000 训练

当前决定跳过 Pilot，直接对 E2-Adapt 执行完整训练：

- train：8000 对；
- validation：1000 对；
- test：1000 对，训练期间不读取；
- epochs：5；
- batch size：8；
- patch：128×128；
- seed：42。

Windows CMD：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_e2_adapt_full_seed42.cmd
```

输出：

```text
models\v21_softmask\e2_adapt_full_seed42
```

根据 validation 选定 `best.pth` 后，才能对冻结模型运行正式 test1000；不得根据 test 结果选择 checkpoint。

### 控制实验变更记录

2026-10-09：用户决定取消完整数据的 C0-Adapt 匹配控制，只训练 E2-Adapt。此前 E2 Zero-shot 与正式 C0-minimax 的结果继续保留。该决定将训练成本减半，但会削弱“适配后额外收益完全来自 SPD”的因果归因；最终报告必须注明未执行同预算 C0-Adapt。

## 正式 test1000

训练完成并由 validation 选出 `best.pth` 后运行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_e2_adapt_test1000.cmd
```

输出：

```text
test_result\v21_softmask\e2_adapt_full_seed42_test1000
```

`summary.json` 中的 `methods.e2_feature` 是最终 E2-Adapt；`methods.c0_raw` 是“已经适配过的 Fusion 改用原始 SAR 特征”的路径消融，不是历史 C0-minimax，不能将它作为 C0 基线登记。
