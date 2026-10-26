# PDCF P2：最小单路融合与同容量对照

## 补充诊断：有符号修正方向

当前正倍率扫描只能判断既有修正方向在 `0 < alpha <= 1` 时是否有害，不能判断其反方向是否改善，也不能核验代理损失变化与最终融合指标是否一致。补充诊断使用：

```cmd
run_pdcf_p2_signed_scale_diagnostic_val200.cmd
```

该命令不训练模型，使用唯一的新输出目录，并同时保存：

- `alpha = -0.10 ... +0.10` 的有符号尺度结果以及显式 `alpha=0` 恒等候选；
- 每个 paper 指标相对 C0 的逐图配对 bootstrap 均值差 95% 区间；
- 原 frequency-minimax、Optical、Structure 代理损失及其相对 C0 的变化；
- 代理损失变化与各指标变化的 Pearson、Spearman 相关性；
- 32×32 局部块的输出 RMS、梯度向量偏移 P95/最大值和超预算比例。

所有差值统一定义为 `candidate - C0`。因此 QAB/F、FMI、VIF 等指标的正差值表示改善，NAB/F 的负差值表示改善。bootstrap 参数、局部块大小、阈值、checkpoint 与 split SHA 必须写入 `summary.json`；不能只在注册表备注中声明置信区间。

## 固定候选与匹配控制的独立确认

有符号 val200 诊断完成后，不能先查看后800张再追加控制。应先在前200张上只按候选的平均输出 RMS 校准两种非学习控制，然后一次性在 `val offset=200, limit=800` 上比较：C0、Predictable/Residual `alpha=-0.05`、Raw Evidence `alpha=-0.05`，以及分别与两种Adapter匹配RMS的 Unsharp 和 C0 Detail Gain。Raw与Predictable的开发集RMS不同，禁止用一套控制冒充同时匹配两者。

```cmd
run_pdcf_p2_signed_controls_holdout800.cmd
```

校准过程禁止读取融合指标，只允许匹配 clamp 后相对 C0 的平均输出 RMS。校准集与确认集必须无重叠；脚本会拒绝其他倍率、重叠切片、已有输出目录及未达到 RMS 容差的控制。结果必须同时保存候选相对 C0以及 PDCF/Raw 与两个简单控制之间的逐图配对 bootstrap 区间。

## 目的

P2 不加入可逆 Refactor、双路径 Fusion 或 SPD。它只检验：

> P1 得到的 Predictable/Residual 表示，是否比原始归一化特征及其简单差异证据更适合训练融合修正。

## 严格冻结范围

- C0 Encoder：冻结；
- C0 BaseFuse/DetailFuse：冻结；
- C0 Decoder：冻结；
- P1 Predictor 和归一化统计：冻结；
- 仅训练 SinglePath Fusion Adapter。

Adapter 输出 `ΔBase` 和 `ΔDetail`，最后一层零初始化，因此两个实验在训练前都严格等于 C0。

## 匹配实验

| 输入模式 | 四组输入 |
|---|---|
| `predictable_residual` | Predictable Optical、Predictable SAR、Residual Optical、Residual SAR |
| `raw_evidence` | 原始归一化 Optical、SAR、绝对差异、逐点乘积 |

两组使用完全相同的 Adapter、参数量、深度、训练数据、随机种子、损失和预算。`raw_evidence` 是强对照，不只是简单复制输入。

修正量按 C0 Base/Detail 的逐通道 RMS 缩放，并限制在 25% 相对幅度内。默认不使用额外 anchor 或 correction penalty，避免零初始化模型被强制留在 C0。

## 已完成检查

- 两种输入模式初始化时均严格复现 C0；
- 两种模式可训练参数量完全一致；
- C0 与 Predictor 不接收梯度；
- 初始梯度能够到达 Adapter 输出层；
- Base/Detail 相对修正幅度不超过配置上限；
- 两种模式 GPU 单批次 dry-run 均通过且初始损失完全相同。

## Dry-run

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_pdcf_p2_dry_run.cmd
```

## Pilot 训练

该命令分别训练两组 1000 对、5 epoch Pilot。属于训练任务，由用户执行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_pdcf_p2_pilots_seed42.cmd
```

## val200 匹配评价

Pilot 完成后执行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_pdcf_p2_pilots_val200.cmd
```

评价同时输出 C0、Predictable/Residual、Raw Evidence 三种路径的完整 14 项指标、逐图差值、修正幅度和固定预览。

## 判定

只有同时满足以下条件，才认为 P2 支持可预测分解：

1. `predictable_residual` 至少在 QAB/F、FMI、VIF 中两个优于 C0；
2. NAB/F 不超过预设恶化容差；
3. `predictable_residual` 优于同容量 `raw_evidence`，而不是仅优于 C0；
4. 改善不是由趋近 Optical、整体平滑或修正量塌缩造成；
5. 输出变化达到可见且合理的幅度；
6. val200 通过后才进入完整 val1000，仍不运行 test1000。

如果 Raw Evidence 达到相同或更好结果，则停止把 Predictable/Residual 作为主要贡献，保留更简单的融合结构。

正式结果生成后必须更新 `EXPERIMENT_REGISTRY.json` 并重新生成实验索引。
