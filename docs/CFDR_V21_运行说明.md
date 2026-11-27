# CFDR-V2.1 实验

## 当前优先执行冻结筛选

冻结已完成训练的 CF0 第 16 轮主干，包括 Encoder、LWPD、Base Fusion、Detail Fusion 和原 Decoder，主干始终使用评估模式。CF1、CF2 只训练新增的 CFDR 模块。两组保留 train8000、种子 42、batch 2 和梯度累积 4，使用 AdamW、FP16 和学习率 `1e-4`。首轮各训练 2 轮，只计算对新增模块有梯度的 `L_I + 10 L_G`。源图重建对 CFDR 参数是常数，因此冻结筛选跳过这两条前向，也省去逐轮验证前向。

```cmd
cd /d D:\#imageInfusion\CDDFuse
set "CFDR_PY=D:\anaconda\envs\lightglue_env\python.exe"
"%CFDR_PY%" -u run_cfdr_frozen.py train --device cuda
"%CFDR_PY%" -u run_cfdr_frozen.py val --device cuda
```

最后一轮的 val1000 复用 CF0 的逐图指标，只对 CF1、CF2 新做推理。评测前逐张量核验两组主干与 CF0 相同。日志保存于 `models\e2_cfdr_v21_frozen\seed42`，配对结果保存于 `test_result\e2_cfdr_v21_frozen\val1000\epoch_002`。先看 `CF1−CF0` 和 `CF2−CF0` 的信息、边缘及伪影取舍，再看 `CF2−CF1` 是否支持邻域聚合的增量。冻结结果显示收益后，再考虑全网训练。

### 两组继续冻结训练到第 4 轮

```cmd
cd /d D:\#imageInfusion\CDDFuse
set "CFDR_PY=D:\anaconda\envs\lightglue_env\python.exe"
"%CFDR_PY%" -u run_cfdr_frozen.py train --device cuda --to-epoch 4
"%CFDR_PY%" -u run_cfdr_frozen.py val --device cuda --to-epoch 4
```

两组分别恢复自己的第 2 轮模型、AdamW 和 FP16 scaler，主干继续冻结，数据种子沿用原规则。新增权重和日志写入 `models\e2_cfdr_v21_frozen\seed42\to_epoch_004`。评测仅为第 4 轮做新推理，复用 CF0 和第 2 轮逐图结果。`test_result\e2_cfdr_v21_frozen\val1000\epoch_004\paired_comparison.csv` 使用五组共同有效的场景，直接报告每组第 4 轮减第 2 轮的变化，以及第 4 轮 CF2 相对 CF1 的变化。

## 条件启动的全网实验

CF0 采用已有光学 Base 锚定 LWPD E2。CF1 在平滑的 E2 解码预测上加入按模态增益加权的原位中心化细节。CF2 增加 8 邻域动态核和聚合强度，单模态原位权重至少为 0.5。条件网络用 1×1 卷积将 `[H_dec, D_F]` 映射至 32 通道，再以 3×3 逐通道卷积提取邻域条件。CF1 参数量为 1,006,379，CF2 为 1,006,973，差值为邻域和聚合强度预测头的 594 个参数。

两组从同一份 CF0 第 4 轮单源重建权重接续，恢复历史 AdamW 和 FP16 scaler。共享融合主干及两路单源重建均保留。新头使用融合组学习率 `1e-4`，其余学习率按 CF0 原 Stage II 配置。融合阶段为 12 轮，使用 train8000、val1000、种子 42、batch 2、梯度累积 4 和原 `L_I + 10 L_G + 0.1 L_R`。最终融合输出直接回传现有损失，按统一评价器在输出图像时裁剪及量化。

旧 CF0 checkpoint 记录了权重、优化器和 scaler，未保存数据采样器状态。CF1 与 CF2 使用相同的逐轮种子和数据顺序，CF0 对照来自其原有训练轨迹。CF2 的初始化中心权重约 0.90，初始模态增益约为 `(0.475, 0.475, 0.05)`。

## Windows CMD

```cmd
cd /d D:\#imageInfusion\CDDFuse
set "CFDR_PY=D:\anaconda\envs\lightglue_env\python.exe"
"%CFDR_PY%" -u run_cfdr.py smoke --device cuda
"%CFDR_PY%" -u run_cfdr.py train --device cuda
"%CFDR_PY%" -u run_cfdr.py val --device cuda --fusion-epochs 4 8 12
```

`train` 默认依次训练 CF1 和 CF2。可指定 `--arms CF1` 或 `--arms CF2`。中断后重执行同一命令会接续最近已保存的 epoch。Stage II 的第 4、8、12 轮分别对应 `epoch_008.pth`、`epoch_012.pth` 和 `epoch_016.pth`。训练日志位于 `models\e2_cfdr_v21\seed42`。

最终 val1000 复用已有 CF0 逐图指标。中间两轮使用各自 CF0 已有权重补做 val1000 推理。每个节点在单独比较组报告 14 项全图指标、各指标共同有效的场景数及 `CF1−CF0`、`CF2−CF1`、`CF2−CF0` 的 95% 配对区间。结果位于 `test_result\e2_cfdr_v21\val1000`。

全量验证完成后，用 `run_cfdr.py test --device cuda` 评测固定第 12 轮 Stage II 的 test1000。已有 CF0 归档通过协议和图像完整性检查时直接复用，CF1、CF2 保存全部三类图像。正式测试结果使用独立比较组。

## H-CFDR 冻结筛选

H2 使用同一个 CF0 第 16 轮作为冻结主干。新增模块从与 CF2 相同的共享预测头初始权重开始训练，并增加一个 33 参数的适配强度预测头，训练参数共 5,174 个。光学侧沿用原动态滤波，SAR 侧按 3×3 中心化邻域对光学方向做 FP32 投影，只调整反向投影分量。投影正则项为 `1e-4`，适配强度初始值为 `0.05`。中心权重下限作用于适配后的 SAR 邻域。

沿用 CF2 的冻结两轮协议、train8000、seed42、FP16、batch 2、梯度累积 4 和 `L_I + 10 L_G`。H0 和 H1 分别复用前一轮 CF0 和 CF2 的 val1000 逐图结果，最终只对 H2 做全图推理。

```cmd
cd /d D:\#imageInfusion\CDDFuse
set "CFDR_PY=D:\anaconda\envs\lightglue_env\python.exe"
"%CFDR_PY%" -u run_hcfdr_frozen.py train --device cuda
"%CFDR_PY%" -u run_hcfdr_frozen.py val --device cuda
```

训练输出位于 `models\e2_hcfdr_v1_frozen\seed42\H2`。日志记录平均适配强度、反向投影区域比例、该区域的平均适配强度和实际邻域改变量。评测位于 `test_result\e2_hcfdr_v1_frozen\val1000\epoch_002`，按三组共同有限场景汇总 14 项指标，重点查看 `H2−H1` 的配对差和 95% 区间。

### CF1、CF2、H2 共同续训至第 4 轮

三组各从自己的第 2 轮 checkpoint 恢复新增模块、AdamW 状态和 FP16 scaler，沿用冻结的 CF0、train8000、原损失和逐轮种子规则。新增训练只执行第 3、4 轮。输出保存在各模型根目录的 `to_epoch_004`，第 2 轮结果继续作为同场景对照。

```cmd
cd /d D:\#imageInfusion\CDDFuse
set "CFDR_PY=D:\anaconda\envs\lightglue_env\python.exe"
"%CFDR_PY%" -u run_cfdr_frozen.py train --device cuda --to-epoch 4
"%CFDR_PY%" -u run_hcfdr_frozen.py train --device cuda --to-epoch 4
"%CFDR_PY%" -u run_hcfdr_frozen.py val --device cuda --to-epoch 4
```

联合评测写入 `test_result\e2_hcfdr_v1_frozen\val1000\epoch_004_joint_v1`。复用 CF0 和三组第 2 轮指标，已有匹配的 CF1、CF2 第 4 轮结果也直接复用。配对表包含第 4 轮横向比较和三组各自的第 4 轮相对第 2 轮差值，采用共同有限场景与 95% 配对区间。

### 一键冻结续训和评测

```cmd
cd /d D:\#imageInfusion\CDDFuse
"D:\anaconda\envs\lightglue_env\python.exe" -u run_cfdr_frozen_study.py --device cuda
```

默认依次推进第 4、6、8 轮。每个节点完成 CF1、CF2、H2 的同预算训练后立即执行同一个 val1000 比较。通过 `--to-epoch 4` 或 `--to-epoch 6` 可先停在该节点查看结果，之后用更高的目标轮次重跑同一命令。脚本复用已完成的 checkpoint 和逐图指标，续训恢复上一节点的模型、AdamW 和 FP16 scaler，始终冻结同一份 CF0 主干。

各节点输出位于 `test_result\e2_hcfdr_v1_frozen\val1000\epoch_004_joint_v1`、`epoch_006_joint_v1` 和 `epoch_008_joint_v1`。`training_curves.csv` 汇总损失、学习率、邻域权重和极性适配量，第 5–8 轮另有预测头梯度范数。`validation_curves.csv` 按当前节点纳入的所有方法和轮次的共同有限场景汇总 14 项指标。是否提前结束由研究者根据节点结果决定。

选定轮次后执行冻结版正式 test1000。以下以已完成 val1000 的第 4 轮为例。脚本核对选定轮的权重与 val1000 记录，复用同口径 CF0 测试档案，只对 CF1、CF2、H2 做新推理。

```cmd
"D:\anaconda\envs\lightglue_env\python.exe" -u run_cfdr_frozen_test.py --device cuda --epoch 4
```

结果写入 `test_result\e2_hcfdr_v1_frozen\test1000\epoch_004_joint_v1`，包括每个新增方法的全部灰度图、RGB 图和三联预览，四组共同有限场景上的 14 项逐图指标、均值、标准差和配对 95% 区间。正式结果使用独立比较组。
