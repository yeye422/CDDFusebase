# SPD 最小实现协议

## 已实现范围

- 冻结 C0 `Restormer_Encoder`；
- Base/Detail 两条零初始化、有界残差净化分支；
- 独立 SAR 去噪 Head；
- 配对 SAR noisy/reference 数据加载；
- 可选的 (M_L)、(M_P)、参考质量 Soft Mask；
- Charbonnier、SSIM、线结构、亮点结构和扰动一致性损失；
- checkpoint、配置、split manifest 和训练历史保存；
- dry-run 与单元测试。

当前不包含 (Q_S)。按照 V6 计划，必须先完成 SPD，再冻结 SPD 生成误差事件标签，单独训练 Reliability Head。

## 数据目录约定

```text
SAR_DENOISE_ROOT/
  noisy/             含噪 SAR
  reference/         多时相或其他方法生成的近似去噪参考
  masks_line/        可选，0--255 PNG Soft Mask
  masks_point/       可选，0--255 PNG Soft Mask
  masks_reference/   可选，参考质量权重
```

各目录必须使用完全相同的文件 stem。正式训练前应建立场景级 split manifest；若不提供，脚本生成的随机图像级划分只允许工程试跑。

## 当前采用的数据集

本地路径：

```text
D:\#imageInfusion\datasets\SAR despeckling filters dataset\SAR despeckling filters dataset\Main folder
```

已审计：

- `Noisy/GTruth`：1500 对训练图；
- `Noisy_val/GTruth_val`：100 对官方验证图；
- 图像为 512×512、uint8 TIFF；
- 三个通道完全相同，加载时安全转换为单通道；
- 文件名是原始大图坐标，训练和官方验证无 stem 重叠；
- 该数据由 Toronto 区域 Sentinel-1 VV 多时相配准与十景平均构造近似 GT；
- 仅覆盖一个区域和一种传感器/极化，不能单独证明跨区域泛化。

SPD 训练器使用官方目录拆分，并将官方 100 对按第一坐标组进一步分为：`5120_*` 用于模型选择，`5632_*` 作为内部确认。二者空间相邻，因此最终论文仍需外部数据确认。

## 单元测试

当前环境无需额外安装 pytest，可直接执行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
.venv\Scripts\python.exe tests\test_spd.py
```

如果环境中已安装 pytest，也可以使用 `python -m pytest tests\test_spd.py -q`。

## Dry-run

下面仅为接口模板，不应在缺少真实配对数据时执行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
.venv\Scripts\python.exe train_spd.py --noisy-dir D:\DATA\SAR_DENOISE\noisy --reference-dir D:\DATA\SAR_DENOISE\reference --init-checkpoint models\reliability_v5\frequency_minimax_baseline_pilot\epoch_005.pth --output-dir models\v21_softmask --run-name spd_dry_run --train-limit 32 --val-limit 8 --batch-size 2 --dry-run
```

## 带 Soft Mask 的 Pilot

该命令属于训练任务，只在 Mask 审计和数据审计通过后由用户执行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
.venv\Scripts\python.exe -u train_spd.py --noisy-dir D:\DATA\SAR_DENOISE\noisy --reference-dir D:\DATA\SAR_DENOISE\reference --line-mask-dir D:\DATA\SAR_DENOISE\masks_line --point-mask-dir D:\DATA\SAR_DENOISE\masks_point --reference-mask-dir D:\DATA\SAR_DENOISE\masks_reference --split-manifest D:\DATA\SAR_DENOISE\splits\scene_split.json --init-checkpoint models\reliability_v5\frequency_minimax_baseline_pilot\epoch_005.pth --output-dir models\v21_softmask --run-name spd_pilot_seed42 --epochs 15 --batch-size 8 --patch-size 128 --lr 1e-4 --line-weight 0.5 --point-weight 0.25 --consistency-weight 0.1 --consistency-sigma 0.03 --seed 42 --amp
```

## 当前数据的无 Soft Mask Pilot

以下是应由用户执行的训练命令；本次实现过程未运行该训练：

```cmd
cd /d D:\#imageInfusion\CDDFuse
.venv\Scripts\python.exe -u train_spd.py --noisy-dir "D:\#imageInfusion\datasets\SAR despeckling filters dataset\SAR despeckling filters dataset\Main folder\Noisy" --reference-dir "D:\#imageInfusion\datasets\SAR despeckling filters dataset\SAR despeckling filters dataset\Main folder\GTruth" --val-noisy-dir "D:\#imageInfusion\datasets\SAR despeckling filters dataset\SAR despeckling filters dataset\Main folder\Noisy_val" --val-reference-dir "D:\#imageInfusion\datasets\SAR despeckling filters dataset\SAR despeckling filters dataset\Main folder\GTruth_val" --official-val-test-fraction 0.5 --init-checkpoint models\reliability_v5\frequency_minimax_baseline_pilot\epoch_005.pth --output-dir models\v21_softmask --run-name spd_nomask_seed42 --epochs 30 --batch-size 8 --patch-size 128 --lr 1e-4 --base-feature-weight 0.2 --detail-feature-weight 0.05 --consistency-weight 0.1 --consistency-sigma 0.03 --invalid-threshold -1 --seed 42 --amp
```

## 从训练结构中删除 Base SPD

根据 val1000 与正式 test1000 的 Base/Detail 强度消融，Base SPD 是 FMI/VIF 损失的主要来源。已实现新的 `detail_only` 训练结构：

```text
Base   = 原始冻结 Encoder Base，严格旁路
Detail = 可训练 SPD Detail 修正
Denoising Head = 原始 Base + 净化 Detail
```

损失删除 `L_B`，一致性只约束 Detail：

```text
L = L_charb + 0.5 L_ssim + 0.05 L_D + 0.1 L_cons_detail
```

训练命令：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_spd_detail_only_training_seed42.cmd
```

输出：

```text
models\v21_softmask\spd_detail_only_seed42
```

该模型从零初始化 Detail SPD 与 Denoising Head，不继承原完整 SPD，以保证训练结构消融可归因。

### 冻结决策

2026-10-12：采用重新训练的 Detail-only SPD 作为当前主模型。选择依据是训练/推理结构一致、参数减少约45%和更清晰的算法解释，而不是 validation 上的微小指标差异。冻结 checkpoint：

```text
models\v21_softmask\spd_detail_only_seed42\best.pth
SHA-256: 2f79541e4220bd2544e3b296407b7f1125c95ddb677a2822dc4eb35063d932ff
```

正式 test1000 命令：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_spd_detail_only_retrained_on_b0_test1000.cmd
```

该正式测试完成后不得根据 test 结果切回旧 checkpoint 或调节修正强度。

## 判定要求

SPD 必须与以下对照比较：

1. 原始 SAR + C0；
2. 外部去噪 + C0；
3. SPD，不使用 Soft Mask；
4. SPD + (M_L)；
5. SPD + (M_P)；
6. SPD + (M_L+M_P)。

如果 SPD 仅提高参考 PSNR/SSIM，却降低线连续性、亮点保持或最终融合的 QAB/F、FMI、VIF，则不得进入 (Q_S) 训练。

## 定量评价

内部确认使用未参与 checkpoint 选择的 `5632_*` 共 50 对，按训练尺度进行 128×128 重叠分块推理：

```cmd
cd /d D:\#imageInfusion\CDDFuse
.venv\Scripts\python.exe evaluate_spd.py --noisy-dir "D:\#imageInfusion\datasets\SAR despeckling filters dataset\SAR despeckling filters dataset\Main folder\Noisy_val" --reference-dir "D:\#imageInfusion\datasets\SAR despeckling filters dataset\SAR despeckling filters dataset\Main folder\GTruth_val" --spd-checkpoint models\v21_softmask\spd_nomask_seed42\best.pth --encoder-checkpoint models\reliability_v5\frequency_minimax_baseline_pilot\epoch_005.pth --split test --official-val-test-fraction 0.5 --tile-size 128 --tile-overlap 32 --save-images 8 --output-dir test_result\v21_softmask\spd_nomask_seed42_internal_test50
```

输出：

- `per_image.csv`：逐图 Noisy/SPD 的 MAE、MSE、PSNR、SSIM、边缘误差和边缘相关性；
- `summary.json`：均值、差值、中位差、改善比例、Base/Detail 特征误差变化；
- `images/`：固定前 8 对 Noisy、SPD、Reference 图像。

## 融合价值 E1/E2 对照

该评价比较：

- `c0_raw`：原始 SAR + C0；
- `e1_image`：SPD 去噪图重新经过 C0 Encoder 后融合；
- `e2_feature`：直接使用 SPD 净化后的 Base/Detail 特征融合。

开发阶段在与既有 C0 实验相同的 val200 上运行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
.venv\Scripts\python.exe compare_spd_fusion.py --data-root ..\datasets\QXSLAB_SAROPT --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json --split val --offset 0 --limit 200 --c0-checkpoint models\reliability_v5\frequency_minimax_baseline_pilot\epoch_005.pth --spd-checkpoint models\v21_softmask\spd_nomask_seed42\best.pth --save-images 8 --output-dir test_result\v21_softmask\spd_nomask_fusion_val200
```

该 val200 已参与既往开发，只能用于判断是否继续研究，不能作为最终独立测试。
