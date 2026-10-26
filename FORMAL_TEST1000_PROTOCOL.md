# 正式 test1000 输出协议

## 强制输出

每次 QXS-SAROPT 正式 test1000 必须产生：

```text
<unique_output_dir>/
  summary.json 或 metrics.csv
  per_image.csv 或 metrics.csv中的逐图行
  images/<method>/*.png       1000张灰度融合图
  rgb/<method>/*.png          1000张RGB融合图
  preview/<method>/*.png      1000张Optical/SAR/Fused三联图
```

评价固定为：

```text
split=test
offset=0
limit=1000
metric_profile=paper
output_policy=clamp_then_uint8
```

完整14项指标：

```text
EN, MI, FMI, PSNR, Qabf, Nabf, AG, SF, SD, CC, SCD, SSIM, MS_SSIM, VIF
```

## 固定五方法对比

正式结果完成后运行：

```cmd
cd /d D:\#imageInfusion\CDDFuse
.venv\Scripts\python.exe build_five_method_test1000_comparison.py
.venv\Scripts\python.exe maintain_experiment_index.py --check --render
```

输出：

```text
test_result\v21_softmask\five_method_test1000_comparison\comparison_14_metrics.csv
test_result\v21_softmask\five_method_test1000_comparison\comparison_14_metrics.md
test_result\v21_softmask\five_method_test1000_comparison\comparison_14_metrics.json
```

五种固定方法：

1. B0；
2. B0 + SPD；
3. B0 + Detail-only SPD；
4. C0-minimax；
5. C0-minimax + SPD。

## 三种 SPD 方法补齐完整可视化

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_three_spd_full_visual_test1000.cmd
```

该脚本使用三个新的 `fullviz_v1` 目录，拒绝覆盖已有目录，并分别保存 B0+SPD、B0+Detail-only-SPD、C0-minimax+SPD 的1000张灰度/RGB/三联融合图及完整14项指标。

## 目录规则

- 每个新模型或新推理策略必须使用新的唯一输出目录；
- 不得覆盖已有正式目录；
- 历史目录中只保存8张样例的结果维持原状，不回写或伪装成完整可视化；
- 若要补齐历史可视化，使用带版本后缀的新目录并重新登记。
