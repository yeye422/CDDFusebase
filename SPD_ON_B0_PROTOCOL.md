# B0 + SPD 零样本接入实验

## 目的

不经过 C0-minimax，直接把已训练 SPD 插入原始 Optical–SAR CDDFuse B0：

```text
Optical -> B0 Encoder ------------------------------┐
                                                    ├-> B0 BaseFuse/DetailFuse -> B0 Decoder
SAR     -> B0 Encoder -> frozen SPD -> clean feature┘
```

比较：

- `c0_raw`：原始 B0；
- `e1_image`：SPD 去噪图 + B0；
- `e2_feature`：B0 + SPD 净化 Base/Detail。

该实验不训练参数，只回答 SPD 是否能直接改善信息量较高、但伪影较多的 B0。

## 正式 test1000

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_spd_on_b0_test1000.cmd
```

输出：

```text
test_result\v21_softmask\spd_nomask_on_b0_test1000
```

`methods.c0_raw` 必须精确复现已登记 B0；`methods.e2_feature` 才是 B0+SPD。

## Base/Detail 修正强度扫描

在 validation 1000 上扫描以下 ((\alpha_B,\alpha_D))：

```text
(0,0), (0.25,0.25), (0.5,0.5), (0.75,0.75), (1,1),
(1,0), (0,1), (0.5,1), (1,0.5)
```

只用于选择强度，不读取 test：

```cmd
cd /d D:\#imageInfusion\CDDFuse
run_spd_scale_sweep_val1000.cmd
```

输出：

```text
test_result\v21_softmask\spd_scale_sweep_b0_val1000
```

- `summary.csv/json`：均值、相对 B0 差值、保留率、逐图改善比例和 Pareto 标记；
- `per_image.csv`：9 组逐图 14 项指标。

选定强度后才能在 test1000 上做一次冻结确认，不得根据 test 反选强度。
