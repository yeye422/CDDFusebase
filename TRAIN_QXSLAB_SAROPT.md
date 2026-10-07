# QXS-SAROPT训练与测试方案

## 数据核验

本地目录 `datasets/QXSLAB_SAROPT` 已核验：

- `opt_256_oc_0.2`：20,000张256×256 RGB PNG；
- `sar_256_oc_0.2`：20,000张256×256单通道PNG；
- 文件编号1--20000完全配对，无缺失或多余文件。

数据集原始论文：https://arxiv.org/abs/2103.08259

## 划分选择

固定划分使用10,000对，另外10,000对暂不参与：

| 划分 | 数量 | 用途 |
| --- | ---: | --- |
| 训练 | 8,000 | 参数更新 |
| 验证 | 1,000 | 阶段内最佳检查点选择 |
| 测试 | 1,000 | 最终融合指标与方法对比 |
| 未使用 | 10,000 | 后续扩展、消融或外部验证 |

理由：CDDFuse总参数量约118.8万，远小于大规模扩散模型，并从已有光学--SAR
权重继续微调；8,000对与此前7,381对Dongying训练规模相近。已有QXS-SAROPT
研究常采用16,000/4,000或8:1:1划分，但这些方法通常从零训练更大网络。当前方案
在训练时间、模型容量和1,000对稳定测试之间更均衡。

随机种子为 `20260924`。划分采用无放回随机抽样，分配完成后各清单按编号排序。
清单文件：

```text
../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.json
../datasets/QXSLAB_SAROPT/splits/cddfuse_8000_1000_1000_seed20260924.csv
```

清单SHA-256：

```text
5199869db0d72c04edffe037dee80ecf55dcb49f6c14e429fd8b543ec687d2a7
```

不要重新随机划分后与本次结果混用。随机图块划分不能保证城市/原始大图层面的
空间互斥；如果以后获得城市或原始场景元数据，应改为按区域划分以严格消除空间泄漏。

## 推荐主实验：从官方CDDFuse权重开始

```powershell
Set-Location 'D:\#imageInfusion\CDDFuse'

.\.venv\Scripts\python.exe -u .\train_OS.py `
  --data-root '..\datasets\QXSLAB_SAROPT' `
  --split-manifest '..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json' `
  --init-checkpoint '.\models\CDDFuse_IVF.pth' `
  --output-dir '.\models\qxslab_saropt_from_ivf' `
  --epochs 60 `
  --phase1-epochs 10 `
  --lr 1e-5 `
  --batch-size 8 `
  --patch-size 128 `
  --amp
```

预计训练时间约8--10小时，具体取决于PNG读取速度和验证开销。这一设置不会引入
额外Dongying光学--SAR训练数据，更适合与其他QXS-SAROPT方法进行公平比较。

## 额外迁移实验：从Dongying模型继续微调

该方案收敛更快，但模型已经见过额外的Dongying光学--SAR数据，不能作为只使用
QXS-SAROPT的公平主实验：

```powershell
.\.venv\Scripts\python.exe -u .\train_OS.py `
  --data-root '..\datasets\QXSLAB_SAROPT' `
  --split-manifest '..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json' `
  --init-checkpoint '.\models\optical_sar\CDDFuse_OS_20260923_161630\last.pth' `
  --output-dir '.\models\qxslab_saropt_transfer' `
  --epochs 40 `
  --phase1-epochs 5 `
  --lr 5e-6 `
  --batch-size 8 `
  --patch-size 128 `
  --amp
```

## 测试

正式测试默认读取清单中的全部1,000对：

```powershell
.\.venv\Scripts\python.exe .\test_OS.py `
  --data-root '..\datasets\QXSLAB_SAROPT' `
  --split-manifest '..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json' `
  --checkpoint '.\models\qxslab_saropt_from_ivf\CDDFuse_OS_YYYYMMDD_HHMMSS\best.pth' `
  --metric-profile paper `
  --output-dir '.\test_result\QXSLAB_SAROPT_1000'
```

训练器会分别保存 `best_phase1.pth` 与 `best_phase2.pth`，并将 `best.pth`
指向阶段二当前最佳融合模型，避免两阶段不同损失尺度导致的错误模型选择。
