# CDDFuse光学-SAR训练说明

## 默认训练协议

- 数据：`../datasets/Dongying/GF2`与`GF3`，按同名JPEG配对；
- 测试集：数值编号1--150，训练过程完全不读取；
- 验证集：编号151--450；
- 训练集：编号451--7831；
- 光学输入：GF-2 BGR转OpenCV YCrCb后取Y；
- SAR输入：GF-3单通道灰度；
- 裁剪：每次随机取128×128配对块；
- 增强：随机水平/垂直翻转与90度旋转；
- 无效区：默认将SAR值为0的像素标记为无效，优先选择有效比例不低于90%的块；
- 初始化：官方 `models/CDDFuse_IVF.pth` 红外--可见光权重；
- 阶段一：10轮重建/分解域适配；
- 阶段二：50轮融合训练；
- 默认学习率：`1e-5`，批量大小8。

测试集1--150与此前所有零样本及对比实验完全一致，因此绝不能把这些编号加入
训练或验证，否则对比结果会产生数据泄漏。

## 推荐启动命令

在PowerShell或cmd中运行：

```bat
cd /d D:\#imageInfusion\CDDFuse
.venv\Scripts\python.exe -u train_OS.py --amp
```

输出目录为 `models/optical_sar/CDDFuse_OS_日期_时间/`：

- `config.json`：本次参数与数据量；
- `history.csv`：每轮训练/验证损失；
- `last.pth`：最近一轮，可用于断点续训；
- `best.pth`：验证损失最低的模型；
- `epoch_XXX.pth`：每5轮保存一次。

## 显存不足

优先减小批量大小：

```bat
.venv\Scripts\python.exe -u train_OS.py --amp --batch-size 4
```

仍然不足时使用：

```bat
.venv\Scripts\python.exe -u train_OS.py --amp --batch-size 2 --patch-size 96
```

## 断点续训

```bat
.venv\Scripts\python.exe -u train_OS.py --amp --resume models\optical_sar\CDDFuse_OS_YYYYMMDD_HHMMSS\last.pth
```

续训时模型、优化器、学习率调度器、AMP缩放器和起始轮次都会恢复。应保持数据
划分和主要超参数与原训练一致。

## 从零训练

从零训练建议恢复原论文代码的120轮/40轮阶段划分和学习率：

```bat
.venv\Scripts\python.exe -u train_OS.py --amp --init-checkpoint none --epochs 120 --phase1-epochs 40 --lr 1e-4
```

从零训练通常比预训练权重域适配需要更长时间，且不一定优于推荐方案。

## 训练完成后测试

将路径替换为实际生成的 `best.pth`：

```bat
.venv\Scripts\python.exe test_OS.py --checkpoint models\optical_sar\CDDFuse_OS_YYYYMMDD_HHMMSS\best.pth --metric-profile paper --limit 150 --output-dir test_result\Dongying_OS_trained_150
```

该命令在保留的1--150测试集上生成融合图，并输出论文口径14项指标。

## 快速检查命令

下面只运行两个小批次，用于确认环境、数据和反向传播可用，不用于产生正式模型：

```bat
.venv\Scripts\python.exe -u train_OS.py --amp --epochs 2 --phase1-epochs 1 --patch-size 32 --batch-size 1 --train-limit 4 --val-limit 2 --max-train-batches 1 --max-val-batches 1 --output-dir tmp\train_os_check
```

## 与原始训练脚本的区别

`train.py`和`dataprocessing.py`保持不变，仍用于复现MSRS红外--可见光训练。
`train_OS.py`直接按需读取JPEG，不生成数GB的HDF5文件，并加入：

- 固定且无泄漏的数据划分；
- SAR无效像素掩膜；
- 无效区域回退到光学亮度目标；
- 验证集与最佳模型保存；
- 断点续训；
- 混合精度训练；
- 新旧检查点键名兼容。

