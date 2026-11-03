# PAC-Mamba 第二模块实施协议

## 目标

在保留 LWPD E4 已获得的 MI、FMI、QAB/F、AG、SF 和 VIF 优势的前提下，利用跨模态共同边界证据限制补充更新，重点改善 NAB/F、CC、SSIM 和 MS-SSIM。

## 当前实现

1. 使用与 LWPD 相同的固定多尺度 Log-Gabor/Riesz 响应，同时计算方向相位对称性 PS 与相位反对称性 PA。
2. PS 继续用于 LWPD 的 SAR 受控高频恢复；光学/SAR PA 用于 PAC 双向 3×3 局部匹配。
3. PA 匹配结果与学习型可靠度相乘，得到交互门控，不设置强制传递下限。
4. 光学 Detail、SAR Detail 和二者绝对差在半分辨率联合编码。
5. 使用行、列双轴正反向的纯 PyTorch Mamba 选择性状态空间扫描；其离散步长、状态输入 B 和状态输出 C 均随位置变化。
6. 已训练的 LWPD Detail 融合器作为基础保留路径；PAC 更新以零初始化残差加入，因此初始输出与 LWPD E4 数值一致。
7. PA 门控只作用一次，避免重复门控造成更新量平方级衰减。

## 参数与训练阶段

| 阶段 | epoch | 可训练范围 |
|---|---:|---|
| Warmup | 2 | PAC 主学习率；LWPD、基础融合、解码器小学习率；编码器冻结 |
| Local | 3 | 上述全部 + 编码器后半部分 |
| Joint | 7 | 全模型分层小学习率联合优化 |

旧模块从 warmup 开始就以小学习率更新，避免 PAC 被固定上下游忽略。仅共享编码器在最初 2 epoch 冻结。

## 必须监控

- `phase_match_mean`：固定 PA 共同边界覆盖率；
- `reliability_mean`：学习型可靠度是否整体关闭或饱和；
- `interaction_mean`：实际交互门控强度；
- `relative_update_rms`：PAC 更新相对基础保留特征的幅度；
- `update_scale`：全局残差尺度；
- 各参数组梯度范数；
- LWPD 门控均值是否因联合训练发生异常漂移。

失败信号包括：交互门控接近常数、更新长期接近零、更新幅度无约束上升、NAB/F 上升、QAB/F 或 VIF 明显回落。

## 完整训练命令

```cmd
cd /d D:\#imageInfusion\CDDFuse

.venv\Scripts\python.exe train_pac_mamba.py ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --train-limit 0 ^
  --val-limit 0 ^
  --batch-size 2 ^
  --accumulation-steps 4 ^
  --warmup-epochs 2 ^
  --local-epochs 3 ^
  --joint-epochs 7 ^
  --init-checkpoint models\lwpd_qxs_full_v1\e4_seed42\epoch_016.pth ^
  --output-dir models\pac_mamba_qxs_full_v1 ^
  --run-name seed42 ^
  --seed 42
```

正式训练由实验人员确认后执行，不在实现阶段自动启动。

