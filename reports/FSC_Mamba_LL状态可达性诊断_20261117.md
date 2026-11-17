# FSC-Mamba：冻结 E2 的 LL 状态可达性诊断

目标是决定 `LL → SSM 初始状态` 是否值得进入 train1024/val64，而非检验新融合模型的画质。

## 诊断范围

- 使用归档 `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`，严格加载训练后的 Encoder、Decoder、LWPD、Base Fusion 和 E2 原 Detail Fusion（keep 分支）。
- FSC 的 LL 投影、状态衰减参数和 SSM 使用固定 `seed42` 新初始化；**没有训练新的 FSC，也没有执行优化器更新**。
- 真实 QXS 验证集的第一张图（默认 211）提供特征，完整图 256×256，扫描网格为 128×128。
- 诊断 `LL` 初态强度 `0, 0.03, 0.1, 0.3, 1.0`；固定同一个 128 长度 LL 摘要，对同一 Detail 序列截取 `32,64,128` 的扫描前缀，分开统计前/中/后段初态响应，排除摘要窗口变化对衰减趋势的干扰。
- 行、列各选 4 条代表性线；正向和逆向分别观察。扫描使用相同的所有参数；正确条件与错配条件仅交换 LL 摘要，不改变 Detail 输入。
- 附加完整二维 FSC SSM 检查：相同 Detail 下，真实 LL、空间滚动的 LL 和**投影后置零**的 LL 输出差。此处探测的是 SSM 输出，不是 Decoder 最终融合图。该检查保留双向扫描。

## Windows CMD

```cmd
cd /d "D:\#imageInfusion\CDDFuse"
git pull origin main
.venv\Scripts\python.exe -m unittest -v test_fsc_state_reachability
.venv\Scripts\python.exe diagnose_fsc_state_reachability.py
```

输出到 `test_result/e2_fsc_state_reachability_v1/scene_211_seed42/`：

- `profiles.csv`：各方向、序列长度及强度下的初态响应 RMS；`tail_over_head` 衡量相对衰减，`delta_to_scan` 衡量相对 SSM 输出幅度。
- `counterfactuals.csv`：正确 LL、错配 LL、零初态的成对 SSM 差异，并记录输入初态幅度，避免把 LL 幅度差误判为语义收益。
- `summary.json`：E2 checkpoint SHA256、脚本 SHA256、固定初始化、完整二维 SSM 差异及解释边界。

历史目录不可覆盖。再次测试请使用 `--output-dir test_result\e2_fsc_state_reachability_v1\another_run`。可选 `--scene`、`--lengths`、`--scales`、`--lines`、`--cpu`。建议在 CUDA 环境下执行；CPU 也能运行，但完整尺寸诊断较慢。

## 判断逻辑

1. `tail_over_head` 明显降低，说明初态对长扫描尾部的影响衰减。**调大初态系数只放大绝对效应，不应改变线性递推下的相对衰减**。
2. 比较 `correct_vs_shuffled_rms` 与 `correct_vs_zero_rms`；这只能证明位置信息影响 SSM，不能证明真实 LL 具有较好的融合价值。
3. 完整二维输出使用行列双向扫描，不能仅根据正向尾段的衰减推断全图最终没有 LL 影响。
4. 即使随机初始化 FSC 的正确 LL 能改变内部输出，也不能证明对 QAB/F、VIF、SSIM 或 NAB/F 有益。只有冻结 E2 的诊断机制明确、与强度/错配对照可分，才进入训练及 val64 配对实验。

**注意**：旧 `run_e2_fsc_mamba.py smoke --full` 使用随机初始化骨干；本脚本改为载入已训练的 E2 共享权重。两者 RMS 绝对值不能直接互相比较。并且原来的更新投影为零初始化，即使 SSM 内部存在差异，初始阶段也不代表最终融合图会变化。
