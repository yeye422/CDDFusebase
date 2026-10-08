# E1 解冻 Encoder：训练与验证

## 目的

E1-StemOnly 已在 val200 上出现 Qabf/FMI/VIF 退化。验证原因是否为更新 Stem 后仍然冻结共享 Encoder 主干，导致特征分布不兼容。

这次从**同一 C0 checkpoint**重新开始 E1-Unfrozen，不从旧的 E1 stem-only epoch5 checkpoint 继续训练，以避免不同预训练历史导致对照失真。

## 固定 E1-Unfrozen 训练

- 训练模式：e1_dual_stem + encoder_only。
- 训练模块：optical_stem、sar_stem、共享 Restormer、Base/Detail 编码头。
- 冻结模块：C0 BaseFuse、DetailFuse、Decoder；保留未参与前向传播的原 C0 patch_embed 作为冻结锚点。
- 学习率：两个 Stem 1e-4，共享主干 1e-5。
- 原 C0 reference-normalized frequency minimax loss，train800 / val200 / epoch5 / seed42。
- 两套 Stem 从同一个 C0 patch_embed 初始化。

严格学习率对照：C0-EncoderOnly，解冻原 PatchEmbed（1e-4）以及原 Encoder 其余层（1e-5），冻结融合与 Decoder，使用相同训练/验证集、目标函数与训练轮数。

探索性比较：E1-End2End（Stem 1e-4，Encoder 主干 1e-5，Fuse/Decoder 1e-6）以及 C0 全网络微调（全部 1e-5）。后者融合端学习率不同，不能作为严格同超参数架构归因结论；若需要严谨对照，应单独匹配每一组参数学习率。

## Windows CMD 命令

在项目根目录运行：

    git pull
    run_e1_unfrozen_encoder_pilot.cmd
    run_c0_encoder_unfrozen_control.cmd
    run_e1_unfrozen_validation.cmd

可选：

    run_e1_end2end_pilot.cmd
    run_c0_full_finetune_control.cmd
    run_e1_unfrozen_validation.cmd

验证脚本会重新评估原 C0、E1 EncoderOnly，并自动加入已有 checkpoint 对应的可选对照。

## 输出

    test_result\reliability_v5\validation_e1_unfrozen_final\e1_unfrozen_comparison_summary.csv
    test_result\reliability_v5\validation_e1_unfrozen_final\e1_unfrozen_comparison_per_image.csv

指标：MI、FMI、Qabf、Nabf（越低越好）、SSIM、MS-SSIM、VIF，逐图配对比较。

## 判定

- 先看 E1-EncoderOnly 是否恢复之前 E1-StemOnly 下降的 Qabf、FMI、VIF。
- 再对照 C0-EncoderOnly，确认收益是否超出普通 Encoder 微调。
- Nabf 单独下降而 Qabf/VIF 系统性下降，不能视为成功。
- val200 已参与模型选择，仅用于开发；完整论文结论需要最终未见过的 test 数据确认。
