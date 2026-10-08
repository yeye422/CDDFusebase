# E1：光学/SAR 双 Stem，共享 C0 Encoder 主干

状态：代码、测试、运行脚本已提交；GPU 训练和指标尚未运行。

## 设计

C0：一套 PatchEmbed + Restormer + Base/Detail Heads，同时编码光学和 SAR。

E1：光学 Stem 和 SAR Stem 各自独立；Restormer 和 Base/Detail Heads 共享。BaseFuse、DetailFuse 和 Decoder 均保留 C0 原模型及原参数。

- E1 Stem 均直接复制经过训练的 C0 PatchEmbed 权重，初始输出严格等于 C0。
- 仅训练两套 3×3 Stem，其他 Encoder 层、两个 Fuse 和 Decoder 全部冻结。
- 沿用 C0 Frequency-Minimax 主损失与频率参考值；默认不引入新的约束损失。
- C0 checkpoint、同一数据 split 与训练配置经程序校验。
- 使用相同 train800/val200、5 epoch、seed42，以及 paper+clamp 评价协议。

## 本地运行

在 Windows CMD 项目根目录依次执行：

1. git pull
2. run_e1_dual_stem_pilot.cmd
3. run_e1_dual_stem_validation.cmd
4. run_e1_dual_stem_visuals.cmd

Pilot 首先执行单元测试和真实 C0 checkpoint 等价性检查，不通过不训练。

指标：

test_result\reliability_v5\validation_e1_dual_stem_final\e1_vs_c0_summary.csv

test_result\reliability_v5\validation_e1_dual_stem_final\e1_vs_c0_per_image.csv

同样样本的图像预览：

test_result\reliability_v5\e1_qualitative_val24\

## 判定

- 比较 FMI、QAB/F、NAB/F↓、VIF、MI、SSIM/MS-SSIM 和逐图改善比例；
- 视觉检查建筑/道路边界、SAR 斑点、伪边缘与结构损失；
- E1 如果 NAB/F 降低而 QAB/F 和 VIF 普遍下降，则不能认为已经解决伪影；
- 如果 E1 无收益，先做光学 Stem-only、SAR Stem-only 梯度/贡献消融，再决定是否研究特定的 SAR 抗斑点结构，避免直接叠加第二阶段特征分解。

已有 val200 多次用于开发比较，独立最终结论必须在未参与模型选择的 test split 验证。
