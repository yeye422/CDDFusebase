# C1：光学/SAR 模态差异适配实验

状态：代码与运行入口已提交；需要在本地 C0 checkpoint 和 QXSLAB 数据上执行验证。

## 核心问题

C0 直接进行 BaseFuse(B_o+B_s) 和 DetailFuse(D_o+D_s)。光学和 SAR 的特征统计、边缘响应与噪声差异可能形成结构冲突、散斑转移或虚假边缘。但这些伪影是否由相加直接造成，需要对照验证。

## 最小改动

C0 Encoder、BaseFuse、DetailFuse、Decoder 冻结。只在 SAR 分支输入相加前加入一个 ModalityDifferenceAdapter：输入为光学与 SAR 的归一化特征及其差分，输出为逐通道小幅度 SAR 特征校正，强度以 SAR 特征 RMS 为单位限制为 alpha=0.10。末层零初始化，初始网络严格等于 C0。不引入二值门控、额外分解支路或噪声伪标签。

分开训练两组对照，不联合修改：

- c1_base_adapter：仅更正 SAR Base，保留 C0 Detail 路径。
- c1_detail_adapter：仅更正 SAR Detail，保留 C0 Base 路径。

继续使用 C0 reference-normalized smooth minimax loss，额外施加 0.01 × normalized_abs_correction 的轻度残差约束。

## 固定训练和评价

相同 C0 epoch_005.pth、相同 C0 config.json、train800/val200、5 epochs、seed42。train 仅更新 C1 adapter，val200 以 paper+clamp 协议计算 FMI、QAB/F、NAB/F、VIF、SSIM、MS-SSIM 与每图 paired delta。对原始 C0、Base-only C1、Detail-only C1 在固定 val24 上生成图像 preview，检查 SAR speckle、伪边缘、错位和有效结构损失。

C1 训练强制校验 C0 checkpoint variant、数据 split manifest 和 loss reference 一致；初始化后两组输出必须与 C0 在 1e-6 内一致。

## Windows CMD 执行

在项目根目录依次：

1. git pull
2. run_c1_modality_adapter_pilots.cmd
3. run_c1_modality_adapter_validation.cmd
4. run_c1_modality_adapter_visuals.cmd

指标：

test_result\reliability_v5\validation_c1_adapters_final\c1_vs_c0_summary.csv

test_result\reliability_v5\validation_c1_adapters_final\c1_vs_c0_per_image.csv

视觉图：

test_result\reliability_v5\c1_qualitative_val24\

## 下一步决策

只有当 Base-only 或 Detail-only 能在减轻伪影的同时基本保住 FMI/QAB/F/VIF，且 NAB/F 不恶化，才继续 C1 定位和后续 C2 分解；不能只凭单一指标提升认定成功。若两者均无效，应先对 C0 做路径归因，不扩大模型容量。

val200 已用于既往阶段，不能当作完全未使用的独立最终测试。后续正式结果需要额外 test split 验证。
