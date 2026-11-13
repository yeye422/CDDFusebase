# LWPD / PAC-Mamba 实验报告索引

本目录集中保存 LWPD 与 PAC-Mamba 阶段的人工分析和总结文档。训练协议、原始指标、运行元数据、图像输出及实验注册信息仍保留在各自的证据目录中。

## 推荐阅读顺序

| 顺序 | 文档 | 数据划分 | 主要比较对象 | 用途 |
|---:|---|---|---|---|
| 1 | [LWPD E4 结果分析](./LWPD_E4_QXS_VAL1000结果分析_20261028.md) | 主体为 val1000，末尾补充 test1000 | val：Direct Fusion R3；test：B0、retrained Detail-only SPD | 判断 LWPD 第一模块是否有效，并记录其收益与局限 |
| 2 | [PAC-Mamba 结果分析](./PAC_MAMBA_QXS_TEST1000结果分析_20261028.md) | test1000 | LWPD E4 | 判断第二模块能否在 LWPD 基础上减少错误高频传播 |
| 3 | [最终多方法比较](./RESULTS_WITH_LWPD.md) | test1000 | Wavelet、U2Fusion、Laplacian、CDDFuse、B0 + SPD、LWPD、PAC-Mamba | 用于论文最终横向比较和指标排名 |
| 4 | [LWPD 与 PAC-Mamba 可视化比较](./VISUAL_COMPARISON_PAC.md) | test1000 预注册场景 | LWPD E4、PAC-Mamba及其他方法 | 检查边缘连续性、背景颗粒、强散斑和结构相似性变化 |

## 指标口径说明

2026-11-04 补充：[E2 六方法对比与改进方向](./E2_六方法对比与改进方向_20261104.md)。使用已有 E2 test1000，已核对与 E4 的评价口径，并补入最终比较及实验注册表。

进一步分析：[E2 改进方向深入分析](./E2_改进方向深入分析_20261104.md)，覆盖监督一致性、特征门控与输出行为、已有失败实验及分阶段研究优先级。

已开始诊断：[E2 目标诊断首轮结果](./E2_目标诊断首轮结果_20261104.md)。含scene4保存图与单张FP32核验、BGC1实际修正覆盖率、输出梯度方向变化，以及用户执行的固定六图命令。

固定六图已完成：[E2 固定六图诊断结果分析](./E2_固定六图诊断结果分析_20261104.md)。重点是匹配B的SAR偏向、BGC1广泛修正与局部导数变化，以及ROI核查尚未完成的限制。

2026-11-04 新架构方向：[E2 + PAC-Mamba 执行说明](../../E2_PAC_MAMBA_RUN.md)。按用户要求保持原损失，从已有E2初始化，新增PAC残差；已通过单元检查与单样本前反传，完整训练由用户执行CMD。

LWPD后续设计：[跨模态多尺度LL调制方案](../../LWPD_LL_CONTEXT_V1_DESIGN.md)。仅增强低频调制上下文，保留原损失与高频门控；[训练测试命令](../../LWPD_LL_CONTEXT_V1_RUN.md)。完整训练与val结果见下。

LL-context完整结果已检查：[12轮训练与val1000分析](./LWPD_LL_CONTEXT_V1_VAL1000结果分析_20261104.md)。QAB/F与VIF提高，但结构指标下降、NAB/F升高，未通过筛选；不建议推进当前配置test1000。

失败定位：[LL-context失败原因与单样本干预](./LWPD_LL_CONTEXT_V1失败原因与单样本干预_20261104.md)。同checkpoint关闭新分支可改善部分结构，但不能消除NAB/F大部分差距；区分即时调制与联合训练，避免把目标拟合改善等同结构改善。

补救评估：[LL-context修改补救可行性](./LWPD_LL_CONTEXT_V1补救可行性分析_20261104.md)。单图减半、空间均值、单模态开关均未全面超过原E2；冻结旧E2、仅训练光学LL修正属于待验证的新候选，不是现有权重的已成功修复。

配准前提：[QXS-SAROPT 原始描述核查](./QXS_SAROPT_原始描述与配准前提核查_20261104.md)。下一步先量测共同控制点残余错位，再解释噪声与监督冲突。

阅读和引用时必须区分以下口径：

1. `val1000` 与 `test1000` 是固定清单中的两组不同图像，均值不会完全一致。
2. LWPD 的同协议消融基线是 `Direct Fusion R3 epoch 9`，用于判断模块本身的增量贡献。
3. 正式 test1000 补充比较中的 `B0 + retrained Detail-only SPD`，与最终多方法表中的 `B0 + SPD` 不是同一个模型。
4. 最终论文主表应引用 test1000；E1–E4 组件消融应引用 val1000，不能根据 test1000 反向选择组件。
5. 所有 LWPD E4 正式结果均使用同一 checkpoint：`models/lwpd_qxs_full_v1/e4_seed42/epoch_016.pth`。

## 核心结果速查

### LWPD E4 相对 Direct Fusion R3（val1000）

- MI：+3.42%。
- QAB/F：+1.32%。
- NAB/F：-10.38%，越低越好。
- SCD：+3.93%。
- VIF：+2.32%。
- SSIM：-0.85%。
- MS-SSIM：-0.45%。

### LWPD E4 相对 B0（test1000）

- FMI：+1.19%。
- QAB/F：+1.90%。
- NAB/F：-14.05%，越低越好。
- VIF：+1.13%。
- SSIM：-0.83%。
- MS-SSIM：-0.79%。

### PAC-Mamba 相对 LWPD E4（test1000）

- MI：+3.76%。
- QAB/F：+0.57%。
- NAB/F：-24.84%，越低越好。
- VIF：+1.36%。
- SSIM：-0.18%。

## 原始证据入口

| 内容 | 路径 |
|---|---|
| LWPD val1000 原始指标 | [metrics.csv](../../test_result/lwpd_qxs_full_v1/e4_seed42_val1000/metrics.csv) |
| LWPD val1000 运行元数据 | [run_metadata.json](../../test_result/lwpd_qxs_full_v1/e4_seed42_val1000/run_metadata.json) |
| LWPD test1000 原始指标 | [metrics.csv](../../test_result/lwpd_qxs_full_v1/e4_seed42_test1000/metrics.csv) |
| LWPD test1000 运行元数据 | [run_metadata.json](../../test_result/lwpd_qxs_full_v1/e4_seed42_test1000/run_metadata.json) |
| PAC-Mamba test1000 原始指标 | [metrics.csv](../../test_result/pac_mamba_qxs_full_v1/seed42_test1000/metrics.csv) |
| PAC-Mamba test1000 运行元数据 | [run_metadata.json](../../test_result/pac_mamba_qxs_full_v1/seed42_test1000/run_metadata.json) |
| 最终比较汇总表 | [summary_with_lwpd.csv](../../../comparation/results/qxslab_paper_1000_manual/summary_with_lwpd.csv) |
| 全局实验索引 | [EXPERIMENT_INDEX.md](../../EXPERIMENT_INDEX.md) |
| 机器可读实验注册表 | [EXPERIMENT_REGISTRY.json](../../EXPERIMENT_REGISTRY.json) |

## 归档范围

本目录仅保存结论性文档。以下内容不移动：

- `*_PROTOCOL.md`：实现和训练协议；
- `test_result/`：逐图指标、运行元数据和可视化输出；
- `models/`：checkpoint、配置和训练历史；
- `EXPERIMENT_REGISTRY.json` 与 `EXPERIMENT_INDEX.md`：全项目统一实验注册信息。

