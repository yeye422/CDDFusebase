# PAC-Mamba 的 PA 引导归因验证

## 旧结果回答的问题

旧 PAC 续训已有同预算的无 PAC 对照，但无 PAC 组将更新投影冻结为零。它比较整条附加路径与没有该路径的续训，未分离 PA 和普通特征驱动的跨模态更新。同组旧口径下，旧 PAC 相对原 E2 的 QAB/F 从 0.7324 升到 0.7356，NAB/F 从 0.0646 升到 0.0782。相对无 PAC 续训，旧 PAC 的 NAB/F 较低，FMI 也较低。这些是既有整模块结果。 [历史同组结果](../EXPERIMENT_INDEX.md#qxs-val1000-direct-lwpd-v1) · [等预算无 PAC 实现](../run_e2_matched_no_pac.py)

旧实现将原始 E2 的空间 Detail 按 2×2 区域平均后送入 SSM。E2 的 Detail 由一级 Haar 的三个高频子带合成，内部对齐的 2×2 块均值为零。冻结旧光学 Base PAC 在 val 首图 4 的复核结果如下。

| 读出 | 光学 | SAR |
|---|---:|---:|
| 原 Detail RMS | 0.0147875 | 0.0854663 |
| 旧 2×2 平均池化后 RMS | 1.90e−10 | 1.38e−9 |
| 保留三个有符号 Haar 子带后的 RMS | 0.0170751 | 0.0986879 |

把旧 SSM 之前的两路池化输入一起置零，未乘门控的更新相对 RMS 差仅 \(1.60\times10^{-7}\)。旧输出仍受真实两源 Detail 影响，因为保留支和可靠度估计仍读取它们。 [单图原始数据](../test_result/pac_input_cancellation_v1/single_scene_20261117/summary.json) · [旧 PAC 池化](../pac_mamba.py#L226)

## 四组对照

所有新组使用同一个光学 Base E2、同一个保留支与双轴跨源 SSM。在半分辨率网格直接输入两路有符号 LH、HL、HH 和它们的绝对差。共享特征头产生局部基础交互系数。相位组在此系数上乘一个空间因子，再将同形状更新叠加到保留支。最后投影零初始化。E2 分解保持原有的方向门控。

| 组 | 更新强度的条件 |
|---|---|
| feature | 共享特征头产生的纯特征条件 |
| pa | 特征条件 × 同方向、局部双向 PA 匹配因子 |
| pc | 特征条件 × 同样位置与范围内的 PC 匹配因子 |
| pa_shifted | 特征条件 × 在半分辨率网格循环平移的 PA 因子 |

PC 使用同一 Log-Gabor/Riesz 滤波组，以跨尺度平均相位、相位偏差和频率扩展权重计算。PA 与 PC 共用局部匹配、尺度和固定阈值设置。相位因子逐图居中，均值为 1，范围为 [0.5, 1.5]。平移发生在匹配及降采样之后，保持直方图与局部结构，改变其与图像的位置对应。乘积的实际更新幅值仍可能因与特征的相关性而不同，验证保存各图的更新 RMS 比。MSPA-OS 提供的是配准任务中使用 PA 的证据，像素融合收益由本实验验证。 [模型实现](../e2_phase_guide.py) · [MSPA-OS](https://doi.org/10.1109/LGRS.2025.3627793) · [PC 方法依据](https://www.peterkovesi.com/papers/phasecorners.pdf)

四变体共享原 train1024/val64、seed42、4 轮单模态重建和 12 轮原损失融合。从头初始化，并复用归档 E2 对照与评价结果。固定真实场景 211 的 64×64 裁块已通过四组前向及反传检查。四组初始权重相同，零更新时输出逐值等于同模型的 E2 保留路径。临时打开更新投影后，PA 组分别移除任一源 Detail 都会影响更新。本次没有启动训练或批量验证。

首要对照是 PA 对纯特征条件。用现行 14 指标协议，记录各指标共同有限样本及逐图差值。首轮只有当 NAB/F 未恶化且 QAB/F 或 FMI 有配对正收益，才继续追加 PC 与平移 PA 归因。稳定性用配对区间和改善图像比例判断。单个种子的 val64 结果用于筛选。

## Windows CMD

先跑两组。评价结果保存逐图指标、共同样本数、灰度图及配对区间。各训练与评测结果完成后自动登记。

~~~cmd
cd /d "D:\#imageInfusion\CDDFuse"
.venv\Scripts\python.exe -u run_e2_phase_guide.py train --mode feature
.venv\Scripts\python.exe -u run_e2_phase_guide.py train --mode pa
.venv\Scripts\python.exe -u run_e2_phase_guide.py val-primary
~~~

首轮通过后追加以下两组。完整比较复用先前已经计算的纯特征组和 PA 组输出。

~~~cmd
.venv\Scripts\python.exe -u run_e2_phase_guide.py train --mode pc
.venv\Scripts\python.exe -u run_e2_phase_guide.py train --mode pa_shifted
.venv\Scripts\python.exe -u run_e2_phase_guide.py val-full
~~~

单样本工程复核使用 smoke 或 audit-legacy 子命令。
