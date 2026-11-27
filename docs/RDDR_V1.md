# RDDR 免训练初筛

基础辐射协调与原位细节协同重建模块（RDDR）先验证固定低频协调对融合指标的综合收益。使用冻结的 CF1 第 4 轮 checkpoint，在 QXS-SAROPT 锁定验证清单中按顺序取前 200 景。四组共享一次 CF1 前向和同一个原位细节项。

## 四组设置

令 \(B\) 为 CF1 原有的 5×5 二项式平滑，\(P\) 为解码器预测，\(O\) 和 \(S\) 分别为光学亮度和 SAR 灰度。基线低频项为 \(L_0=B(P)\)，原位细节项为 \(H_0=g_o[O-B(O)]+g_s[S-B(S)]\)，其中 \(g_o,g_s\) 取自 CF1 冻结的增益头。附件未给出 \(T_R\) 的具体构造，本次初筛使用 \(T_R=[B(O)+B(S)]/2\)。它逐像素最小化对双源低频的等权平方偏差，为后续学习空间自适应权重提供一个固定起点。

| 组别 | 基础辐射项 | 细节项 |
| --- | --- | --- |
| R0 · CF1 | \(L_0\) | \(H_0\) |
| R1 · 温和协调 | \(0.75L_0+0.25T_R\) | \(H_0\) |
| R2 · 中等协调 | \(0.5L_0+0.5T_R\) | \(H_0\) |
| R3 · 完全协调 | \(T_R\) | \(H_0\) |

## Windows CMD

```cmd
cd /d "D:\#imageInfusion\CDDFuse"
"D:\anaconda\envs\lightglue_env\python.exe" run_rddr_val200.py smoke --device cuda
"D:\anaconda\envs\lightglue_env\python.exe" run_rddr_val200.py val200 --device cuda
```

单景检查核对 R0 与原 CF1 的裁剪前张量完全一致，归档可用时也核对 14 项指标。正式初筛同样检查第一景。在 checkpoint、验证清单、评价器和输出口径匹配时，复用 CF1 val1000 归档的前 200 景逐图指标。四组统一使用 `reference_v2_candidate_with_paper_fmi` 和 `clamp_then_uint8`。

结果位于 `test_result/e2_rddr_fixed_v1/val200/first200`。各组 `metrics.csv` 保存 14 项逐图指标和共同有限样本上的均值、标准差。`paired_comparison.csv` 给出 R1–R3 相对 R0 的均值差、样本数和 95% 配对 bootstrap 区间，`excluded_scene_ids.json` 记录各指标排除的场景。`run_metadata.json` 保存清单、权重、归档与评价器的来源及 SHA-256。完成后自动登记独立比较组并刷新索引。

## 筛选与后续实验

`screening.json` 使用以下预设阈值判定初筛趋势。配对区间按单项指标报告，用于开发筛选。

- SSIM 和 MS-SSIM 至少一项平均增加 0.003，且区间下界为正。
- CC、SCD、PSNR 至少两项分别达到 0.003、0.003、0.1 dB 的最小增量，且对应区间下界为正。
- MI、FMI、QAB/F、VIF 相对 CF1 的容许回撤依次为 0.030、0.005、0.002、0.003。外部方法的现有对比使用 test1000，外部优势留待同场景、同口径的正式对比确认。
- NAB/F 优先持平或下降，报告其原始差值，负值表示改善。结合 EN、AG、SF、SD 的实际变化判断整体取舍。

初筛出现正向趋势后，再设计可学习的空间协调权重。固定 CF1 第 4 轮的全部参数，只训练新增 RDDR 模块 4 轮，在锁定 val1000 上与同 checkpoint 的 CF1 做完整模块对照。取得明确综合收益后安排从头完整训练及论文实验。初筛缺乏正向趋势时暂停本方案。四轮结果缺乏收益时，按明确的失效现象决定是否修正。
