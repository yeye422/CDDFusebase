# 实验数据索引

更新时间：2026-11-03  
Schema：v1

## 使用规则

- 只有 `comparable_group` 完全相同的实验才能直接进入同一张定量对比表。
- 新实验必须先写入 `EXPERIMENT_REGISTRY.json`，再执行校验和渲染。
- `test`、`val`、offset、样本数、checkpoint、评价器或输出策略任一不同，都必须使用不同分组。
- Markdown 仅用于阅读；JSON 注册表和原始结果文件才是事实来源。

校验命令：

```cmd
cd /d D:\#imageInfusion\CDDFuse
.venv\Scripts\python.exe maintain_experiment_index.py --check --render
```

## MFFA-V1-PHASE1-QXS-VAL1000-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| MFFA-V1-PHASE1-QXS-VAL1000 | MFFA v1 Phase-1 symmetric frequency autoencoder | development_passed | val@0 | 1000 | optical_mae=0.0000, optical_ssim=1.0000, sar_mae=0.0004, sar_psnr=59.7408, sar_ssim=1.0000, optical_high=0.0127, sar_high=0.0081 |

来源：

- `MFFA-V1-PHASE1-QXS-VAL1000`：`test_result/mffa_v1/phase1_dtypefix_seed42_val1000/summary.json`；checkpoint `models/mffa_v1/phase1_dtypefix_seed42/best.pth`。

## QXS-E2-GENERAL-LOSS-SMOKE-s-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-GENERAL-LOSS-V1-S-SEED42-SMOKE | E2 loss s: grad=10.0, structure=0.5 | engineering_smoke | val@0 | 1 | loss=12.5559, structure=0.8389 |

来源：

- `E2-GENERAL-LOSS-V1-S-SEED42-SMOKE`：`models/e2_general_loss_smoke_v1/s_seed42/dry_run_diagnostics.json`；checkpoint `models/direct_fusion_qxs/qxs_direct_seed42_v1/r3/lowest_val_loss_diagnostic.pth`。

## QXS-E2-GENERAL-LOSS-TRAIN8000-b-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-GENERAL-LOSS-V1-B-SEED42-TRAIN8000 | E2 loss b: grad=10.0, structure=0.0 | training_complete | train@0 | 8000 | train_loss=0.7420, val_loss=0.7462 |

来源：

- `E2-GENERAL-LOSS-V1-B-SEED42-TRAIN8000`：`models/e2_general_loss_v1/b_seed42/history.csv`；checkpoint `models/e2_general_loss_v1/b_seed42/epoch_016.pth`。

## QXS-E2-GENERAL-LOSS-TRAIN8000-s-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-GENERAL-LOSS-V1-S-SEED42-TRAIN8000 | E2 loss s: grad=10.0, structure=0.5 | training_complete | train@0 | 8000 | train_loss=0.9798, val_loss=0.9652 |

来源：

- `E2-GENERAL-LOSS-V1-S-SEED42-TRAIN8000`：`models/e2_general_loss_v1/s_seed42/history.csv`；checkpoint `models/e2_general_loss_v1/s_seed42/epoch_016.pth`。

## QXS-PDCF-P1-PREDICTABILITY-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| PDCF-P1-PREDICTABILITY-QXS-TRAIN-SEED42 | PDCF P1 frozen-C0 bidirectional cross-modal predictor | development_positive | train+val@0 | 9000 | predictor_loss=0.5217, linear_loss=0.5476, o2s_base_nrmse=0.8140, o2s_detail_nrmse=0.9316, s2o_base_nrmse=0.6808, s2o_detail_nrmse=0.5962 |

来源：

- `PDCF-P1-PREDICTABILITY-QXS-TRAIN-SEED42`：`models/pdcf_p1/predictability_seed42/history.csv`；checkpoint `models/pdcf_p1/predictability_seed42/best.pth`。

## QXS-PDCF-P2-PILOT1000-INVALID-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| PDCF-P2-PREDICTABLE-PILOT1000-SEED42-NAN-INVALID | PDCF P2 predictable-residual single-path pilot before zero-gradient fix | invalid | train+val@0 | 1200 | — | — | — | — | — | — |

来源：

- `PDCF-P2-PREDICTABLE-PILOT1000-SEED42-NAN-INVALID`：`models/pdcf_p2/predictable_residual_seed42_pilot1000/history.csv`；checkpoint `models/pdcf_p2/predictable_residual_seed42_pilot1000/last.pth`。

## QXS-TEST1000-PAPER-CLAMP-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0-QXS-TEST1000-PAPER-CLAMP | B0 original CDDFuse | formal_test | test@0 | 1000 | 0.6946 | 0.7138 | 0.1145 | 0.5312 | 0.5737 | 0.3948 |
| B1V5-QXS-TEST1000-PAPER-CLAMP | B1-v5 noise-suppression baseline | formal_test | test@0 | 1000 | 0.2976 | 0.3825 | 0.0522 | 0.5012 | 0.5811 | 0.2359 |
| C0-MINIMAX-QXS-TEST1000-PAPER-CLAMP | C0 frequency-minimax epoch 5 | formal_test | test@0 | 1000 | 0.4358 | 0.5905 | 0.0585 | 0.5145 | 0.5931 | 0.2503 |
| SPD-E1IMAGE-QXS-TEST1000-PAPER-CLAMP | E1 SPD image preprocessing | formal_test | test@0 | 1000 | 0.2532 | 0.3104 | 0.0454 | 0.4791 | 0.5894 | 0.2077 |
| SPD-E2FEATURE-QXS-TEST1000-PAPER-CLAMP | E2 SPD feature purification | formal_test | test@0 | 1000 | 0.4307 | 0.6007 | 0.0556 | 0.5204 | 0.5985 | 0.2516 |
| SPD-E2ADAPT-QXS-TEST1000-PAPER-CLAMP | E2-Adapt full 8000/1000 | formal_test_negative | test@0 | 1000 | 0.4112 | 0.5634 | 0.0633 | 0.5101 | 0.5925 | 0.2409 |
| SPD-E2FEATURE-ON-B0-QXS-TEST1000-PAPER-CLAMP | B0 + E2 SPD feature purification | formal_test | test@0 | 1000 | 0.6553 | 0.7015 | 0.0813 | 0.5355 | 0.5810 | 0.3554 |
| SPD-DETAILONLY-ON-B0-QXS-TEST1000-PAPER-CLAMP | B0 + SPD Detail-only (alpha_B=0, alpha_D=1) | formal_test_selected | test@0 | 1000 | 0.6960 | 0.7141 | 0.0933 | 0.5326 | 0.5729 | 0.3958 |
| SPD-DETAILONLY-RETRAINED-ON-B0-QXS-TEST1000-PAPER-CLAMP | B0 + retrained SPD Detail-only | formal_test_main | test@0 | 1000 | 0.6954 | 0.7137 | 0.0946 | 0.5323 | 0.5730 | 0.3953 |
| LWPD-E4-QXS-TEST1000-SEED42 | Complete LWPD E4: directional Haar gating, controlled SAR recovery and phase symmetry | formal_test_main | test@0 | 1000 | 0.7028 | 0.7274 | 0.0984 | 0.5268 | 0.5692 | 0.3992 |
| PAC-MAMBA-QXS-TEST1000-SEED42 | LWPD E4 + phase-antisymmetry-conditioned bi-axial selective SSM | formal_test_main | test@0 | 1000 | 0.7074 | 0.7315 | 0.0740 | 0.5259 | 0.5700 | 0.4047 |

来源：

- `B0-QXS-TEST1000-PAPER-CLAMP`：`test_result/reliability_v4/B0_clamp/metrics.csv`；checkpoint `models/qxslab_saropt_from_ivf/CDDFuse_OS_20260924_012025/best.pth`。
- `B1V5-QXS-TEST1000-PAPER-CLAMP`：`test_result/reliability_v4/B1_v5_seed42/metrics.csv`；checkpoint `models/reliability_v4/B1_v5_seed42/best.pth`。
- `C0-MINIMAX-QXS-TEST1000-PAPER-CLAMP`：`test_result/reliability_v5/frequency_minimax_seed42/metrics.csv`；checkpoint `models/reliability_v5/frequency_minimax_seed42/epoch_005.pth`。
- `SPD-E1IMAGE-QXS-TEST1000-PAPER-CLAMP`：`test_result/v21_softmask/spd_nomask_fusion_test1000_c0full/summary.json`；checkpoint `models/v21_softmask/spd_nomask_seed42/best.pth`。
- `SPD-E2FEATURE-QXS-TEST1000-PAPER-CLAMP`：`test_result/v21_softmask/spd_nomask_fusion_test1000_c0full/summary.json`；checkpoint `models/v21_softmask/spd_nomask_seed42/best.pth`。
- `SPD-E2ADAPT-QXS-TEST1000-PAPER-CLAMP`：`test_result/v21_softmask/e2_adapt_full_seed42_test1000/summary.json`；checkpoint `models/v21_softmask/spd_nomask_seed42/best.pth`。
- `SPD-E2FEATURE-ON-B0-QXS-TEST1000-PAPER-CLAMP`：`test_result/v21_softmask/spd_nomask_on_b0_test1000/summary.json`；checkpoint `models/v21_softmask/spd_nomask_seed42/best.pth`。
- `SPD-DETAILONLY-ON-B0-QXS-TEST1000-PAPER-CLAMP`：`test_result/v21_softmask/spd_detail_only_on_b0_test1000/summary.json`；checkpoint `models/v21_softmask/spd_nomask_seed42/best.pth`。
- `SPD-DETAILONLY-RETRAINED-ON-B0-QXS-TEST1000-PAPER-CLAMP`：`test_result/v21_softmask/spd_detail_only_retrained_on_b0_test1000/summary.json`；checkpoint `models/v21_softmask/spd_detail_only_seed42/best.pth`。
- `LWPD-E4-QXS-TEST1000-SEED42`：`test_result/lwpd_qxs_full_v1/e4_seed42_test1000/metrics.csv`；checkpoint `models/lwpd_qxs_full_v1/e4_seed42/epoch_016.pth`。
- `PAC-MAMBA-QXS-TEST1000-SEED42`：`test_result/pac_mamba_qxs_full_v1/seed42_test1000/metrics.csv`；checkpoint `models/pac_mamba_qxs_full_v1/seed42/epoch_012.pth`。

## QXS-VAL-FIRST200-PDCF-P1-INTERVENTION-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| PDCF-P1-RESIDUAL-AUDIT-QXS-VAL200 | PDCF P1 prediction-residual intervention audit | development_positive_conditional | val@0 | 200 | c0_FMI=0.4805, c0_Qabf=0.6230, c0_Nabf=0.0489, c0_MS_SSIM=0.5777, c0_VIF=0.2410, original_output_deviation_rms=0.0000, zero_sar_delta_Qabf=-0.5390, spatial_shuffle_delta_Qabf=-0.5244, batch_swap_delta_Qabf=-0.5275, rms_noise_delta_Qabf=-0.5342 |

来源：

- `PDCF-P1-RESIDUAL-AUDIT-QXS-VAL200`：`test_result/pdcf_p1/predictability_seed42_val200/summary.json`；checkpoint `models/pdcf_p1/predictability_seed42/best.pth`。

## QXS-VAL-FIRST200-PDCF-P2-MATCHED-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| PDCF-P2-PREDICTABLE-SINGLEPATH-QXS-VAL200 | PDCF P2 Predictable/Residual single-path pilot | development_negative | val@0 | 200 | 0.3496 | 0.4091 | 0.0748 | 0.4661 | 0.5863 | 0.1719 |
| PDCF-P2-RAW-EVIDENCE-SINGLEPATH-QXS-VAL200 | PDCF P2 capacity-matched Raw Evidence single-path control | development_negative | val@0 | 200 | 0.3377 | 0.3700 | 0.0827 | 0.4579 | 0.5833 | 0.1652 |

来源：

- `PDCF-P2-PREDICTABLE-SINGLEPATH-QXS-VAL200`：`test_result/pdcf_p2/matched_pilots_seed42_val200_gradfix/summary.json`；checkpoint `models/pdcf_p2/predictable_residual_seed42_pilot1000_gradfix/best.pth`。
- `PDCF-P2-RAW-EVIDENCE-SINGLEPATH-QXS-VAL200`：`test_result/pdcf_p2/matched_pilots_seed42_val200_gradfix/summary.json`；checkpoint `models/pdcf_p2/raw_evidence_seed42_pilot1000_gradfix/best.pth`。

## QXS-VAL-FIRST200-PDCF-P2-SCALE-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| PDCF-P2-SCALE-DIAGNOSTIC-QXS-VAL200 | PDCF P2 no-training correction-scale diagnostic | development_negative_terminal | val@0 | 200 | FMI=0.4693, Qabf=0.6130, Nabf=0.0496, SSIM=0.5055, MS_SSIM=0.5789, VIF=0.2368 |

来源：

- `PDCF-P2-SCALE-DIAGNOSTIC-QXS-VAL200`：`test_result/pdcf_p2/scale_diagnostic_seed42_val200/summary.json`；checkpoint `models/pdcf_p2/predictable_residual_seed42_pilot1000_gradfix/best.pth`。

## QXS-VAL-FIRST200-PDCF-P2-SIGNED-SCALE-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| PDCF-P2-SIGNED-C0-QXS-VAL200 | C0 identity in PDCF signed-scale diagnostic | diagnostic_baseline | val@0 | 200 | 0.4805 | 0.6231 | 0.0489 | 0.5063 | 0.5777 | 0.2410 |
| PDCF-P2-SIGNED-PREDICTABLE-NEG010-QXS-VAL200 | PDCF P2 predictable signed correction alpha=-0.1 | development_positive_conditional | val@0 | 200 | 0.5040 | 0.6420 | 0.0480 | 0.5074 | 0.5758 | 0.2496 |
| PDCF-P2-SIGNED-PREDICTABLE-NEG005-QXS-VAL200 | PDCF P2 predictable signed correction alpha=-0.05 | development_positive_conditional | val@0 | 200 | 0.4922 | 0.6327 | 0.0484 | 0.5069 | 0.5767 | 0.2453 |
| PDCF-P2-SIGNED-RAW-NEG010-QXS-VAL200 | PDCF P2 raw signed correction alpha=-0.1 | development_positive_conditional | val@0 | 200 | 0.5033 | 0.6450 | 0.0476 | 0.5076 | 0.5761 | 0.2504 |
| PDCF-P2-SIGNED-RAW-NEG005-QXS-VAL200 | PDCF P2 raw signed correction alpha=-0.05 | development_positive_conditional | val@0 | 200 | 0.4924 | 0.6343 | 0.0482 | 0.5070 | 0.5768 | 0.2457 |

来源：

- `PDCF-P2-SIGNED-C0-QXS-VAL200`：`test_result/pdcf_p2/signed_scale_diagnostic_seed42_val200/summary.json`；checkpoint `models/reliability_v5/frequency_minimax_seed42/best.pth`。
- `PDCF-P2-SIGNED-PREDICTABLE-NEG010-QXS-VAL200`：`test_result/pdcf_p2/signed_scale_diagnostic_seed42_val200/summary.json`；checkpoint `models/pdcf_p2/predictable_residual_seed42_pilot1000_gradfix/best.pth`。
- `PDCF-P2-SIGNED-PREDICTABLE-NEG005-QXS-VAL200`：`test_result/pdcf_p2/signed_scale_diagnostic_seed42_val200/summary.json`；checkpoint `models/pdcf_p2/predictable_residual_seed42_pilot1000_gradfix/best.pth`。
- `PDCF-P2-SIGNED-RAW-NEG010-QXS-VAL200`：`test_result/pdcf_p2/signed_scale_diagnostic_seed42_val200/summary.json`；checkpoint `models/pdcf_p2/raw_evidence_seed42_pilot1000_gradfix/best.pth`。
- `PDCF-P2-SIGNED-RAW-NEG005-QXS-VAL200`：`test_result/pdcf_p2/signed_scale_diagnostic_seed42_val200/summary.json`；checkpoint `models/pdcf_p2/raw_evidence_seed42_pilot1000_gradfix/best.pth`。

## QXS-VAL-FIRST200-SPD-COMPARISON-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| SPD-C0RAW-QXS-VAL200 | C0 raw path in SPD comparison | development | val@0 | 200 | 0.6571 | 0.7387 | 0.0368 | 0.5303 | 0.5775 | 0.3269 |
| SPD-E1IMAGE-QXS-VAL200 | E1 SPD image preprocessing | development | val@0 | 200 | 0.2402 | 0.2610 | 0.0087 | 0.4933 | 0.6034 | 0.1766 |
| SPD-E2FEATURE-QXS-VAL200 | E2 SPD feature purification | development | val@0 | 200 | 0.6513 | 0.7389 | 0.0249 | 0.5333 | 0.5862 | 0.3271 |

来源：

- `SPD-C0RAW-QXS-VAL200`：`test_result/v21_softmask/spd_nomask_fusion_val200/summary.json`；checkpoint `models/reliability_v5/frequency_minimax_baseline_pilot/epoch_005.pth`。
- `SPD-E1IMAGE-QXS-VAL200`：`test_result/v21_softmask/spd_nomask_fusion_val200/summary.json`；checkpoint `models/v21_softmask/spd_nomask_seed42/best.pth`。
- `SPD-E2FEATURE-QXS-VAL200`：`test_result/v21_softmask/spd_nomask_fusion_val200/summary.json`；checkpoint `models/v21_softmask/spd_nomask_seed42/best.pth`。

## QXS-VAL-OFFSET200-LIMIT800-PAPER-CLAMP-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0-QXS-VAL800-PAPER-CLAMP | B0 original CDDFuse | confirmation | val@200 | 800 | 0.6621 | 0.7024 | 0.1163 | 0.5332 | 0.5760 | 0.3785 |
| B1V5-QXS-VAL800-PAPER-CLAMP | B1-v5 noise-suppression baseline | confirmation | val@200 | 800 | 0.3129 | 0.4015 | 0.0554 | 0.5049 | 0.5761 | 0.2492 |

来源：

- `B0-QXS-VAL800-PAPER-CLAMP`：`test_result/reliability_v5/confirmation_val800/B0/metrics.csv`；checkpoint `models/qxslab_saropt_from_ivf/CDDFuse_OS_20260924_012025/best.pth`。
- `B1V5-QXS-VAL800-PAPER-CLAMP`：`test_result/reliability_v5/confirmation_val800/B1_v5/metrics.csv`；checkpoint `models/reliability_v4/B1_v5_seed42/best.pth`。

## QXS-VAL-SCENE4-PRL-INTERVENTION-20261101-E2-FP32

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| LWPD-PRL-DIAG-SCENE4-20261101-E2-FP32 | e2_fp32 | diagnostic_single_scene | val@0 | 1 | 1.1271 | 0.7525 | 0.0511 | 0.5249 | 0.5421 | 0.4762 |

来源：

- `LWPD-PRL-DIAG-SCENE4-20261101-E2-FP32`：`test_result/lwpd_prl_diagnosis_v1/scene4_20261101/e2_fp32/metrics.csv`；checkpoint `models/lwpd_qxs_ablation_v1/e2_seed42/epoch_016.pth`。

## QXS-VAL-SCENE4-PRL-INTERVENTION-20261101-PRL-AMP

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| LWPD-PRL-DIAG-SCENE4-20261101-PRL-AMP | prl_amp | diagnostic_single_scene | val@0 | 1 | 1.1286 | 0.7524 | 0.0897 | 0.5243 | 0.5406 | 0.4765 |
| LWPD-PRL-DIAG-SCENE4-20261101-PRL-HAAR-AMP | prl_haar_amp | diagnostic_single_scene | val@0 | 1 | 1.1286 | 0.7523 | 0.0899 | 0.5243 | 0.5406 | 0.4765 |

来源：

- `LWPD-PRL-DIAG-SCENE4-20261101-PRL-AMP`：`test_result/lwpd_prl_diagnosis_v1/scene4_20261101/prl_amp/metrics.csv`；checkpoint `models/lwpd_qxs_prl_v1/e2_prl_seed42/epoch_016.pth`。
- `LWPD-PRL-DIAG-SCENE4-20261101-PRL-HAAR-AMP`：`test_result/lwpd_prl_diagnosis_v1/scene4_20261101/prl_haar_amp/metrics.csv`；checkpoint `models/lwpd_qxs_prl_v1/e2_prl_seed42/epoch_016.pth`。

## QXS-VAL-SCENE4-PRL-INTERVENTION-20261101-PRL-FP32

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| LWPD-PRL-DIAG-SCENE4-20261101-PRL-FP32 | prl_fp32 | diagnostic_single_scene | val@0 | 1 | 1.1301 | 0.7524 | 0.0903 | 0.5244 | 0.5406 | 0.4767 |
| LWPD-PRL-DIAG-SCENE4-20261101-PRL-HAAR-FP32 | prl_haar_fp32 | diagnostic_single_scene | val@0 | 1 | 1.1307 | 0.7524 | 0.0903 | 0.5244 | 0.5406 | 0.4766 |

来源：

- `LWPD-PRL-DIAG-SCENE4-20261101-PRL-FP32`：`test_result/lwpd_prl_diagnosis_v1/scene4_20261101/prl_fp32/metrics.csv`；checkpoint `models/lwpd_qxs_prl_v1/e2_prl_seed42/epoch_016.pth`。
- `LWPD-PRL-DIAG-SCENE4-20261101-PRL-HAAR-FP32`：`test_result/lwpd_prl_diagnosis_v1/scene4_20261101/prl_haar_fp32/metrics.csv`；checkpoint `models/lwpd_qxs_prl_v1/e2_prl_seed42/epoch_016.pth`。

## QXS-VAL1000-DIRECT-LWPD-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| DIRECT-R3-QXS-VAL1000-EPOCH9-LWPD-REFERENCE | Direct Fusion R3 epoch 9 reference for LWPD | development_comparison | val@0 | 1000 | 0.7033 | 0.7223 | 0.1084 | 0.5325 | 0.5680 | 0.3927 |
| LWPD-E4-QXS-VAL1000-SEED42 | Complete LWPD E4: directional Haar gating, controlled SAR recovery and phase symmetry | development_positive_conditional | val@0 | 1000 | 0.7092 | 0.7318 | 0.0971 | 0.5280 | 0.5655 | 0.4018 |
| LWPD-E2-QXS-VAL1000-SEED42 | LWPD E2: fixed Haar with directional context gating | development_baseline | val@0 | 1000 | 0.7104 | 0.7324 | 0.0646 | 0.5286 | 0.5678 | 0.4009 |
| LWPD-E2-PRL-QXS-VAL1000-SEED42 | LWPD E2-PRL: Haar-initialized learnable perfect-reconstruction lifting | development_screen_failed | val@0 | 1000 | 0.7110 | 0.7322 | 0.0959 | 0.5279 | 0.5653 | 0.4033 |

来源：

- `DIRECT-R3-QXS-VAL1000-EPOCH9-LWPD-REFERENCE`：`test_result/direct_fusion_qxs/qxs_direct_seed42_v1/val_extended/r3_epoch_009/metrics.csv`；checkpoint `models/direct_fusion_qxs/qxs_direct_seed42_v1/r3/epoch_009.pth`。
- `LWPD-E4-QXS-VAL1000-SEED42`：`test_result/lwpd_qxs_full_v1/e4_seed42_val1000/metrics.csv`；checkpoint `models/lwpd_qxs_full_v1/e4_seed42/epoch_016.pth`。
- `LWPD-E2-QXS-VAL1000-SEED42`：`test_result/lwpd_qxs_ablation_v1/val1000/e2_seed42/metrics.csv`；checkpoint `models/lwpd_qxs_ablation_v1/e2_seed42/epoch_016.pth`。
- `LWPD-E2-PRL-QXS-VAL1000-SEED42`：`test_result/lwpd_qxs_prl_v1/e2_prl_seed42_val1000/metrics.csv`；checkpoint `models/lwpd_qxs_prl_v1/e2_prl_seed42/epoch_016.pth`。

## QXS-VAL1000-E2-GENERAL-LOSS-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| E2-GENERAL-LOSS-V1-B-SEED42-VAL1000 | E2 loss b: grad=10.0, structure=0.0 | development_validation | val@0 | 1000 | 0.7110 | 0.7326 | 0.0657 | 0.5292 | 0.5665 | 0.4025 |
| E2-GENERAL-LOSS-V1-S-SEED42-VAL1000 | E2 loss s: grad=10.0, structure=0.5 | development_validation | val@0 | 1000 | 0.7164 | 0.7322 | 0.0780 | 0.5309 | 0.5678 | 0.4034 |

来源：

- `E2-GENERAL-LOSS-V1-B-SEED42-VAL1000`：`test_result/e2_general_loss_v1/val1000/b_seed42/metrics.csv`；checkpoint `models/e2_general_loss_v1/b_seed42/epoch_016.pth`。
- `E2-GENERAL-LOSS-V1-S-SEED42-VAL1000`：`test_result/e2_general_loss_v1/val1000/s_seed42/metrics.csv`；checkpoint `models/e2_general_loss_v1/s_seed42/epoch_016.pth`。

## QXS-VAL1000-SPD-SCALE-SWEEP-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0-QXS-VAL1000-SPD-SCALE-SWEEP | B0 scale-sweep identity | development_selection | val@0 | 1000 | 0.7122 | 0.7192 | 0.1119 | 0.5324 | 0.5696 | 0.3985 |
| SPD-DETAILONLY-QXS-VAL1000-SCALE-SWEEP | B0 + SPD Detail-only (alpha_B=0, alpha_D=1) | development_selected | val@0 | 1000 | 0.7095 | 0.7195 | 0.0906 | 0.5338 | 0.5688 | 0.3994 |
| SPD-DETAILONLY-RETRAINED-QXS-VAL1000 | B0 + retrained SPD Detail-only | development_comparison | val@0 | 1000 | 0.7104 | 0.7191 | 0.0916 | 0.5335 | 0.5689 | 0.3989 |
| SPD-DETAILONLY-ML-W010-QXS-VAL1000 | B0 + SPD Detail-only + high-confidence M_L (lambda_L=0.10) | development_negative | val@0 | 1000 | 0.7101 | 0.7191 | 0.0926 | 0.5335 | 0.5688 | 0.3990 |
| SPD-LINEGATE-P025-QXS-VAL1000 | B0 + SPD Detail-only + continuity-filtered line gate (p=0.25) | development_negative | val@0 | 1000 | 0.7104 | 0.7191 | 0.0916 | 0.5335 | 0.5689 | 0.3989 |
| SPD-LINEGATE-P050-QXS-VAL1000 | B0 + SPD Detail-only + continuity-filtered line gate (p=0.50) | development_negative | val@0 | 1000 | 0.7106 | 0.7191 | 0.0917 | 0.5335 | 0.5689 | 0.3989 |
| SPD-LINEGATE-P100-QXS-VAL1000 | B0 + SPD Detail-only + continuity-filtered line gate (p=1.00) | development_negative | val@0 | 1000 | 0.7107 | 0.7190 | 0.0918 | 0.5335 | 0.5690 | 0.3988 |
| P1-STABLE-DETAIL-QXS-VAL1000 | B0 + P1 provenance-preserving stable-SAR Detail router | development_negative | val@0 | 1000 | 0.7119 | 0.7191 | 0.1128 | 0.5323 | 0.5696 | 0.3983 |
| P2A-SOURCE-PRESERVING-QXS-VAL1000 | B0 + P2-A source-preserving pseudo-sensing Detail adapter | development_negative | val@0 | 1000 | 0.7073 | 0.7180 | 0.1001 | 0.5330 | 0.5702 | 0.3964 |
| P2A-SCALE025-QXS-VAL1000 | B0 + P2-A source-preserving residual (alpha=0.25) | development_selected | val@0 | 1000 | 0.7111 | 0.7190 | 0.1084 | 0.5326 | 0.5697 | 0.3983 |
| P2A-SCALE050-QXS-VAL1000 | B0 + P2-A source-preserving residual (alpha=0.50) | development_negative | val@0 | 1000 | 0.7097 | 0.7187 | 0.1051 | 0.5327 | 0.5698 | 0.3980 |
| P2A-SCALE075-QXS-VAL1000 | B0 + P2-A source-preserving residual (alpha=0.75) | development_negative | val@0 | 1000 | 0.7084 | 0.7184 | 0.1023 | 0.5329 | 0.5700 | 0.3973 |
| OCAF-V2-INDEPENDENT-AMPLITUDE-QXS-VAL1000 | OCAF-v2 independent additive gates and explicit amplitude | development_negative | val@0 | 1000 | 0.3761 | 0.3263 | 0.0076 | 0.5320 | 0.5463 | 0.5022 |
| OCAF-V1-COUNTERFACTUAL-IDENTITY-QXS-VAL1000 | OCAF-v1 optical-identity counterfactual SAR injection | development_negative | val@0 | 1000 | 0.3836 | 0.3269 | 0.0009 | 0.5320 | 0.5462 | 0.5038 |
| CSR-V2-CALIBRATED-ROUTING-QXS-VAL1000 | Detail-only SPD + calibrated competitive structure routing v2 | development_negative | val@0 | 1000 | 0.7097 | 0.7190 | 0.0870 | 0.5337 | 0.5690 | 0.3984 |
| CSR-V1-DETAIL-SPD-QXS-VAL1000 | Detail-only SPD + competitive visible-structure routing v1 | development_negative | val@0 | 1000 | 0.7103 | 0.7190 | 0.0907 | 0.5335 | 0.5689 | 0.3989 |
| OPTICAL-RESCUE-DETAIL-SPD-QXS-VAL1000 | Detail-only SPD + feature-space Optical Rescue | development_negative | val@0 | 1000 | 0.7100 | 0.7190 | 0.0907 | 0.5335 | 0.5689 | 0.3989 |
| MFFA-V1-PHASE2-QXS-VAL1000 | MFFA v1 frequency-factorized provenance-preserving fusion | development_negative | val@0 | 1000 | 0.3419 | 0.4167 | 0.1150 | 0.4938 | 0.5944 | 0.1882 |

来源：

- `B0-QXS-VAL1000-SPD-SCALE-SWEEP`：`test_result/v21_softmask/spd_scale_sweep_b0_val1000/summary.json`；checkpoint `models/qxslab_saropt_from_ivf/CDDFuse_OS_20260924_012025/best.pth`。
- `SPD-DETAILONLY-QXS-VAL1000-SCALE-SWEEP`：`test_result/v21_softmask/spd_scale_sweep_b0_val1000/summary.json`；checkpoint `models/v21_softmask/spd_nomask_seed42/best.pth`。
- `SPD-DETAILONLY-RETRAINED-QXS-VAL1000`：`test_result/v21_softmask/spd_detail_only_retrained_on_b0_val1000/summary.json`；checkpoint `models/v21_softmask/spd_detail_only_seed42/best.pth`。
- `SPD-DETAILONLY-ML-W010-QXS-VAL1000`：`test_result/v21_softmask/spd_detail_only_ml_w010_on_b0_val1000/summary.json`；checkpoint `models/v21_softmask/spd_detail_only_ml_w010_seed42/best.pth`。
- `SPD-LINEGATE-P025-QXS-VAL1000`：`test_result/v21_softmask/spd_line_gated_detail_val1000/summary.json`；checkpoint `models/v21_softmask/spd_detail_only_seed42/best.pth`。
- `SPD-LINEGATE-P050-QXS-VAL1000`：`test_result/v21_softmask/spd_line_gated_detail_val1000/summary.json`；checkpoint `models/v21_softmask/spd_detail_only_seed42/best.pth`。
- `SPD-LINEGATE-P100-QXS-VAL1000`：`test_result/v21_softmask/spd_line_gated_detail_val1000/summary.json`；checkpoint `models/v21_softmask/spd_detail_only_seed42/best.pth`。
- `P1-STABLE-DETAIL-QXS-VAL1000`：`test_result/fusion_p1/stable_detail_seed42_val1000/summary.json`；checkpoint `models/fusion_p1/stable_detail_seed42/best.pth`。
- `P2A-SOURCE-PRESERVING-QXS-VAL1000`：`test_result/fusion_p2a/source_preserving_seed42_val1000/summary.json`；checkpoint `models/fusion_p2a/source_preserving_seed42/best.pth`。
- `P2A-SCALE025-QXS-VAL1000`：`test_result/fusion_p2a/source_preserving_scale_sweep_val1000/summary.json`；checkpoint `models/fusion_p2a/source_preserving_seed42/best.pth`。
- `P2A-SCALE050-QXS-VAL1000`：`test_result/fusion_p2a/source_preserving_scale_sweep_val1000/summary.json`；checkpoint `models/fusion_p2a/source_preserving_seed42/best.pth`。
- `P2A-SCALE075-QXS-VAL1000`：`test_result/fusion_p2a/source_preserving_scale_sweep_val1000/summary.json`；checkpoint `models/fusion_p2a/source_preserving_seed42/best.pth`。
- `OCAF-V2-INDEPENDENT-AMPLITUDE-QXS-VAL1000`：`test_result/ocaf_v2/independent_amplitude_seed42_val1000/summary.json`；checkpoint `models/ocaf_v2/independent_amplitude_seed42/best.pth`。
- `OCAF-V1-COUNTERFACTUAL-IDENTITY-QXS-VAL1000`：`test_result/ocaf_v1/counterfactual_identity_seed42_val1000/summary.json`；checkpoint `models/ocaf_v1/counterfactual_identity_seed42/best.pth`。
- `CSR-V2-CALIBRATED-ROUTING-QXS-VAL1000`：`test_result/csr_v2/calibrated_routing_seed42_val1000/summary.json`；checkpoint `models/csr_v2/calibrated_routing_seed42/best.pth`。
- `CSR-V1-DETAIL-SPD-QXS-VAL1000`：`test_result/csr_v1/detail_spd_seed42_val1000/summary.json`；checkpoint `models/csr_v1/detail_spd_seed42/best.pth`。
- `OPTICAL-RESCUE-DETAIL-SPD-QXS-VAL1000`：`test_result/optical_rescue/detail_spd_gradfix_seed42_val1000/summary.json`；checkpoint `models/optical_rescue/detail_spd_gradfix_seed42/best.pth`。
- `MFFA-V1-PHASE2-QXS-VAL1000`：`test_result/mffa_v1/phase2_seed42_val1000_evalfix/summary.json`；checkpoint `models/mffa_v1/phase2_seed42/best.pth`。

## SAR-DESPECKLE-5632STRIPE50-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| SPD-NOMASK-SAR-DESPECKLE-INTERNAL50 | SPD no-mask seed42 | internal_confirmation | internal_test_coordinate_group_5632@0 | 50 | PSNR=25.5198, SSIM=0.7140, MAE=0.0418, edge_MAE=0.0169, edge_correlation=0.6623 |
| SPD-DETAILONLY-RETRAINED-SAR-DESPECKLE-INTERNAL50 | SPD Detail-only retrained seed42 | internal_confirmation | internal_test_coordinate_group_5632@0 | 50 | PSNR=25.6083, SSIM=0.7131, MAE=0.0412, edge_MAE=0.0169, edge_correlation=0.6520 |
| SPD-DETAILONLY-ML-W010-SAR-DESPECKLE-INTERNAL50 | SPD Detail-only + high-confidence M_L (lambda_L=0.10) | development_validation | internal_test_coordinate_group_5632@0 | 50 | PSNR=25.6039, SSIM=0.7135, MAE=0.0412, edge_MAE=0.0169, edge_correlation=0.6538 |

来源：

- `SPD-NOMASK-SAR-DESPECKLE-INTERNAL50`：`test_result/v21_softmask/spd_nomask_seed42_internal_test50/summary.json`；checkpoint `models/v21_softmask/spd_nomask_seed42/best.pth`。
- `SPD-DETAILONLY-RETRAINED-SAR-DESPECKLE-INTERNAL50`：`test_result/v21_softmask/spd_detail_only_seed42_internal_test50/summary.json`；checkpoint `models/v21_softmask/spd_detail_only_seed42/best.pth`。
- `SPD-DETAILONLY-ML-W010-SAR-DESPECKLE-INTERNAL50`：`test_result/v21_softmask/spd_detail_only_ml_w010_seed42_internal_test50/summary.json`；checkpoint `models/v21_softmask/spd_detail_only_ml_w010_seed42/best.pth`。

## 已追溯的用户表格

用户给出的 B0、B1-v5、C0-minimax 六项指标均来自 `QXS-TEST1000-PAPER-CLAMP-v1`，可以直接相互比较。
当前 SPD 只有 val200 开发结果，不能直接追加到该正式 test1000 表中；必须在模型冻结后单独执行 test1000 并登记为同一组。
