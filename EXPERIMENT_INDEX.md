# 实验数据索引

更新时间：2026-11-17  
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

## E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE11412

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE11412 | E2 + PAC-Mamba residual scale intervention (0, 0.5, 1) | diagnostic_single_scene | val@599 | 1 | original_e2_nabf=0.0343, pac_off_nabf=0.0460, pac_half_nabf=0.0465, pac_on_nabf=0.0474, on_off_raw_mae=0.0008, applied_residual_rms_ratio=0.0132 |

来源：

- `E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE11412`：`test_result/e2_pac_mamba_diagnosis_v1/val_scene11412/summary.json`；checkpoint `models/e2_pac_mamba_v1/seed42/epoch_012.pth`。

## E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE15600

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE15600 | E2 + PAC-Mamba residual scale intervention (0, 0.5, 1) | diagnostic_single_scene | val@799 | 1 | original_e2_nabf=0.1142, pac_off_nabf=0.1186, pac_half_nabf=0.1185, pac_on_nabf=0.1179, on_off_raw_mae=0.0019, applied_residual_rms_ratio=0.0111 |

来源：

- `E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE15600`：`test_result/e2_pac_mamba_diagnosis_v1/val_scene15600/summary.json`；checkpoint `models/e2_pac_mamba_v1/seed42/epoch_012.pth`。

## E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE19977

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE19977 | E2 + PAC-Mamba residual scale intervention (0, 0.5, 1) | diagnostic_single_scene | val@999 | 1 | original_e2_nabf=0.1339, pac_off_nabf=0.1305, pac_half_nabf=0.1306, pac_on_nabf=0.1306, on_off_raw_mae=0.0024, applied_residual_rms_ratio=0.0113 |

来源：

- `E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE19977`：`test_result/e2_pac_mamba_diagnosis_v1/val_scene19977/summary.json`；checkpoint `models/e2_pac_mamba_v1/seed42/epoch_012.pth`。

## E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE3616

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE3616 | E2 + PAC-Mamba residual scale intervention (0, 0.5, 1) | diagnostic_single_scene | val@199 | 1 | original_e2_nabf=0.0358, pac_off_nabf=0.0497, pac_half_nabf=0.0498, pac_on_nabf=0.0515, on_off_raw_mae=0.0008, applied_residual_rms_ratio=0.0127 |

来源：

- `E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE3616`：`test_result/e2_pac_mamba_diagnosis_v1/val_scene3616/summary.json`；checkpoint `models/e2_pac_mamba_v1/seed42/epoch_012.pth`。

## E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE4

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE4 | E2 + PAC-Mamba residual scale intervention (0, 0.5, 1) | diagnostic_single_scene | val@0 | 1 | original_e2_nabf=0.0511, pac_off_nabf=0.0667, pac_half_nabf=0.0680, pac_on_nabf=0.0687, on_off_raw_mae=0.0008, applied_residual_rms_ratio=0.0118 |

来源：

- `E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE4`：`test_result/e2_pac_mamba_diagnosis_v1/val_scene4/summary.json`；checkpoint `models/e2_pac_mamba_v1/seed42/epoch_012.pth`。

## E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE7377

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE7377 | E2 + PAC-Mamba residual scale intervention (0, 0.5, 1) | diagnostic_single_scene | val@399 | 1 | original_e2_nabf=0.0254, pac_off_nabf=0.0395, pac_half_nabf=0.0403, pac_on_nabf=0.0411, on_off_raw_mae=0.0006, applied_residual_rms_ratio=0.0128 |

来源：

- `E2-PAC-MAMBA-V1-RESIDUAL-DIAG-VAL-SCENE7377`：`test_result/e2_pac_mamba_diagnosis_v1/val_scene7377/summary.json`；checkpoint `models/e2_pac_mamba_v1/seed42/epoch_012.pth`。

## E2-TARGET-DIAG-749FF4BFE485

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-TARGET-DIAG-749FF4BFE485 | Saved E2-B/G1 target-conflict and BGC1 diagnostic; no candidate training | diagnostic_fixed_subset | val@0 | 6 | target_gap_abs_mean=0.3836, correction_mean=0.0425, removed_excess_ratio=0.0462 |

来源：

- `E2-TARGET-DIAG-749FF4BFE485`：`test_result/e2_target_diagnosis_v1/fixed6_review_v1/summary.json`；checkpoint `models/e2_general_loss_v1/b_seed42/epoch_016.pth`。

## E2-TARGET-DIAG-E5D71BD3E750

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-TARGET-DIAG-E5D71BD3E750 | Saved E2-B/G1 target-conflict and BGC1 diagnostic; no candidate training | diagnostic_single_scene | val@0 | 1 | target_gap_abs_mean=0.4231, correction_mean=0.0374, removed_excess_ratio=0.0448 |

来源：

- `E2-TARGET-DIAG-E5D71BD3E750`：`test_result/e2_target_diagnosis_v1/scene4_fp32_20261104/summary.json`；checkpoint `models/e2_general_loss_v1/b_seed42/epoch_016.pth`。

## E2-TARGET-DIAG-EE5E90C58906

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-TARGET-DIAG-EE5E90C58906 | Saved E2-B/G1 target-conflict and BGC1 diagnostic; no candidate training | diagnostic_single_scene | val@0 | 1 | target_gap_abs_mean=0.4231, correction_mean=0.0374, removed_excess_ratio=0.0448 |

来源：

- `E2-TARGET-DIAG-EE5E90C58906`：`test_result/e2_target_diagnosis_v1/scene4_20261104/summary.json`；checkpoint `models/e2_general_loss_v1/b_seed42/epoch_016.pth`。

## LLCTX-SINGLE-DIAG-4E0745F092DC

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| LLCTX-SINGLE-DIAG-4E0745F092DC | LL-context current-output on/off; original E2 as single-scene reference | diagnostic_single_scene | val@0 | 1 | on_off_raw_mae=0.0055, on_off_u8_changed_fraction=0.9120, on_nabf=0.0709, off_nabf=0.0680, original_nabf=0.0511 |

来源：

- `LLCTX-SINGLE-DIAG-4E0745F092DC`：`test_result/lwpd_llctx_diagnosis_v1/scene4_20261104/summary.json`；checkpoint `models/lwpd_e2_llctx_v1/seed42/epoch_012.pth`。

## LLCTX-SINGLE-DIAG-C27319CA6B86

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| LLCTX-SINGLE-DIAG-C27319CA6B86 | LL-context one-scene intervention probes; original E2 as reference | diagnostic_single_scene | val@0 | 1 | on_off_raw_mae=0.0055, on_off_u8_changed_fraction=0.9120, on_nabf=0.0709, off_nabf=0.0680, original_nabf=0.0511 |

来源：

- `LLCTX-SINGLE-DIAG-C27319CA6B86`：`test_result/lwpd_llctx_diagnosis_v1/scene4_repair_probes_20261104/summary.json`；checkpoint `models/lwpd_e2_llctx_v1/seed42/epoch_012.pth`。

## MFFA-V1-PHASE1-QXS-VAL1000-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| MFFA-V1-PHASE1-QXS-VAL1000 | MFFA v1 Phase-1 symmetric frequency autoencoder | development_passed | val@0 | 1000 | optical_mae=0.0000, optical_ssim=1.0000, sar_mae=0.0004, sar_psnr=59.7408, sar_ssim=1.0000, optical_high=0.0127, sar_high=0.0081 |

来源：

- `MFFA-V1-PHASE1-QXS-VAL1000`：`test_result/mffa_v1/phase1_dtypefix_seed42_val1000/summary.json`；checkpoint `models/mffa_v1/phase1_dtypefix_seed42/best.pth`。

## QXS-DETAIL-DIRECTIONS-82b43edf064a

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-DETAIL-DIRECTIONS-82b43edf064a | Frozen Detail directions diagnosis | diagnostic_fixed_subset | val@0 | 4 | optical_full_reduce_channel_gain=0.6408, optical_fine_reduce_channel_gain=0.6525, sar_full_reduce_channel_gain=0.7438, sar_fine_reduce_channel_gain=0.7747, optical_full_output_gain=2.3032, optical_fine_output_gain=1.8276, sar_full_output_gain=5.1279, sar_fine_output_gain=5.1560, optical_full_output_fine_gain=1.6192, optical_fine_output_fine_gain=1.7140, sar_full_output_fine_gain=4.8015, sar_fine_output_fine_gain=4.9564 |

来源：

- `QXS-DETAIL-DIRECTIONS-82b43edf064a`：`test_result/e2_detail_direction_v1/fixed4_20261116/diagnosis.json`；checkpoint `models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。

## QXS-DETAIL-DIRECTIONS-e1a364ade02c

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-DETAIL-DIRECTIONS-e1a364ade02c | Frozen Detail directions diagnosis | diagnostic_fixed_subset | val@0 | 1 | optical_full_reduce_channel_gain=0.6423, optical_fine_reduce_channel_gain=0.6555, sar_full_reduce_channel_gain=0.7445, sar_fine_reduce_channel_gain=0.7750, optical_full_output_gain=2.0302, optical_fine_output_gain=1.7725, sar_full_output_gain=5.2905, sar_fine_output_gain=5.3185, optical_full_output_fine_gain=1.5313, optical_fine_output_fine_gain=1.6409, sar_full_output_fine_gain=4.9730, sar_fine_output_fine_gain=5.1146 |

来源：

- `QXS-DETAIL-DIRECTIONS-e1a364ade02c`：`test_result/e2_detail_direction_v1/scene4_20261116/diagnosis.json`；checkpoint `models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。

## QXS-DETAIL-ROUTES-03508ba6dfd8

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-DETAIL-ROUTES-03508ba6dfd8 | Frozen Detail routes diagnosis | diagnostic_fixed_subset | val@0 | 4 | regional_Qabf=0.7590, fixed_pair_Qabf=0.7589, rolled_Qabf=0.7588, regional_Nabf=0.0416, fixed_pair_Nabf=0.0415, rolled_Nabf=0.0412, regional_VIF=0.2928, fixed_pair_VIF=0.2939, rolled_VIF=0.2925 |

来源：

- `QXS-DETAIL-ROUTES-03508ba6dfd8`：`test_result/s2_route_intervention_v1/fixed4/diagnosis.json`；checkpoint `models/s2_expert_pilot_v2/top2_seed42/final.pth`。

## QXS-DETAIL-ROUTES-c6499c49f85a

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-DETAIL-ROUTES-c6499c49f85a | Frozen Detail routes diagnosis | diagnostic_fixed_subset | val@0 | 1 | regional_Qabf=0.7969, fixed_pair_Qabf=0.7970, rolled_Qabf=0.7969, regional_Nabf=0.0360, fixed_pair_Nabf=0.0363, rolled_Nabf=0.0357, regional_VIF=0.3029, fixed_pair_VIF=0.3079, rolled_VIF=0.3026 |

来源：

- `QXS-DETAIL-ROUTES-c6499c49f85a`：`test_result/s2_route_intervention_v1/scene17157_20261113/diagnosis.json`；checkpoint `models/s2_expert_pilot_v2/top2_seed42/final.pth`。

## QXS-DETAIL-TRANSFER-44240793b8a8

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-DETAIL-TRANSFER-44240793b8a8 | Frozen Detail transfer diagnosis | diagnostic_fixed_subset | val@0 | 4 | optical_fine_detail_same_band_gain=0.2452, optical_middle_detail_same_band_gain=0.2209, sar_fine_detail_same_band_gain=0.1404, sar_middle_detail_same_band_gain=0.1556, decoder_fine_detail_same_band_gain=1.0000, decoder_middle_detail_same_band_gain=1.0000, decoder_dc_detail_same_band_gain=1.0000, optical_fine_output_same_band_gain=0.5399, optical_middle_output_same_band_gain=0.2558, sar_fine_output_same_band_gain=0.8446, sar_middle_output_same_band_gain=0.8917, decoder_fine_output_same_band_gain=4.1408, decoder_middle_output_same_band_gain=1.9225, decoder_dc_output_same_band_gain=0.9505 |

来源：

- `QXS-DETAIL-TRANSFER-44240793b8a8`：`test_result/e2_detail_transfer_v1/fixed4/diagnosis.json`；checkpoint `models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。

## QXS-DETAIL-TRANSFER-cba6da26b2cb

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-DETAIL-TRANSFER-cba6da26b2cb | Frozen Detail transfer diagnosis | diagnostic_fixed_subset | val@0 | 1 | optical_fine_detail_same_band_gain=0.2094, optical_middle_detail_same_band_gain=0.2124, sar_fine_detail_same_band_gain=0.1499, sar_middle_detail_same_band_gain=0.1601, decoder_fine_detail_same_band_gain=1.0000, decoder_middle_detail_same_band_gain=1.0000, decoder_dc_detail_same_band_gain=1.0000, optical_fine_output_same_band_gain=0.4097, optical_middle_output_same_band_gain=0.1860, sar_fine_output_same_band_gain=0.9186, sar_middle_output_same_band_gain=0.9651, decoder_fine_output_same_band_gain=4.9101, decoder_middle_output_same_band_gain=2.2013, decoder_dc_output_same_band_gain=1.5795 |

来源：

- `QXS-DETAIL-TRANSFER-cba6da26b2cb`：`test_result/e2_detail_transfer_v1/scene4_20261113/diagnosis.json`；checkpoint `models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。

## QXS-E2-ASPR-PILOT-V1-FIXED-VAL64-REFERENCE-FMI

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-E2-ASPR-PILOT-V1-FIXED-VAL64-REFERENCE-FMI-e2 | E2 | diagnostic_fixed_subset | val@0 | 64 | 0.6306 | 0.7890 | 0.0319 | 0.5293 | 0.5876 | 0.3353 |
| QXS-E2-ASPR-PILOT-V1-FIXED-VAL64-REFERENCE-FMI-lowpass | E2 + optical lowpass | diagnostic_fixed_subset | val@0 | 64 | 0.5021 | 0.7406 | 0.0151 | 0.5316 | 0.6119 | 0.2442 |
| QXS-E2-ASPR-PILOT-V1-FIXED-VAL64-REFERENCE-FMI-optical | E2 + optical constraints | diagnostic_fixed_subset | val@0 | 64 | 0.4448 | 0.7215 | 0.0082 | 0.5375 | 0.6144 | 0.2541 |
| QXS-E2-ASPR-PILOT-V1-FIXED-VAL64-REFERENCE-FMI-shifted | E2 + shifted SAR protection | diagnostic_fixed_subset | val@0 | 64 | 0.4500 | 0.7251 | 0.0082 | 0.5378 | 0.6143 | 0.2554 |
| QXS-E2-ASPR-PILOT-V1-FIXED-VAL64-REFERENCE-FMI-fixed | E2 + fixed ASPR | diagnostic_fixed_subset | val@0 | 64 | 0.4500 | 0.7254 | 0.0082 | 0.5378 | 0.6144 | 0.2556 |

来源：

- `QXS-E2-ASPR-PILOT-V1-FIXED-VAL64-REFERENCE-FMI-e2`：`test_result/e2_aspr_fixed_pilot_v1/val64/fixed-val/e2/metrics.csv`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-ASPR-PILOT-V1-FIXED-VAL64-REFERENCE-FMI-lowpass`：`test_result/e2_aspr_fixed_pilot_v1/val64/fixed-val/lowpass/metrics.csv`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-ASPR-PILOT-V1-FIXED-VAL64-REFERENCE-FMI-optical`：`test_result/e2_aspr_fixed_pilot_v1/val64/fixed-val/optical/metrics.csv`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-ASPR-PILOT-V1-FIXED-VAL64-REFERENCE-FMI-shifted`：`test_result/e2_aspr_fixed_pilot_v1/val64/fixed-val/shifted/metrics.csv`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-ASPR-PILOT-V1-FIXED-VAL64-REFERENCE-FMI-fixed`：`test_result/e2_aspr_fixed_pilot_v1/val64/fixed-val/fixed/metrics.csv`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-C-DIAG-4-741730601732

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-E2-C-DIAG-4-741730601732 | E2 光学 Base 固定权重的区域通道关系干预 | diagnostic_single_scene | val@0 | 1 | affine_global_base_rms=0.0040, affine_local_base_rms=0.0000, affine_centered_local_base_rms=0.0000, affine_balanced_local_base_rms=0.0000, shuffle_global_base_rms=0.0001, shuffle_local_base_rms=0.0000, shuffle_centered_local_base_rms=0.0000, shuffle_balanced_local_base_rms=0.0003 |

来源：

- `QXS-E2-C-DIAG-4-741730601732`：`test_result/c_base_relation_diagnosis_v1/scene4_20261113/summary.json`；checkpoint `models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。

## QXS-E2-CFAO-OPERATOR-V1-SCENE1276

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-E2-CFAO-OPERATOR-V1-SCENE1276 | Fourier convolution vs explicit spatial channel mixing | diagnostic_single_scene | val@0 | 1 | circular_spatial_relative_error=0.0000, circular_imaginary_relative_rms=0.0000, zero_padding_relative_difference=0.0274, zero_padding_imaginary_relative_rms=0.0286, shift1_relative_equivariance_error=0.0202, shift8_relative_equivariance_error=0.1617 |

来源：

- `QXS-E2-CFAO-OPERATOR-V1-SCENE1276`：`test_result/cfao_sampling_origin_v1/scene1276_20261117/fourier.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-CONTOUR-EVIDENCE-V1-DEVELOPMENT-VAL8

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-E2-CONTOUR-EVIDENCE-V1-DEVELOPMENT-VAL8 | Archived STG probes, fixed val8 development | diagnostic_fixed_subset | val@0 | 8 | missing_probes=3.0000, hard_negatives=16.0000, optical_hits_at_1_false_link=3.0000, best_single_hits_at_1_false_link=2.0000, linked_hits_at_1_false_link=3.0000, linked_only_missing=0.0000, linked_only_negatives=1.0000 |

来源：

- `QXS-E2-CONTOUR-EVIDENCE-V1-DEVELOPMENT-VAL8`：`test_result/e2_contour_evidence_v1/development/summary.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-CONTOUR-EVIDENCE-V1-HOLDOUT-VAL8

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-E2-CONTOUR-EVIDENCE-V1-HOLDOUT-VAL8 | Disjoint QXS patch IDs, fixed val8 holdout | diagnostic_fixed_subset | val@0 | 8 | missing_probes=4.0000, hard_negatives=16.0000, optical_hits_at_1_false_link=3.0000, best_single_hits_at_1_false_link=3.0000, linked_hits_at_1_false_link=2.0000, linked_only_missing=0.0000, linked_only_negatives=3.0000 |

来源：

- `QXS-E2-CONTOUR-EVIDENCE-V1-HOLDOUT-VAL8`：`test_result/e2_contour_evidence_v1/holdout/summary.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-DCAF-V1-SCENE5730-CORRECTED-SAR-REF-FMI

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-E2-DCAF-V1-SCENE5730-CORRECTED-SAR-REF-FMI-ZERO | Frozen E2, zero offset | diagnostic_single_scene | val@0 | 1 | 0.1924 | 0.1186 | 0.1634 | 0.0604 | 0.1830 | 0.0178 |
| QXS-E2-DCAF-V1-SCENE5730-CORRECTED-SAR-REF-FMI-CORRECT-PRE-HAAR | Known SAR translation before E2 | diagnostic_single_scene | val@0 | 1 | 1.0398 | 0.8738 | 0.0314 | 0.5178 | 0.5541 | 0.3974 |
| QXS-E2-DCAF-V1-SCENE5730-CORRECTED-SAR-REF-FMI-REVERSE-PRE-HAAR | Opposite SAR translation before E2 | diagnostic_single_scene | val@0 | 1 | 0.1452 | 0.1218 | 0.1618 | 0.0565 | 0.1624 | 0.0163 |
| QXS-E2-DCAF-V1-SCENE5730-CORRECTED-SAR-REF-FMI-CORRECT-POST-GATE | Known translation of SAR Detail after direction gating | diagnostic_single_scene | val@0 | 1 | 0.2744 | 0.1940 | 0.0981 | 0.2689 | 0.2796 | 0.0340 |
| QXS-E2-DCAF-V1-SCENE5730-CORRECTED-SAR-REF-FMI-SHIFT-EXISTING-OUTPUT | Translation of the archived-path E2 output | diagnostic_single_scene | val@0 | 1 | 0.8944 | 0.8741 | 0.0316 | 0.5160 | 0.5517 | 0.3992 |

来源：

- `QXS-E2-DCAF-V1-SCENE5730-CORRECTED-SAR-REF-FMI-ZERO`：`test_result/dcaf_geometry_v1/scene5730_paired_20261117/per_image.csv`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-DCAF-V1-SCENE5730-CORRECTED-SAR-REF-FMI-CORRECT-PRE-HAAR`：`test_result/dcaf_geometry_v1/scene5730_paired_20261117/per_image.csv`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-DCAF-V1-SCENE5730-CORRECTED-SAR-REF-FMI-REVERSE-PRE-HAAR`：`test_result/dcaf_geometry_v1/scene5730_paired_20261117/per_image.csv`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-DCAF-V1-SCENE5730-CORRECTED-SAR-REF-FMI-CORRECT-POST-GATE`：`test_result/dcaf_geometry_v1/scene5730_paired_20261117/per_image.csv`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-DCAF-V1-SCENE5730-CORRECTED-SAR-REF-FMI-SHIFT-EXISTING-OUTPUT`：`test_result/dcaf_geometry_v1/scene5730_paired_20261117/per_image.csv`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-DCAF-V1-SCENE5730-RAW-SAR-REF-FMI

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-E2-DCAF-V1-SCENE5730-RAW-SAR-REF-FMI-ZERO | Frozen E2, zero offset | diagnostic_single_scene | val@0 | 1 | 0.9019 | 0.8753 | 0.0324 | 0.5264 | 0.5665 | 0.4005 |
| QXS-E2-DCAF-V1-SCENE5730-RAW-SAR-REF-FMI-CORRECT-PRE-HAAR | Known SAR translation before E2 | diagnostic_single_scene | val@0 | 1 | 0.1381 | 0.1191 | 0.1628 | 0.0523 | 0.1713 | 0.0167 |
| QXS-E2-DCAF-V1-SCENE5730-RAW-SAR-REF-FMI-REVERSE-PRE-HAAR | Opposite SAR translation before E2 | diagnostic_single_scene | val@0 | 1 | 0.1552 | 0.1182 | 0.1628 | 0.0579 | 0.1857 | 0.0164 |
| QXS-E2-DCAF-V1-SCENE5730-RAW-SAR-REF-FMI-CORRECT-POST-GATE | Known translation of SAR Detail after direction gating | diagnostic_single_scene | val@0 | 1 | 0.3860 | 0.3077 | 0.0707 | 0.3518 | 0.5342 | 0.1386 |
| QXS-E2-DCAF-V1-SCENE5730-RAW-SAR-REF-FMI-SHIFT-EXISTING-OUTPUT | Translation of the archived-path E2 output | diagnostic_single_scene | val@0 | 1 | 0.1368 | 0.1188 | 0.1629 | 0.0499 | 0.1682 | 0.0164 |

来源：

- `QXS-E2-DCAF-V1-SCENE5730-RAW-SAR-REF-FMI-ZERO`：`test_result/dcaf_geometry_v1/scene5730_paired_20261117/per_image.csv`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-DCAF-V1-SCENE5730-RAW-SAR-REF-FMI-CORRECT-PRE-HAAR`：`test_result/dcaf_geometry_v1/scene5730_paired_20261117/per_image.csv`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-DCAF-V1-SCENE5730-RAW-SAR-REF-FMI-REVERSE-PRE-HAAR`：`test_result/dcaf_geometry_v1/scene5730_paired_20261117/per_image.csv`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-DCAF-V1-SCENE5730-RAW-SAR-REF-FMI-CORRECT-POST-GATE`：`test_result/dcaf_geometry_v1/scene5730_paired_20261117/per_image.csv`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-DCAF-V1-SCENE5730-RAW-SAR-REF-FMI-SHIFT-EXISTING-OUTPUT`：`test_result/dcaf_geometry_v1/scene5730_paired_20261117/per_image.csv`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-DCAF-V1-VAL8-SOURCE-GEOMETRY

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-E2-DCAF-V1-VAL8-SOURCE-GEOMETRY | Manual source-only landmarks with one held-out point | diagnostic_fixed_subset | val@0 | 8 | scenes_inspected=8.0000, exploratory_scenes=1.0000, held_out_distance_zero=12.1655, held_out_distance_corrected=3.1623, held_out_distance_reverse=21.2132 |

来源：

- `QXS-E2-DCAF-V1-VAL8-SOURCE-GEOMETRY`：`test_result/dcaf_geometry_v1/scene5730_paired_20261117/geometry_summary.json`。

## QXS-E2-DRF-FIXED-SPLIT-V1-VAL8-REF-FMI

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-E2-DRF-FIXED-SPLIT-V1-VAL8-REF-FMI-E2 | Frozen optical Base E2 | diagnostic_fixed_subset | val@0 | 8 | 0.6537 | 0.7863 | 0.0332 | 0.5262 | 0.5673 | 0.3338 |
| QXS-E2-DRF-FIXED-SPLIT-V1-VAL8-REF-FMI-UNIFORM | Equal-RMS uniform SAR Detail attenuation | diagnostic_fixed_subset | val@0 | 8 | 0.6395 | 0.7860 | 0.0210 | 0.5281 | 0.5672 | 0.3327 |
| QXS-E2-DRF-FIXED-SPLIT-V1-VAL8-REF-FMI-OPTICAL-BASE-VARIANCE | Equal-RMS fixed optical Base variance split | diagnostic_fixed_subset | val@0 | 8 | 0.6441 | 0.7848 | 0.0283 | 0.5278 | 0.5672 | 0.3284 |
| QXS-E2-DRF-FIXED-SPLIT-V1-VAL8-REF-FMI-DETAIL-COHERENCE | Equal-RMS SAR Detail directional coherence split | diagnostic_fixed_subset | val@0 | 8 | 0.6388 | 0.7860 | 0.0211 | 0.5281 | 0.5671 | 0.3321 |
| QXS-E2-DRF-FIXED-SPLIT-V1-VAL8-REF-FMI-DETAIL-AND-FUSED-BASE | Equal-RMS SAR Detail coherence and fused Base split | diagnostic_fixed_subset | val@0 | 8 | 0.6426 | 0.7858 | 0.0216 | 0.5282 | 0.5672 | 0.3315 |

来源：

- `QXS-E2-DRF-FIXED-SPLIT-V1-VAL8-REF-FMI-E2`：`test_result/drf_fixed_split_v1/val8_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-DRF-FIXED-SPLIT-V1-VAL8-REF-FMI-UNIFORM`：`test_result/drf_fixed_split_v1/val8_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-DRF-FIXED-SPLIT-V1-VAL8-REF-FMI-OPTICAL-BASE-VARIANCE`：`test_result/drf_fixed_split_v1/val8_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-DRF-FIXED-SPLIT-V1-VAL8-REF-FMI-DETAIL-COHERENCE`：`test_result/drf_fixed_split_v1/val8_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-DRF-FIXED-SPLIT-V1-VAL8-REF-FMI-DETAIL-AND-FUSED-BASE`：`test_result/drf_fixed_split_v1/val8_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-DRF-MANUAL-BACKGROUND-V1-VAL8-REF-FMI

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-E2-DRF-MANUAL-BACKGROUND-V1-VAL8-REF-FMI-E2 | Archived frozen optical Base E2 | diagnostic_fixed_subset | val@0 | 8 | 0.6537 | 0.7863 | 0.0332 | 0.5262 | 0.5673 | 0.3338 |
| QXS-E2-DRF-MANUAL-BACKGROUND-V1-VAL8-REF-FMI-UNIFORM | Archived 10 percent uniform SAR Detail attenuation | diagnostic_fixed_subset | val@0 | 8 | 0.6395 | 0.7860 | 0.0210 | 0.5281 | 0.5672 | 0.3327 |
| QXS-E2-DRF-MANUAL-BACKGROUND-V1-VAL8-REF-FMI-MANUAL-BACKGROUND | Equal-RMS source-annotated background-only SAR Detail attenuation | diagnostic_fixed_subset | val@0 | 8 | 0.6376 | 0.7842 | 0.0320 | 0.5265 | 0.5671 | 0.3277 |

来源：

- `QXS-E2-DRF-MANUAL-BACKGROUND-V1-VAL8-REF-FMI-E2`：`test_result/drf_manual_mask_v1/val8_paired_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-DRF-MANUAL-BACKGROUND-V1-VAL8-REF-FMI-UNIFORM`：`test_result/drf_manual_mask_v1/val8_paired_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-DRF-MANUAL-BACKGROUND-V1-VAL8-REF-FMI-MANUAL-BACKGROUND`：`test_result/drf_manual_mask_v1/val8_paired_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-GATE-SCALE-FIXED4-20261113

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-E2-GATE-SCALE-FIXED4-20261113 | e2_gate_and_orthogonal_scale_v1 | diagnostic_fixed_subset | val@0 | 4 | optical_gate_energy_survival=0.4161, sar_gate_energy_survival=0.8152, fused_detail_dc_fraction=0.8897 |

来源：

- `QXS-E2-GATE-SCALE-FIXED4-20261113`：`test_result/e2_gate_scale_v1/fixed4_20261113/diagnosis.json`；checkpoint `models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。

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

## QXS-E2-GENERAL-LOSS-TRAIN8000-g1-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-G1-V1-G1-SEED42-TRAIN8000 | E2-G1 PA directional gradient (10), reconstruction (0.1) | training_complete | train@0 | 8000 | train_loss=0.0698, val_loss=0.0694 |

来源：

- `E2-G1-V1-G1-SEED42-TRAIN8000`：`models/e2_g1_v1/g1_seed42/history.csv`；checkpoint `models/e2_g1_v1/g1_seed42/epoch_016.pth`。

## QXS-E2-GENERAL-LOSS-TRAIN8000-s-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-GENERAL-LOSS-V1-S-SEED42-TRAIN8000 | E2 loss s: grad=10.0, structure=0.5 | training_complete | train@0 | 8000 | train_loss=0.9798, val_loss=0.9652 |

来源：

- `E2-GENERAL-LOSS-V1-S-SEED42-TRAIN8000`：`models/e2_general_loss_v1/s_seed42/history.csv`；checkpoint `models/e2_general_loss_v1/s_seed42/epoch_016.pth`。

## QXS-E2-INFORMATION-PATH-FIXED4-20261113

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-E2-INFORMATION-PATH-FIXED4-20261113 | frozen_e2_information_path_v1 | diagnostic_fixed_subset | val@0 | 4 | mean_SSIM_O=0.2604, mean_SSIM_S=0.7998 |

来源：

- `QXS-E2-INFORMATION-PATH-FIXED4-20261113`：`test_result/e2_information_path_v1/fixed4_20261113/diagnosis.json`；checkpoint `models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。

## QXS-E2-LOCAL-CORRESPONDENCE-V1-SEED42-VAL64-REFERENCE-FMI

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-E2-LOCAL-CORRESPONDENCE-V1-SEED42-VAL64-REFERENCE-FMI-candidate | 光学 Base E2 + 可拒配的局部 Detail 交换 | development_validation | val@0 | 64 | 0.6631 | 0.7928 | 0.0281 | 0.5321 | 0.5881 | 0.3419 |

来源：

- `QXS-E2-LOCAL-CORRESPONDENCE-V1-SEED42-VAL64-REFERENCE-FMI-candidate`：`test_result/e2_scratch_local_correspondence_pilot_v1/val64/seed42/candidate/metrics.csv`；checkpoint `models/e2_scratch_local_correspondence_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-MATCHED-NO-PAC-V1-TRAIN8000

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-MATCHED-NO-PAC-V1-SEED42-TRAIN8000 | E2 matched continuation; PAC frozen with exactly zero output | training_complete | train@0 | 8000 | train_loss=0.7070, val_loss=0.6974 |

来源：

- `E2-MATCHED-NO-PAC-V1-SEED42-TRAIN8000`：`models/e2_matched_no_pac_v1/seed42/history.csv`；checkpoint `models/e2_matched_no_pac_v1/seed42/epoch_012.pth`。

## QXS-E2-NO-PAC-STAGE-VAL100-V1

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| E2-MATCHED-NO-PAC-STAGE-V1-E02 | E2 no-PAC stage diagnostic, epoch 2 | diagnostic_fixed_subset | val@450 | 100 | 0.9569 | 0.8161 | 0.0529 | 0.5332 | 0.5485 | 0.4879 |
| E2-MATCHED-NO-PAC-STAGE-V1-E05 | E2 no-PAC stage diagnostic, epoch 5 | diagnostic_fixed_subset | val@450 | 100 | 0.9528 | 0.8163 | 0.0611 | 0.5325 | 0.5481 | 0.4879 |
| E2-MATCHED-NO-PAC-STAGE-V1-E06 | E2 no-PAC stage diagnostic, epoch 6 | diagnostic_fixed_subset | val@450 | 100 | 0.9696 | 0.8163 | 0.0645 | 0.5331 | 0.5481 | 0.4893 |

来源：

- `E2-MATCHED-NO-PAC-STAGE-V1-E02`：`test_result/e2_matched_no_pac_stage_v1/epoch_002/metrics.csv`；checkpoint `models/e2_matched_no_pac_v1/seed42/epoch_002.pth`。
- `E2-MATCHED-NO-PAC-STAGE-V1-E05`：`test_result/e2_matched_no_pac_stage_v1/epoch_005/metrics.csv`；checkpoint `models/e2_matched_no_pac_v1/seed42/epoch_005.pth`。
- `E2-MATCHED-NO-PAC-STAGE-V1-E06`：`test_result/e2_matched_no_pac_stage_v1/epoch_006/metrics.csv`；checkpoint `models/e2_matched_no_pac_v1/seed42/epoch_006.pth`。

## QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS0P2PCT-VAL8-REF-FMI

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS0P2PCT-VAL8-REF-FMI-E2 | Frozen optical Base E2 | diagnostic_fixed_subset | val@0 | 8 | 0.6537 | 0.7863 | 0.0332 | 0.5262 | 0.5673 | 0.3338 |
| QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS0P2PCT-VAL8-REF-FMI-ORIGINAL | Original fusion loss direction | diagnostic_fixed_subset | val@0 | 8 | 0.6540 | 0.7876 | 0.0323 | 0.5263 | 0.5673 | 0.3361 |
| QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS0P2PCT-VAL8-REF-FMI-ALIGNED | Original plus aligned optical structure direction | diagnostic_fixed_subset | val@0 | 8 | 0.6541 | 0.7872 | 0.0328 | 0.5263 | 0.5673 | 0.3354 |
| QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS0P2PCT-VAL8-REF-FMI-SHIFTED | Original plus spatially shifted optical structure direction | diagnostic_fixed_subset | val@0 | 8 | 0.6566 | 0.7872 | 0.0329 | 0.5262 | 0.5672 | 0.3352 |

来源：

- `QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS0P2PCT-VAL8-REF-FMI-E2`：`test_result/okd_frozen_gradient_v1/val8_small_step_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS0P2PCT-VAL8-REF-FMI-ORIGINAL`：`test_result/okd_frozen_gradient_v1/val8_small_step_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS0P2PCT-VAL8-REF-FMI-ALIGNED`：`test_result/okd_frozen_gradient_v1/val8_small_step_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS0P2PCT-VAL8-REF-FMI-SHIFTED`：`test_result/okd_frozen_gradient_v1/val8_small_step_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS2PCT-VAL8-REF-FMI

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS2PCT-VAL8-REF-FMI-E2 | Frozen optical Base E2 | diagnostic_fixed_subset | val@0 | 8 | 0.6537 | 0.7863 | 0.0332 | 0.5262 | 0.5673 | 0.3338 |
| QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS2PCT-VAL8-REF-FMI-ORIGINAL | Original fusion loss direction | diagnostic_fixed_subset | val@0 | 8 | 0.6312 | 0.7875 | 0.0268 | 0.5233 | 0.5667 | 0.3203 |
| QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS2PCT-VAL8-REF-FMI-ALIGNED | Original plus aligned optical structure direction | diagnostic_fixed_subset | val@0 | 8 | 0.6320 | 0.7858 | 0.0296 | 0.5236 | 0.5659 | 0.3186 |
| QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS2PCT-VAL8-REF-FMI-SHIFTED | Original plus spatially shifted optical structure direction | diagnostic_fixed_subset | val@0 | 8 | 0.6357 | 0.7847 | 0.0299 | 0.5218 | 0.5654 | 0.3156 |

来源：

- `QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS2PCT-VAL8-REF-FMI-E2`：`test_result/okd_frozen_gradient_v1/val8_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS2PCT-VAL8-REF-FMI-ORIGINAL`：`test_result/okd_frozen_gradient_v1/val8_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS2PCT-VAL8-REF-FMI-ALIGNED`：`test_result/okd_frozen_gradient_v1/val8_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-OKD-FROZEN-GRADIENT-V1-RMS2PCT-VAL8-REF-FMI-SHIFTED`：`test_result/okd_frozen_gradient_v1/val8_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-PAC-ALIGNED-HAAR-CANCELLATION-V1-SCENE4

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-E2-PAC-ALIGNED-HAAR-CANCELLATION-V1-SCENE4 | Frozen legacy PAC input cancellation | diagnostic_single_scene | val@0 | 1 | optical_rms=0.0148, sar_rms=0.0855, optical_pooled_rms=0.0000, sar_pooled_rms=0.0000, delta_relative_difference_with_zero=0.0000 |

来源：

- `QXS-E2-PAC-ALIGNED-HAAR-CANCELLATION-V1-SCENE4`：`test_result/pac_input_cancellation_v1/single_scene_20261117/summary.json`；checkpoint `models/e2_scratch_base_anchor_pac_v1/seed42/epoch_016.pth`。

## QXS-E2-PAC-MAMBA-V1-SMOKE

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-PAC-MAMBA-V1-SEED42-SMOKE | E2 + PAC-Mamba; unchanged intensity + 10*gradient + 0.1*reconstruction | engineering_smoke | val@0 | 1 | loss=0.8553 |

来源：

- `E2-PAC-MAMBA-V1-SEED42-SMOKE`：`models/e2_pac_mamba_smoke_v1/seed42/dry_run_diagnostics.json`；checkpoint `models/lwpd_qxs_ablation_v1/e2_seed42/epoch_016.pth`。

## QXS-E2-PAC-MAMBA-V1-TRAIN8000

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-PAC-MAMBA-V1-SEED42-TRAIN8000 | E2 + PAC-Mamba; unchanged intensity + 10*gradient + 0.1*reconstruction | training_complete | train@0 | 8000 | train_loss=0.7069, val_loss=0.6961 |

来源：

- `E2-PAC-MAMBA-V1-SEED42-TRAIN8000`：`models/e2_pac_mamba_v1/seed42/history.csv`；checkpoint `models/e2_pac_mamba_v1/seed42/epoch_012.pth`。

## QXS-E2-PAC-OVERSHOOT-V1-TRAIN

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-PAC-OVERSHOOT-V1-SEED42-TRAIN | E2 + PAC two-epoch warmup with L2 Sobel overshoot penalty (2.5) | training_complete | train@0 | 8000 | train_loss=0.7505, val_loss=0.7362, train_overshoot=0.0086, val_overshoot=0.0074 |

来源：

- `E2-PAC-OVERSHOOT-V1-SEED42-TRAIN`：`models/e2_pac_overshoot_v1/seed42/history.csv`；checkpoint `models/e2_pac_overshoot_v1/seed42/epoch_002.pth`。

## QXS-E2-POLARITY-V1-SEED42-VAL1000-REFERENCE-FMI

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-E2-POLARITY-V1-SEED42-VAL1000-REFERENCE-FMI-e2 | E2 光学 Base（已有） | development_validation | val@0 | 1000 | 0.6888 | 0.8158 | 0.0189 | 0.5280 | 0.5873 | 0.3933 |
| QXS-E2-POLARITY-V1-SEED42-VAL1000-REFERENCE-FMI-polarity | LWPD E2 + 亮暗极性解耦边界融合 | development_validation | val@0 | 1000 | 0.2445 | 0.4100 | 0.0123 | 0.4749 | 0.5877 | 0.1495 |

来源：

- `QXS-E2-POLARITY-V1-SEED42-VAL1000-REFERENCE-FMI-e2`：`test_result/e2_scratch_polarity_v1/val1000/seed42/e2/metrics.csv`；checkpoint `models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。
- `QXS-E2-POLARITY-V1-SEED42-VAL1000-REFERENCE-FMI-polarity`：`test_result/e2_scratch_polarity_v1/val1000/seed42/polarity/metrics.csv`；checkpoint `models/e2_scratch_polarity_v1/seed42/epoch_016.pth`。

## QXS-E2-RCCR-L2-V1-SEED42-VAL1000-REFERENCE-FMI

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-E2-RCCR-L2-V1-SEED42-VAL1000-REFERENCE-FMI-e2 | E2 光学 Base（已有） | development_validation | val@0 | 1000 | 0.6888 | 0.8158 | 0.0189 | 0.5280 | 0.5867 | 0.3933 |
| QXS-E2-RCCR-L2-V1-SEED42-VAL1000-REFERENCE-FMI-f0 | RCCR-L2 校正前 | development_validation | val@0 | 1000 | 0.6998 | 0.8163 | 0.0348 | 0.5259 | 0.5853 | 0.3926 |
| QXS-E2-RCCR-L2-V1-SEED42-VAL1000-REFERENCE-FMI-fused | RCCR-L2 校正后 | development_validation | val@0 | 1000 | 0.6986 | 0.8164 | 0.0347 | 0.5259 | 0.5853 | 0.3925 |

来源：

- `QXS-E2-RCCR-L2-V1-SEED42-VAL1000-REFERENCE-FMI-e2`：`test_result/e2_scratch_rccr_l2_v1/val1000/seed42/e2/metrics.csv`；checkpoint `models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。
- `QXS-E2-RCCR-L2-V1-SEED42-VAL1000-REFERENCE-FMI-f0`：`test_result/e2_scratch_rccr_l2_v1/val1000/seed42/f0/metrics.csv`；checkpoint `models/e2_scratch_rccr_l2_v1/seed42/epoch_016.pth`。
- `QXS-E2-RCCR-L2-V1-SEED42-VAL1000-REFERENCE-FMI-fused`：`test_result/e2_scratch_rccr_l2_v1/val1000/seed42/fused/metrics.csv`；checkpoint `models/e2_scratch_rccr_l2_v1/seed42/epoch_016.pth`。

## QXS-E2-S2-REGIONS-FIXED4-20261113

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-E2-S2-REGIONS-FIXED4-20261113 | frozen_e2_s2_regional_response_v1 | diagnostic_fixed_subset | val@0 | 4 | regions_analyzed=64.0000, sar_gain_higher_fraction=0.5781 |

来源：

- `QXS-E2-S2-REGIONS-FIXED4-20261113`：`test_result/s2_region_diag_v1/fixed4_tile64/diagnosis.json`；checkpoint `models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。

## QXS-E2-SAMPLING-ORIGIN-HOLDOUT-V1-VAL8-THREE-DIRECTIONS-REF-FMI

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-E2-SAMPLING-ORIGIN-HOLDOUT-V1-VAL8-THREE-DIRECTIONS-REF-FMI-ZERO | Frozen E2 single sampling origin | diagnostic_fixed_subset | val@0 | 8 | 0.7410 | 0.7929 | 0.0329 | 0.5229 | 0.5638 | 0.3303 |
| QXS-E2-SAMPLING-ORIGIN-HOLDOUT-V1-VAL8-THREE-DIRECTIONS-REF-FMI-FOUR_ORIGINS | Frozen E2 four equally weighted sampling origins | diagnostic_fixed_subset | val@0 | 8 | 0.7430 | 0.7935 | 0.0336 | 0.5229 | 0.5625 | 0.3326 |
| QXS-E2-SAMPLING-ORIGIN-HOLDOUT-V1-VAL8-THREE-DIRECTIONS-REF-FMI-MATCHED_SMOOTHING | Frozen E2 equal-output-RMS smoothing | diagnostic_fixed_subset | val@0 | 8 | 0.7383 | 0.7926 | 0.0212 | 0.5248 | 0.5645 | 0.3315 |

来源：

- `QXS-E2-SAMPLING-ORIGIN-HOLDOUT-V1-VAL8-THREE-DIRECTIONS-REF-FMI-ZERO`：`test_result/e2_sampling_origin_holdout_v1/val8_xyz_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-SAMPLING-ORIGIN-HOLDOUT-V1-VAL8-THREE-DIRECTIONS-REF-FMI-FOUR_ORIGINS`：`test_result/e2_sampling_origin_holdout_v1/val8_xyz_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-SAMPLING-ORIGIN-HOLDOUT-V1-VAL8-THREE-DIRECTIONS-REF-FMI-MATCHED_SMOOTHING`：`test_result/e2_sampling_origin_holdout_v1/val8_xyz_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-SAMPLING-ORIGIN-V1-VAL8-REF-FMI

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-E2-SAMPLING-ORIGIN-V1-VAL8-REF-FMI-ZERO | Frozen E2 single sampling origin | diagnostic_fixed_subset | val@0 | 8 | 0.7312 | 0.7848 | 0.0331 | 0.5263 | 0.5715 | 0.3347 |
| QXS-E2-SAMPLING-ORIGIN-V1-VAL8-REF-FMI-FOUR_ORIGINS | Frozen E2 four equally weighted sampling origins | diagnostic_fixed_subset | val@0 | 8 | 0.7247 | 0.7859 | 0.0334 | 0.5263 | 0.5700 | 0.3370 |
| QXS-E2-SAMPLING-ORIGIN-V1-VAL8-REF-FMI-MATCHED_SMOOTHING | Frozen E2 equal-output-RMS smoothing | diagnostic_fixed_subset | val@0 | 8 | 0.7264 | 0.7841 | 0.0214 | 0.5282 | 0.5723 | 0.3358 |

来源：

- `QXS-E2-SAMPLING-ORIGIN-V1-VAL8-REF-FMI-ZERO`：`test_result/cfao_sampling_origin_v1/sampling_val8_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-SAMPLING-ORIGIN-V1-VAL8-REF-FMI-FOUR_ORIGINS`：`test_result/cfao_sampling_origin_v1/sampling_val8_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-SAMPLING-ORIGIN-V1-VAL8-REF-FMI-MATCHED_SMOOTHING`：`test_result/cfao_sampling_origin_v1/sampling_val8_20261117/registry_metrics.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-SAR-DIRECTIONAL-OVERSHOOT-VAL12-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-E2-SAR-DIRECTIONAL-OVERSHOOT-VAL12-v1 | 方向收缩版存档图轻微过冲分解 | diagnostic_fixed_subset | val@0 | 12 | — | — | — | — | — | — |

来源：

- `QXS-E2-SAR-DIRECTIONAL-OVERSHOOT-VAL12-v1`：`test_result/e2_sar_directional_overshoot_val12_v1/summary.csv`；checkpoint `models/e2_scratch_sar_directional_v1/seed42/epoch_016.pth`。

## QXS-E2-SAR-DIRECTIONAL-V1-SEED42-VAL1000-REFERENCE-FMI

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-E2-SAR-DIRECTIONAL-V1-SEED42-VAL1000-REFERENCE-FMI-e2 | E2 光学 Base（已有） | development_validation | val@0 | 1000 | 0.6888 | 0.8158 | 0.0189 | 0.5280 | 0.5867 | 0.3933 |
| QXS-E2-SAR-DIRECTIONAL-V1-SEED42-VAL1000-REFERENCE-FMI-directional | LWPD E2 + SAR 方向子带收缩 | development_validation | val@0 | 1000 | 0.6886 | 0.8160 | 0.0217 | 0.5286 | 0.5873 | 0.3934 |
| QXS-E2-SAR-DIRECTIONAL-V1-SEED42-VAL1000-REFERENCE-FMI-mean | 同权重，SAR 子带均匀收缩 | development_validation | val@0 | 1000 | 0.6885 | 0.8160 | 0.0214 | 0.5287 | 0.5873 | 0.3935 |
| QXS-E2-SAR-DIRECTIONAL-V1-SEED42-VAL1000-REFERENCE-FMI-off | 同权重，关闭 SAR 子带收缩 | development_validation | val@0 | 1000 | 0.6898 | 0.8160 | 0.0232 | 0.5285 | 0.5873 | 0.3932 |

来源：

- `QXS-E2-SAR-DIRECTIONAL-V1-SEED42-VAL1000-REFERENCE-FMI-e2`：`test_result/e2_scratch_sar_directional_v1/val1000/seed42/e2/metrics.csv`；checkpoint `models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。
- `QXS-E2-SAR-DIRECTIONAL-V1-SEED42-VAL1000-REFERENCE-FMI-directional`：`test_result/e2_scratch_sar_directional_v1/val1000/seed42/directional/metrics.csv`；checkpoint `models/e2_scratch_sar_directional_v1/seed42/epoch_016.pth`。
- `QXS-E2-SAR-DIRECTIONAL-V1-SEED42-VAL1000-REFERENCE-FMI-mean`：`test_result/e2_scratch_sar_directional_v1/val1000/seed42/mean/metrics.csv`；checkpoint `models/e2_scratch_sar_directional_v1/seed42/epoch_016.pth`。
- `QXS-E2-SAR-DIRECTIONAL-V1-SEED42-VAL1000-REFERENCE-FMI-off`：`test_result/e2_scratch_sar_directional_v1/val1000/seed42/off/metrics.csv`；checkpoint `models/e2_scratch_sar_directional_v1/seed42/epoch_016.pth`。

## QXS-E2-SCRATCH-BASE-ANCHOR-PAC-V1-TRAIN8000

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-SCRATCH-BASE-ANCHOR-PAC-V1-SEED42-TRAIN8000 | Scratch optical Base anchored LWPD E2 with PAC-Mamba Detail | training_complete | train@0 | 8000 | train_loss=0.7508, val_loss=0.7521 |

来源：

- `E2-SCRATCH-BASE-ANCHOR-PAC-V1-SEED42-TRAIN8000`：`models/e2_scratch_base_anchor_pac_v1/seed42/history.csv`；checkpoint `models/e2_scratch_base_anchor_pac_v1/seed42/epoch_016.pth`。

## QXS-E2-SCRATCH-BASE-ANCHOR-V1-TRAIN8000

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-SCRATCH-BASE-ANCHOR-V1-SEED42-TRAIN8000 | Scratch LWPD E2 with optical Base anchor and zero-initialized correction | training_complete | train@0 | 8000 | train_loss=0.7554, val_loss=0.8051 |

来源：

- `E2-SCRATCH-BASE-ANCHOR-V1-SEED42-TRAIN8000`：`models/e2_scratch_base_anchor_v1/seed42/history.csv`；checkpoint `models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。

## QXS-E2-SCRATCH-LOCAL-CORRESPONDENCE-PILOT-V1-TRAIN1024

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-SCRATCH-LOCAL-CORRESPONDENCE-PILOT-V1-SEED42-TRAIN1024 | 光学 Base E2 + 可拒配的局部 Detail 交换 | training_complete | train@0 | 1024 | train_loss=1.0414, val_loss=1.0024 |

来源：

- `E2-SCRATCH-LOCAL-CORRESPONDENCE-PILOT-V1-SEED42-TRAIN1024`：`models/e2_scratch_local_correspondence_pilot_v1/seed42/history.csv`；checkpoint `models/e2_scratch_local_correspondence_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-SCRATCH-ORIGINAL-LOSS-PILOT-V1-TRAIN1024

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-SCRATCH-ORIGINAL-LOSS-PILOT-V1-SEED42-TRAIN1024 | Scratch optical Base anchored LWPD E2, matched original-loss pilot | training_complete | train@0 | 1024 | train_loss=1.0653, val_loss=1.0449 |

来源：

- `E2-SCRATCH-ORIGINAL-LOSS-PILOT-V1-SEED42-TRAIN1024`：`models/e2_scratch_original_loss_pilot_v1/seed42/history.csv`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-SCRATCH-POLARITY-V1-TRAIN8000

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-SCRATCH-POLARITY-V1-SEED42-TRAIN8000 | LWPD E2 + 亮暗极性解耦边界融合 | training_complete | train@0 | 8000 | train_loss=0.0222, val_loss=0.0224, val_common_same_fraction=0.0179, val_common_reversed_fraction=0.0182, val_optical_private_fraction=0.0678, val_sar_private_fraction=0.0830, val_common_sar_weight=0.6351, val_sar_private_gain=1.2841, val_reversed_polarity_accuracy=0.9330, val_reversed_polarity_pixels=1193.7510, val_reversed_polarity_correct_pixels=1109.3790, val_solver_relative_residual=0.0012, val_field_reconstruction_error=0.0241, val_out_of_range_fraction=0.0094, val_boundary_loss=0.0118, val_coarse_loss=0.0208, val_range_loss=0.0007 |

来源：

- `E2-SCRATCH-POLARITY-V1-SEED42-TRAIN8000`：`models/e2_scratch_polarity_v1/seed42/history.csv`；checkpoint `models/e2_scratch_polarity_v1/seed42/epoch_016.pth`。

## QXS-E2-SCRATCH-RCCR-L2-V1-TRAIN8000

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-SCRATCH-RCCR-L2-V1-SEED42-TRAIN8000 | LWPD E2 + RCCR-L2 | training_complete | train@0 | 8000 | train_loss=0.7584, val_loss=0.7284 |

来源：

- `E2-SCRATCH-RCCR-L2-V1-SEED42-TRAIN8000`：`models/e2_scratch_rccr_l2_v1/seed42/history.csv`；checkpoint `models/e2_scratch_rccr_l2_v1/seed42/epoch_016.pth`。

## QXS-E2-SCRATCH-SAR-DIRECTIONAL-V1-TRAIN8000

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-SCRATCH-SAR-DIRECTIONAL-V1-SEED42-TRAIN8000 | LWPD E2 + SAR 方向子带收缩 | training_complete | train@0 | 8000 | train_loss=0.7599, val_loss=0.7496, val_sar_support_mean=0.7875, val_sar_attenuation_mean=0.0093 |

来源：

- `E2-SCRATCH-SAR-DIRECTIONAL-V1-SEED42-TRAIN8000`：`models/e2_scratch_sar_directional_v1/seed42/history.csv`；checkpoint `models/e2_scratch_sar_directional_v1/seed42/epoch_016.pth`。

## QXS-E2-SCRATCH-SOURCE-INTERVAL-PILOT-V1-TRAIN1024

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-SCRATCH-SOURCE-INTERVAL-PILOT-V1-SEED42-TRAIN1024 | Scratch optical Base anchored LWPD E2 with a source-specific gradient interval loss, pilot | training_complete | train@0 | 1024 | train_loss=0.9086, val_loss=0.8820 |

来源：

- `E2-SCRATCH-SOURCE-INTERVAL-PILOT-V1-SEED42-TRAIN1024`：`models/e2_scratch_source_interval_pilot_v1/seed42/history.csv`；checkpoint `models/e2_scratch_source_interval_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-SCRATCH-SOURCE-INTERVAL-PILOT-V1-VAL64

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| E2-SCRATCH-SOURCE-INTERVAL-PILOT-V1-SEED42-VAL64 | Scratch optical Base anchored LWPD E2 with a source-specific gradient interval loss, pilot | diagnostic_fixed_subset | val@0 | 64 | 0.6307 | 0.6995 | 0.0917 | 0.5320 | 0.5740 | 0.3348 |
| E2-SCRATCH-ORIGINAL-LOSS-PILOT-V1-SEED42-VAL64 | Scratch optical Base anchored LWPD E2, matched original-loss pilot | diagnostic_fixed_subset | val@0 | 64 | 0.6306 | 0.6984 | 0.1180 | 0.5293 | 0.5713 | 0.3353 |

来源：

- `E2-SCRATCH-SOURCE-INTERVAL-PILOT-V1-SEED42-VAL64`：`test_result/e2_scratch_source_interval_pilot_v1/val64/seed42/metrics.csv`；checkpoint `models/e2_scratch_source_interval_pilot_v1/seed42/epoch_016.pth`。
- `E2-SCRATCH-ORIGINAL-LOSS-PILOT-V1-SEED42-VAL64`：`test_result/e2_scratch_original_loss_pilot_v1/val64/seed42/metrics.csv`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-SCRATCH-SOURCE-INTERVAL-V2-TRAIN8000

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-SCRATCH-SOURCE-INTERVAL-V2-SEED42-TRAIN8000 | Scratch optical Base anchored LWPD E2 with a source-specific gradient interval loss | training_complete | train@0 | 8000 | train_loss=0.6604, val_loss=0.6582 |

来源：

- `E2-SCRATCH-SOURCE-INTERVAL-V2-SEED42-TRAIN8000`：`models/e2_scratch_source_interval_v2/seed42/history.csv`；checkpoint `models/e2_scratch_source_interval_v2/seed42/epoch_016.pth`。

## QXS-E2-SCRATCH-STRICT-OVERSHOOT-V1-TRAIN8000

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-SCRATCH-STRICT-OVERSHOOT-V1-SEED42-TRAIN8000 | E2 光学 Base + 严格过冲惩罚 | training_complete | train@0 | 8000 | train_loss=0.7832, val_loss=0.7946, train_overshoot=0.0092, val_overshoot=0.0134 |

来源：

- `E2-SCRATCH-STRICT-OVERSHOOT-V1-SEED42-TRAIN8000`：`models/e2_scratch_strict_overshoot_v1/seed42/history.csv`；checkpoint `models/e2_scratch_strict_overshoot_v1/seed42/epoch_016.pth`。

## QXS-E2-SCRATCH-STRUCT-JOINT-V1-TRAIN8000

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-SCRATCH-STRUCT-JOINT-V1-SEED42-TRAIN8000 | Scratch LWPD E2 with structure-guided Base–Detail joint fusion | training_complete | train@0 | 8000 | train_loss=0.7750, val_loss=0.7490 |

来源：

- `E2-SCRATCH-STRUCT-JOINT-V1-SEED42-TRAIN8000`：`models/e2_scratch_struct_joint_v1/seed42/history.csv`；checkpoint `models/e2_scratch_struct_joint_v1/seed42/epoch_016.pth`。

## QXS-E2-SCRATCH-TWOSTAGE-V1-TRAIN8000

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| E2-SCRATCH-TWOSTAGE-V1-SEED42-TRAIN8000 | Scratch LWPD E2, 4 reconstruction + 12 fusion epochs, original losses | training_complete | train@0 | 8000 | train_loss=0.8787, val_loss=0.8489 |

来源：

- `E2-SCRATCH-TWOSTAGE-V1-SEED42-TRAIN8000`：`models/e2_scratch_twostage_v1/seed42/history.csv`；checkpoint `models/e2_scratch_twostage_v1/seed42/epoch_016.pth`。

## QXS-E2-SOURCE-INTERVAL-V2-SEED42-VAL1000-REFERENCE-FMI

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-E2-SOURCE-INTERVAL-V2-SEED42-VAL1000-REFERENCE-FMI-e2 | E2 光学 Base（已有） | development_validation | val@0 | 1000 | 0.6888 | 0.8158 | 0.0189 | 0.5280 | 0.5863 | 0.3933 |
| QXS-E2-SOURCE-INTERVAL-V2-SEED42-VAL1000-REFERENCE-FMI-interval | E2 + 源特异梯度区间 v2 | development_validation | val@0 | 1000 | 0.6617 | 0.8178 | 0.0386 | 0.5241 | 0.5877 | 0.3861 |

来源：

- `QXS-E2-SOURCE-INTERVAL-V2-SEED42-VAL1000-REFERENCE-FMI-e2`：`test_result/e2_scratch_source_interval_v2/val1000/seed42/e2/metrics.csv`；checkpoint `models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。
- `QXS-E2-SOURCE-INTERVAL-V2-SEED42-VAL1000-REFERENCE-FMI-interval`：`test_result/e2_scratch_source_interval_v2/val1000/seed42/interval/metrics.csv`；checkpoint `models/e2_scratch_source_interval_v2/seed42/epoch_016.pth`。

## QXS-E2-SOURCE-INTERVAL-V2-TOLERANCE-VAL64-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-E2-SOURCE-INTERVAL-V2-TOLERANCE-VAL64-v1 | E2 梯度区间v2固定val64过冲分解 | diagnostic_fixed_subset | val@0 | 64 | — | — | — | — | — | — |

来源：

- `QXS-E2-SOURCE-INTERVAL-V2-TOLERANCE-VAL64-v1`：`test_result/e2_source_interval_tolerance_val64_v1/summary.csv`；checkpoint `models/e2_scratch_source_interval_v2/seed42/epoch_016.pth`。

## QXS-E2-STG-NOTRAIN-V1-FIXED-VAL8-PROBE-RANK

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-E2-STG-NOTRAIN-V1-FIXED-VAL8-PROBE-RANK-MATCHED-2HOP | STG two-hop paths and length-bin-preserving null | diagnostic_fixed_subset | val@0 | 8 | missing_probes=3.0000, missing_with_candidates=2.0000, missing_supported_by_graph=0.0000, negative_probes_supported_by_graph=0.0000, point_hits_at_1_false_link=1.0000, graph_hits_at_1_false_link=1.0000 |
| QXS-E2-STG-NOTRAIN-V1-FIXED-VAL8-PROBE-RANK-MATCHED-6HOP | STG six-hop paths and length-bin-preserving null | diagnostic_fixed_subset | val@0 | 8 | missing_probes=3.0000, missing_with_candidates=2.0000, missing_supported_by_graph=0.0000, negative_probes_supported_by_graph=3.0000, point_hits_at_1_false_link=1.0000, graph_hits_at_1_false_link=0.0000 |
| QXS-E2-STG-NOTRAIN-V1-FIXED-VAL8-PROBE-RANK-LOCAL-6HOP | STG six-hop paths and degree-preserving local null | diagnostic_fixed_subset | val@0 | 8 | missing_probes=3.0000, missing_with_candidates=2.0000, missing_supported_by_graph=0.0000, negative_probes_supported_by_graph=3.0000, point_hits_at_1_false_link=1.0000, graph_hits_at_1_false_link=0.0000 |

来源：

- `QXS-E2-STG-NOTRAIN-V1-FIXED-VAL8-PROBE-RANK-MATCHED-2HOP`：`test_result/stg_no_train_v1/val8/matched_2hop/summary.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-STG-NOTRAIN-V1-FIXED-VAL8-PROBE-RANK-MATCHED-6HOP`：`test_result/stg_no_train_v1/val8/matched_6hop/summary.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。
- `QXS-E2-STG-NOTRAIN-V1-FIXED-VAL8-PROBE-RANK-LOCAL-6HOP`：`test_result/stg_no_train_v1/val8/local_6hop/summary.json`；checkpoint `models/e2_scratch_original_loss_pilot_v1/seed42/epoch_016.pth`。

## QXS-E2-STRICT-OVERSHOOT-V1-SEED42-VAL1000-REFERENCE-FMI

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-E2-STRICT-OVERSHOOT-V1-SEED42-VAL1000-REFERENCE-FMI-e2 | E2 光学 Base（已有） | development_validation | val@0 | 1000 | 0.6888 | 0.8158 | 0.0189 | 0.5280 | 0.5876 | 0.3933 |
| QXS-E2-STRICT-OVERSHOOT-V1-SEED42-VAL1000-REFERENCE-FMI-strict_overshoot | E2 光学 Base + 严格过冲惩罚 | development_validation | val@0 | 1000 | 0.6899 | 0.8149 | 0.0385 | 0.5274 | 0.5871 | 0.3961 |

来源：

- `QXS-E2-STRICT-OVERSHOOT-V1-SEED42-VAL1000-REFERENCE-FMI-e2`：`test_result/e2_scratch_strict_overshoot_v1/val1000/seed42/e2/metrics.csv`；checkpoint `models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。
- `QXS-E2-STRICT-OVERSHOOT-V1-SEED42-VAL1000-REFERENCE-FMI-strict_overshoot`：`test_result/e2_scratch_strict_overshoot_v1/val1000/seed42/strict_overshoot/metrics.csv`；checkpoint `models/e2_scratch_strict_overshoot_v1/seed42/epoch_016.pth`。

## QXS-LWPD-E2-LLCTX-V1-SMOKE

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| LWPD-E2-LLCTX-V1-SEED42-SMOKE | E2 + LL-context; unchanged intensity + 10*gradient + 0.1*reconstruction | engineering_smoke | val@0 | 1 | loss=0.8553 |

来源：

- `LWPD-E2-LLCTX-V1-SEED42-SMOKE`：`models/lwpd_e2_llctx_smoke_v1/seed42/dry_run_diagnostics.json`；checkpoint `models/lwpd_qxs_ablation_v1/e2_seed42/epoch_016.pth`。

## QXS-LWPD-E2-LLCTX-V1-TRAIN8000

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| LWPD-E2-LLCTX-V1-SEED42-TRAIN8000 | E2 + LL-context; unchanged intensity + 10*gradient + 0.1*reconstruction | training_complete | train@0 | 8000 | train_loss=0.6993, val_loss=0.6873 |

来源：

- `LWPD-E2-LLCTX-V1-SEED42-TRAIN8000`：`models/lwpd_e2_llctx_v1/seed42/history.csv`；checkpoint `models/lwpd_e2_llctx_v1/seed42/epoch_012.pth`。

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

## QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-ROUTES-VAL64-DENSE

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-ROUTES-VAL64-DENSE-dense | S2 pilot dense | development_validation | val@0 | 64 | route_entropy=1.0938, cos_opt_sar=0.0107, cos_opt_col=0.1477, cos_sar_col=0.1447, optical_selected_fraction=1.0000, optical_top1_fraction=0.0000, optical_weight_mean=0.3001, sar_selected_fraction=1.0000, sar_top1_fraction=0.0000, sar_weight_mean=0.3223, cooperative_selected_fraction=1.0000, cooperative_top1_fraction=1.0000, cooperative_weight_mean=0.3776 |

来源：

- `QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-ROUTES-VAL64-DENSE-dense`：`test_result/s2_expert_pilot_v2/dense_seed42_routes_val64/summary.json`；checkpoint `models/s2_expert_pilot_v2/dense_seed42/final.pth`。

## QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-ROUTES-VAL64-TOP2

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-ROUTES-VAL64-TOP2-top2 | S2 pilot top2 | development_validation | val@0 | 64 | route_entropy=0.6916, cos_opt_sar=0.0065, cos_opt_col=0.0596, cos_sar_col=0.0963, optical_selected_fraction=0.0641, optical_top1_fraction=0.0000, optical_weight_mean=0.0304, sar_selected_fraction=0.9359, sar_top1_fraction=0.0000, sar_weight_mean=0.4421, cooperative_selected_fraction=1.0000, cooperative_top1_fraction=1.0000, cooperative_weight_mean=0.5274 |

来源：

- `QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-ROUTES-VAL64-TOP2-top2`：`test_result/s2_expert_pilot_v2/top2_seed42_routes_val64/summary.json`；checkpoint `models/s2_expert_pilot_v2/top2_seed42/final.pth`。

## QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-TRAIN-CONV

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-TRAIN-CONV-conv | S2 pilot conv | training_complete | train@0 | 1024 | train_loss=0.8120, val_loss=0.7742 |

来源：

- `QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-TRAIN-CONV-conv`：`models/s2_expert_pilot_v2/conv_seed42/history.csv`；checkpoint `models/s2_expert_pilot_v2/conv_seed42/final.pth`。

## QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-TRAIN-DENSE

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-TRAIN-DENSE-dense | S2 pilot dense | training_complete | train@0 | 1024 | train_loss=0.7662, val_loss=0.7454 |

来源：

- `QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-TRAIN-DENSE-dense`：`models/s2_expert_pilot_v2/dense_seed42/history.csv`；checkpoint `models/s2_expert_pilot_v2/dense_seed42/final.pth`。

## QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-TRAIN-E2_FT

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-TRAIN-E2_FT-e2_ft | S2 pilot e2_ft | training_complete | train@0 | 1024 | train_loss=0.7211, val_loss=0.7112 |

来源：

- `QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-TRAIN-E2_FT-e2_ft`：`models/s2_expert_pilot_v2/e2_ft_seed42/history.csv`；checkpoint `models/s2_expert_pilot_v2/e2_ft_seed42/final.pth`。

## QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-TRAIN-TOP2

| 实验 ID | 方法 | 状态 | 数据切分 | N | 指标 |
|---|---|---|---|---:|---|
| QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-TRAIN-TOP2-top2 | S2 pilot top2 | training_complete | train@0 | 1024 | train_loss=0.7619, val_loss=0.7416 |

来源：

- `QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-TRAIN-TOP2-top2`：`models/s2_expert_pilot_v2/top2_seed42/history.csv`；checkpoint `models/s2_expert_pilot_v2/top2_seed42/final.pth`。

## QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-VAL64-CONV

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-VAL64-CONV-conv | S2 pilot conv | development_validation | val@0 | 64 | — | — | — | — | — | — |

来源：

- `QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-VAL64-CONV-conv`：`test_result/s2_expert_pilot_v2/conv_seed42_val64/metrics.csv`；checkpoint `models/s2_expert_pilot_v2/conv_seed42/final.pth`。

## QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-VAL64-DENSE

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-VAL64-DENSE-dense | S2 pilot dense | development_validation | val@0 | 64 | — | — | — | — | — | — |

来源：

- `QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-VAL64-DENSE-dense`：`test_result/s2_expert_pilot_v2/dense_seed42_val64/metrics.csv`；checkpoint `models/s2_expert_pilot_v2/dense_seed42/final.pth`。

## QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-VAL64-E2_FT

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-VAL64-E2_FT-e2_ft | S2 pilot e2_ft | development_validation | val@0 | 64 | — | — | — | — | — | — |

来源：

- `QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-VAL64-E2_FT-e2_ft`：`test_result/s2_expert_pilot_v2/e2_ft_seed42_val64/metrics.csv`；checkpoint `models/s2_expert_pilot_v2/e2_ft_seed42/final.pth`。

## QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-VAL64-TOP2

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-VAL64-TOP2-top2 | S2 pilot top2 | development_validation | val@0 | 64 | — | — | — | — | — | — |

来源：

- `QXS-S2-EXPERT-PILOT-V2-SEED42-5329e07c38db-VAL64-TOP2-top2`：`test_result/s2_expert_pilot_v2/top2_seed42_val64/metrics.csv`；checkpoint `models/s2_expert_pilot_v2/top2_seed42/final.pth`。

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
| LWPD-E2-QXS-TEST1000-SEED42 | LWPD E2: fixed Haar with directional context gating | formal_test_supplement | test@0 | 1000 | 0.7011 | 0.7279 | 0.0660 | 0.5273 | 0.5718 | 0.3982 |
| E2-PAC-MAMBA-V1-SEED42-TEST1000 | E2 + PAC-Mamba; unchanged intensity + 10*gradient + 0.1*reconstruction | formal_test_supplement | test@0 | 1000 | 0.7065 | 0.7314 | 0.0797 | 0.5258 | 0.5697 | 0.4048 |

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
- `LWPD-E2-QXS-TEST1000-SEED42`：`test_result/lwpd_qxs_full_v1/e2_seed42_test1000/metrics.csv`；checkpoint `models/lwpd_qxs_ablation_v1/e2_seed42/epoch_016.pth`。
- `E2-PAC-MAMBA-V1-SEED42-TEST1000`：`test_result/e2_pac_mamba_v1/test1000/seed42/metrics.csv`；checkpoint `models/e2_pac_mamba_v1/seed42/epoch_012.pth`。

## QXS-TEST1000-REFERENCE-V2-CANDIDATE-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-TEST1000-REFERENCE-V2-CANDIDATE-v1-wavelet | Wavelet | reference_candidate_test | test@0 | 1000 | 0.3015 | 0.4387 | 0.0020 | 0.4859 | — | 0.1566 |
| QXS-TEST1000-REFERENCE-V2-CANDIDATE-v1-u2fusion | U2Fusion（多曝光权重，零样本） | reference_candidate_test | test@0 | 1000 | 0.4516 | 0.4243 | 0.0000 | 0.5290 | — | 0.2292 |
| QXS-TEST1000-REFERENCE-V2-CANDIDATE-v1-laplacian | Laplacian | reference_candidate_test | test@0 | 1000 | 0.4139 | 0.6960 | 0.0450 | 0.4991 | — | 0.2093 |
| QXS-TEST1000-REFERENCE-V2-CANDIDATE-v1-cddfuse_zero_shot | CDDFuse 零样本（clamp） | reference_candidate_test | test@0 | 1000 | 0.3856 | 0.6730 | 0.0203 | 0.5434 | — | 0.2712 |
| QXS-TEST1000-REFERENCE-V2-CANDIDATE-v1-lwpd_e4 | LWPD E4 | reference_candidate_test | test@0 | 1000 | 0.4381 | 0.8120 | 0.0317 | 0.5268 | — | 0.3992 |
| QXS-TEST1000-REFERENCE-V2-CANDIDATE-v1-lwpd_e2 | LWPD E2 | reference_candidate_test | test@0 | 1000 | 0.4401 | 0.8129 | 0.0200 | 0.5273 | — | 0.3982 |
| QXS-TEST1000-REFERENCE-V2-CANDIDATE-v1-lwpd_e2_scratch | LWPD E2 从头训练（光学 Base，零初始化修正） | reference_candidate_test | test@0 | 1000 | 0.4347 | 0.8130 | 0.0196 | 0.5267 | — | 0.3909 |

来源：

- `QXS-TEST1000-REFERENCE-V2-CANDIDATE-v1-wavelet`：`test_result/qxs_reference_v2_candidate_test1000_v1/wavelet/metrics.csv`。
- `QXS-TEST1000-REFERENCE-V2-CANDIDATE-v1-u2fusion`：`test_result/qxs_reference_v2_candidate_test1000_v1/u2fusion/metrics.csv`；checkpoint `../comparation/U2Fusion-main/model/model.pth`。
- `QXS-TEST1000-REFERENCE-V2-CANDIDATE-v1-laplacian`：`test_result/qxs_reference_v2_candidate_test1000_v1/laplacian/metrics.csv`。
- `QXS-TEST1000-REFERENCE-V2-CANDIDATE-v1-cddfuse_zero_shot`：`test_result/qxs_reference_v2_candidate_test1000_v1/cddfuse_zero_shot/metrics.csv`；checkpoint `models/CDDFuse_IVF.pth`。
- `QXS-TEST1000-REFERENCE-V2-CANDIDATE-v1-lwpd_e4`：`test_result/qxs_reference_v2_candidate_test1000_v1/lwpd_e4/metrics.csv`；checkpoint `D:/#imageInfusion/CDDFuse/models/lwpd_qxs_full_v1/e4_seed42/epoch_016.pth`。
- `QXS-TEST1000-REFERENCE-V2-CANDIDATE-v1-lwpd_e2`：`test_result/qxs_reference_v2_candidate_test1000_v1/lwpd_e2/metrics.csv`；checkpoint `D:/#imageInfusion/CDDFuse/models/lwpd_qxs_ablation_v1/e2_seed42/epoch_016.pth`。
- `QXS-TEST1000-REFERENCE-V2-CANDIDATE-v1-lwpd_e2_scratch`：`test_result/qxs_reference_v2_candidate_test1000_v1/lwpd_e2_scratch/metrics.csv`；checkpoint `D:/#imageInfusion/CDDFuse/models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。

## QXS-TEST1000-REFERENCE-V2-LEGACY-FMI-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| QXS-TEST1000-REFERENCE-V2-LEGACY-FMI-v1-wavelet | Wavelet | reference_candidate_test | test@0 | 1000 | 0.3125 | 0.4387 | 0.0020 | 0.4859 | — | 0.1566 |
| QXS-TEST1000-REFERENCE-V2-LEGACY-FMI-v1-u2fusion | U2Fusion（多曝光权重，零样本） | reference_candidate_test | test@0 | 1000 | 0.3355 | 0.4243 | 0.0000 | 0.5290 | — | 0.2292 |
| QXS-TEST1000-REFERENCE-V2-LEGACY-FMI-v1-laplacian | Laplacian | reference_candidate_test | test@0 | 1000 | 0.4964 | 0.6960 | 0.0450 | 0.4991 | — | 0.2093 |
| QXS-TEST1000-REFERENCE-V2-LEGACY-FMI-v1-cddfuse_zero_shot | CDDFuse 零样本（clamp） | reference_candidate_test | test@0 | 1000 | 0.3974 | 0.6730 | 0.0203 | 0.5434 | — | 0.2712 |
| QXS-TEST1000-REFERENCE-V2-LEGACY-FMI-v1-lwpd_e4 | LWPD E4 | reference_candidate_test | test@0 | 1000 | 0.7028 | 0.8120 | 0.0317 | 0.5268 | — | 0.3992 |
| QXS-TEST1000-REFERENCE-V2-LEGACY-FMI-v1-lwpd_e2 | LWPD E2 | reference_candidate_test | test@0 | 1000 | 0.7011 | 0.8129 | 0.0200 | 0.5273 | — | 0.3982 |
| QXS-TEST1000-REFERENCE-V2-LEGACY-FMI-v1-lwpd_e2_scratch | LWPD E2 从头训练（光学 Base，零初始化修正） | reference_candidate_test | test@0 | 1000 | 0.6783 | 0.8130 | 0.0196 | 0.5267 | — | 0.3909 |

来源：

- `QXS-TEST1000-REFERENCE-V2-LEGACY-FMI-v1-wavelet`：`test_result/qxs_reference_v2_legacy_fmi_test1000_v1/wavelet/metrics.csv`。
- `QXS-TEST1000-REFERENCE-V2-LEGACY-FMI-v1-u2fusion`：`test_result/qxs_reference_v2_legacy_fmi_test1000_v1/u2fusion/metrics.csv`；checkpoint `../comparation/U2Fusion-main/model/model.pth`。
- `QXS-TEST1000-REFERENCE-V2-LEGACY-FMI-v1-laplacian`：`test_result/qxs_reference_v2_legacy_fmi_test1000_v1/laplacian/metrics.csv`。
- `QXS-TEST1000-REFERENCE-V2-LEGACY-FMI-v1-cddfuse_zero_shot`：`test_result/qxs_reference_v2_legacy_fmi_test1000_v1/cddfuse_zero_shot/metrics.csv`；checkpoint `models/CDDFuse_IVF.pth`。
- `QXS-TEST1000-REFERENCE-V2-LEGACY-FMI-v1-lwpd_e4`：`test_result/qxs_reference_v2_legacy_fmi_test1000_v1/lwpd_e4/metrics.csv`；checkpoint `D:/#imageInfusion/CDDFuse/models/lwpd_qxs_full_v1/e4_seed42/epoch_016.pth`。
- `QXS-TEST1000-REFERENCE-V2-LEGACY-FMI-v1-lwpd_e2`：`test_result/qxs_reference_v2_legacy_fmi_test1000_v1/lwpd_e2/metrics.csv`；checkpoint `D:/#imageInfusion/CDDFuse/models/lwpd_qxs_ablation_v1/e2_seed42/epoch_016.pth`。
- `QXS-TEST1000-REFERENCE-V2-LEGACY-FMI-v1-lwpd_e2_scratch`：`test_result/qxs_reference_v2_legacy_fmi_test1000_v1/lwpd_e2_scratch/metrics.csv`；checkpoint `D:/#imageInfusion/CDDFuse/models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。

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
| LWPD-E2-LLCTX-V1-SEED42-VAL1000 | E2 + LL-context; unchanged intensity + 10*gradient + 0.1*reconstruction | development_validation | val@0 | 1000 | 0.7072 | 0.7365 | 0.0820 | 0.5266 | 0.5639 | 0.4072 |
| E2-PAC-MAMBA-V1-SEED42-VAL1000 | E2 + PAC-Mamba; unchanged intensity + 10*gradient + 0.1*reconstruction | development_validation | val@0 | 1000 | 0.7119 | 0.7356 | 0.0782 | 0.5272 | 0.5659 | 0.4074 |
| E2-MATCHED-NO-PAC-V1-SEED42-VAL1000 | E2 matched continuation; PAC frozen with exactly zero output | development_validation | val@0 | 1000 | 0.7140 | 0.7353 | 0.0982 | 0.5261 | 0.5656 | 0.4066 |
| E2-PAC-OVERSHOOT-V1-SEED42-E01-VAL1000 | E2 + PAC with L2 Sobel overshoot penalty (2.5), warmup epoch 1 | development_validation | val@0 | 1000 | 0.7120 | 0.7333 | 0.0882 | 0.5289 | 0.5664 | 0.4053 |
| E2-PAC-OVERSHOOT-V1-SEED42-E02-VAL1000 | E2 + PAC with L2 Sobel overshoot penalty (2.5), warmup epoch 2 | development_validation | val@0 | 1000 | 0.7115 | 0.7337 | 0.0741 | 0.5287 | 0.5671 | 0.4057 |
| E2-SCRATCH-TWOSTAGE-V1-SEED42-VAL1000 | Scratch LWPD E2, 4 reconstruction + 12 fusion epochs, original losses | development_validation | val@0 | 1000 | 0.6748 | 0.7237 | 0.1198 | 0.3542 | 0.4424 | 0.3377 |
| E2-SCRATCH-BASE-ANCHOR-V1-SEED42-VAL1000 | Scratch LWPD E2 with optical Base anchor and zero-initialized correction | development_validation | val@0 | 1000 | 0.6888 | 0.7321 | 0.0601 | 0.5280 | 0.5674 | 0.3933 |
| E2-SCRATCH-BASE-ANCHOR-PAC-V1-SEED42-VAL1000 | Scratch optical Base anchored LWPD E2 with PAC-Mamba Detail | development_validation | val@0 | 1000 | 0.6863 | 0.7329 | 0.1239 | 0.5247 | 0.5660 | 0.3924 |
| E2-SCRATCH-STRUCT-JOINT-V1-SEED42-VAL1000 | Scratch LWPD E2 with structure-guided Base–Detail joint fusion | development_validation | val@0 | 1000 | 0.6712 | 0.7301 | 0.1067 | 0.5246 | 0.5668 | 0.3811 |
| E2-SCRATCH-BASE-ANCHOR-V1-SEED42-VAL1000-EPOCH015 | Scratch LWPD E2 with optical Base anchor and zero-initialized correction | development_validation | val@0 | 1000 | 0.6925 | 0.7309 | 0.1016 | 0.5257 | 0.5669 | 0.3905 |

来源：

- `DIRECT-R3-QXS-VAL1000-EPOCH9-LWPD-REFERENCE`：`test_result/direct_fusion_qxs/qxs_direct_seed42_v1/val_extended/r3_epoch_009/metrics.csv`；checkpoint `models/direct_fusion_qxs/qxs_direct_seed42_v1/r3/epoch_009.pth`。
- `LWPD-E4-QXS-VAL1000-SEED42`：`test_result/lwpd_qxs_full_v1/e4_seed42_val1000/metrics.csv`；checkpoint `models/lwpd_qxs_full_v1/e4_seed42/epoch_016.pth`。
- `LWPD-E2-QXS-VAL1000-SEED42`：`test_result/lwpd_qxs_ablation_v1/val1000/e2_seed42/metrics.csv`；checkpoint `models/lwpd_qxs_ablation_v1/e2_seed42/epoch_016.pth`。
- `LWPD-E2-PRL-QXS-VAL1000-SEED42`：`test_result/lwpd_qxs_prl_v1/e2_prl_seed42_val1000/metrics.csv`；checkpoint `models/lwpd_qxs_prl_v1/e2_prl_seed42/epoch_016.pth`。
- `LWPD-E2-LLCTX-V1-SEED42-VAL1000`：`test_result/lwpd_e2_llctx_v1/val1000/seed42/metrics.csv`；checkpoint `models/lwpd_e2_llctx_v1/seed42/epoch_012.pth`。
- `E2-PAC-MAMBA-V1-SEED42-VAL1000`：`test_result/e2_pac_mamba_v1/val1000/seed42/metrics.csv`；checkpoint `models/e2_pac_mamba_v1/seed42/epoch_012.pth`。
- `E2-MATCHED-NO-PAC-V1-SEED42-VAL1000`：`test_result/e2_matched_no_pac_v1/val1000/seed42/metrics.csv`；checkpoint `models/e2_matched_no_pac_v1/seed42/epoch_012.pth`。
- `E2-PAC-OVERSHOOT-V1-SEED42-E01-VAL1000`：`test_result/e2_pac_overshoot_v1/val1000/epoch_001/metrics.csv`；checkpoint `models/e2_pac_overshoot_v1/seed42/epoch_001.pth`。
- `E2-PAC-OVERSHOOT-V1-SEED42-E02-VAL1000`：`test_result/e2_pac_overshoot_v1/val1000/epoch_002/metrics.csv`；checkpoint `models/e2_pac_overshoot_v1/seed42/epoch_002.pth`。
- `E2-SCRATCH-TWOSTAGE-V1-SEED42-VAL1000`：`test_result/e2_scratch_twostage_v1/val1000/seed42/metrics.csv`；checkpoint `models/e2_scratch_twostage_v1/seed42/epoch_016.pth`。
- `E2-SCRATCH-BASE-ANCHOR-V1-SEED42-VAL1000`：`test_result/e2_scratch_base_anchor_v1/val1000/seed42/metrics.csv`；checkpoint `models/e2_scratch_base_anchor_v1/seed42/epoch_016.pth`。
- `E2-SCRATCH-BASE-ANCHOR-PAC-V1-SEED42-VAL1000`：`test_result/e2_scratch_base_anchor_pac_v1/val1000/seed42/metrics.csv`；checkpoint `models/e2_scratch_base_anchor_pac_v1/seed42/epoch_016.pth`。
- `E2-SCRATCH-STRUCT-JOINT-V1-SEED42-VAL1000`：`test_result/e2_scratch_struct_joint_v1/val1000/seed42/metrics.csv`；checkpoint `models/e2_scratch_struct_joint_v1/seed42/epoch_016.pth`。
- `E2-SCRATCH-BASE-ANCHOR-V1-SEED42-VAL1000-EPOCH015`：`test_result/e2_scratch_base_anchor_v1/val1000/seed42_epoch_015/metrics.csv`；checkpoint `models/e2_scratch_base_anchor_v1/seed42/epoch_015.pth`。

## QXS-VAL1000-E2-GENERAL-LOSS-v1

| 实验 ID | 方法 | 状态 | 数据切分 | N | FMI↑ | QAB/F↑ | NAB/F↓ | SSIM↑ | MS-SSIM↑ | VIF↑ |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| E2-GENERAL-LOSS-V1-B-SEED42-VAL1000 | E2 loss b: grad=10.0, structure=0.0 | development_validation | val@0 | 1000 | 0.7110 | 0.7326 | 0.0657 | 0.5292 | 0.5665 | 0.4025 |
| E2-GENERAL-LOSS-V1-S-SEED42-VAL1000 | E2 loss s: grad=10.0, structure=0.5 | development_validation | val@0 | 1000 | 0.7164 | 0.7322 | 0.0780 | 0.5309 | 0.5678 | 0.4034 |
| E2-G1-V1-G1-SEED42-VAL1000 | E2-G1 PA directional gradient (10), reconstruction (0.1) | development_validation | val@0 | 1000 | 0.3623 | 0.3244 | 0.0266 | 0.5298 | 0.5490 | 0.4953 |

来源：

- `E2-GENERAL-LOSS-V1-B-SEED42-VAL1000`：`test_result/e2_general_loss_v1/val1000/b_seed42/metrics.csv`；checkpoint `models/e2_general_loss_v1/b_seed42/epoch_016.pth`。
- `E2-GENERAL-LOSS-V1-S-SEED42-VAL1000`：`test_result/e2_general_loss_v1/val1000/s_seed42/metrics.csv`；checkpoint `models/e2_general_loss_v1/s_seed42/epoch_016.pth`。
- `E2-G1-V1-G1-SEED42-VAL1000`：`test_result/e2_g1_v1/val1000/g1_seed42/metrics.csv`；checkpoint `models/e2_g1_v1/g1_seed42/epoch_016.pth`。

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
