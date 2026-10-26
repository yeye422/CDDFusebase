# 测试输出目录记录

更新时间：2026-10-22  
来源：`EXPERIMENT_REGISTRY.json`（自动生成，请勿手工修改表格）

## 维护规则

- 每次模型、损失、推理路径、数据切分或评价协议发生变化，必须使用新的输出目录。
- 定量结果生成后先登记 `EXPERIMENT_REGISTRY.json`，再运行维护命令。
- 同一目录可包含多个同步评价路径，例如 `c0_raw/e1_image/e2_feature`。
- Smoke test 使用 `tmp/`，不进入正式结果表；若用于研究结论，必须移入 `test_result/` 并登记。
- 不允许覆盖已有正式结果目录。

维护命令：

```cmd
cd /d D:\#imageInfusion\CDDFuse
.venv\Scripts\python.exe maintain_experiment_index.py --check --render
```

## 已登记测试输出

| 输出目录 | 实验/路径 | 状态 | 数据协议 | N | 可比组 |
|---|---|---|---|---:|---|
| `test_result/pdcf_p2/signed_scale_diagnostic_seed42_val200` | `PDCF-P2-SIGNED-C0-QXS-VAL200`：C0 identity in PDCF signed-scale diagnostic<br>`PDCF-P2-SIGNED-PREDICTABLE-NEG010-QXS-VAL200`：PDCF P2 predictable signed correction alpha=-0.1<br>`PDCF-P2-SIGNED-PREDICTABLE-NEG005-QXS-VAL200`：PDCF P2 predictable signed correction alpha=-0.05<br>`PDCF-P2-SIGNED-RAW-NEG010-QXS-VAL200`：PDCF P2 raw signed correction alpha=-0.1<br>`PDCF-P2-SIGNED-RAW-NEG005-QXS-VAL200`：PDCF P2 raw signed correction alpha=-0.05 | diagnostic_baseline<br>development_positive_conditional | QXS-SAROPT / val@0 | 200 | QXS-VAL-FIRST200-PDCF-P2-SIGNED-SCALE-v1 |
| `test_result/pdcf_p2/scale_diagnostic_seed42_val200` | `PDCF-P2-SCALE-DIAGNOSTIC-QXS-VAL200`：PDCF P2 no-training correction-scale diagnostic | development_negative_terminal | QXS-SAROPT / val@0 | 200 | QXS-VAL-FIRST200-PDCF-P2-SCALE-v1 |
| `test_result/pdcf_p2/matched_pilots_seed42_val200_gradfix` | `PDCF-P2-PREDICTABLE-SINGLEPATH-QXS-VAL200`：PDCF P2 Predictable/Residual single-path pilot<br>`PDCF-P2-RAW-EVIDENCE-SINGLEPATH-QXS-VAL200`：PDCF P2 capacity-matched Raw Evidence single-path control | development_negative | QXS-SAROPT / val@0 | 200 | QXS-VAL-FIRST200-PDCF-P2-MATCHED-v1 |
| `models/pdcf_p2/predictable_residual_seed42_pilot1000` | `PDCF-P2-PREDICTABLE-PILOT1000-SEED42-NAN-INVALID`：PDCF P2 predictable-residual single-path pilot before zero-gradient fix | invalid | QXS-SAROPT / train+val@0 | 1200 | QXS-PDCF-P2-PILOT1000-INVALID-v1 |
| `models/pdcf_p1/predictability_seed42` | `PDCF-P1-PREDICTABILITY-QXS-TRAIN-SEED42`：PDCF P1 frozen-C0 bidirectional cross-modal predictor | development_positive | QXS-SAROPT / train+val@0 | 9000 | QXS-PDCF-P1-PREDICTABILITY-v1 |
| `test_result/pdcf_p1/predictability_seed42_val200` | `PDCF-P1-RESIDUAL-AUDIT-QXS-VAL200`：PDCF P1 prediction-residual intervention audit | development_positive_conditional | QXS-SAROPT / val@0 | 200 | QXS-VAL-FIRST200-PDCF-P1-INTERVENTION-v1 |
| `test_result/reliability_v4/B0_clamp` | `B0-QXS-TEST1000-PAPER-CLAMP`：B0 original CDDFuse | formal_test | QXS-SAROPT / test@0 | 1000 | QXS-TEST1000-PAPER-CLAMP-v1 |
| `test_result/reliability_v4/B1_v5_seed42` | `B1V5-QXS-TEST1000-PAPER-CLAMP`：B1-v5 noise-suppression baseline | formal_test | QXS-SAROPT / test@0 | 1000 | QXS-TEST1000-PAPER-CLAMP-v1 |
| `test_result/reliability_v5/frequency_minimax_seed42` | `C0-MINIMAX-QXS-TEST1000-PAPER-CLAMP`：C0 frequency-minimax epoch 5 | formal_test | QXS-SAROPT / test@0 | 1000 | QXS-TEST1000-PAPER-CLAMP-v1 |
| `test_result/reliability_v5/confirmation_val800/B0` | `B0-QXS-VAL800-PAPER-CLAMP`：B0 original CDDFuse | confirmation | QXS-SAROPT / val@200 | 800 | QXS-VAL-OFFSET200-LIMIT800-PAPER-CLAMP-v1 |
| `test_result/v21_softmask/spd_nomask_fusion_test1000_c0full` | `SPD-E1IMAGE-QXS-TEST1000-PAPER-CLAMP`：E1 SPD image preprocessing<br>`SPD-E2FEATURE-QXS-TEST1000-PAPER-CLAMP`：E2 SPD feature purification | formal_test | QXS-SAROPT / test@0 | 1000 | QXS-TEST1000-PAPER-CLAMP-v1 |
| `test_result/v21_softmask/e2_adapt_full_seed42_test1000` | `SPD-E2ADAPT-QXS-TEST1000-PAPER-CLAMP`：E2-Adapt full 8000/1000 | formal_test_negative | QXS-SAROPT / test@0 | 1000 | QXS-TEST1000-PAPER-CLAMP-v1 |
| `test_result/reliability_v5/confirmation_val800/B1_v5` | `B1V5-QXS-VAL800-PAPER-CLAMP`：B1-v5 noise-suppression baseline | confirmation | QXS-SAROPT / val@200 | 800 | QXS-VAL-OFFSET200-LIMIT800-PAPER-CLAMP-v1 |
| `test_result/v21_softmask/spd_nomask_on_b0_test1000` | `SPD-E2FEATURE-ON-B0-QXS-TEST1000-PAPER-CLAMP`：B0 + E2 SPD feature purification | formal_test | QXS-SAROPT / test@0 | 1000 | QXS-TEST1000-PAPER-CLAMP-v1 |
| `test_result/v21_softmask/spd_detail_only_on_b0_test1000` | `SPD-DETAILONLY-ON-B0-QXS-TEST1000-PAPER-CLAMP`：B0 + SPD Detail-only (alpha_B=0, alpha_D=1) | formal_test_selected | QXS-SAROPT / test@0 | 1000 | QXS-TEST1000-PAPER-CLAMP-v1 |
| `test_result/v21_softmask/spd_detail_only_retrained_on_b0_test1000` | `SPD-DETAILONLY-RETRAINED-ON-B0-QXS-TEST1000-PAPER-CLAMP`：B0 + retrained SPD Detail-only | formal_test_main | QXS-SAROPT / test@0 | 1000 | QXS-TEST1000-PAPER-CLAMP-v1 |
| `test_result/v21_softmask/spd_nomask_seed42_internal_test50` | `SPD-NOMASK-SAR-DESPECKLE-INTERNAL50`：SPD no-mask seed42 | internal_confirmation | SAR despeckling filters dataset / internal_test_coordinate_group_5632@0 | 50 | SAR-DESPECKLE-5632STRIPE50-v1 |
| `test_result/v21_softmask/spd_scale_sweep_b0_val1000` | `B0-QXS-VAL1000-SPD-SCALE-SWEEP`：B0 scale-sweep identity<br>`SPD-DETAILONLY-QXS-VAL1000-SCALE-SWEEP`：B0 + SPD Detail-only (alpha_B=0, alpha_D=1) | development_selection<br>development_selected | QXS-SAROPT / val@0 | 1000 | QXS-VAL1000-SPD-SCALE-SWEEP-v1 |
| `test_result/v21_softmask/spd_detail_only_seed42_internal_test50` | `SPD-DETAILONLY-RETRAINED-SAR-DESPECKLE-INTERNAL50`：SPD Detail-only retrained seed42 | internal_confirmation | SAR despeckling filters dataset / internal_test_coordinate_group_5632@0 | 50 | SAR-DESPECKLE-5632STRIPE50-v1 |
| `test_result/v21_softmask/spd_detail_only_retrained_on_b0_val1000` | `SPD-DETAILONLY-RETRAINED-QXS-VAL1000`：B0 + retrained SPD Detail-only | development_comparison | QXS-SAROPT / val@0 | 1000 | QXS-VAL1000-SPD-SCALE-SWEEP-v1 |
| `test_result/v21_softmask/spd_nomask_fusion_val200` | `SPD-C0RAW-QXS-VAL200`：C0 raw path in SPD comparison<br>`SPD-E1IMAGE-QXS-VAL200`：E1 SPD image preprocessing<br>`SPD-E2FEATURE-QXS-VAL200`：E2 SPD feature purification | development | QXS-SAROPT / val@0 | 200 | QXS-VAL-FIRST200-SPD-COMPARISON-v1 |
| `test_result/v21_softmask/spd_detail_only_ml_w010_seed42_internal_test50` | `SPD-DETAILONLY-ML-W010-SAR-DESPECKLE-INTERNAL50`：SPD Detail-only + high-confidence M_L (lambda_L=0.10) | development_validation | SAR despeckling filters dataset / internal_test_coordinate_group_5632@0 | 50 | SAR-DESPECKLE-5632STRIPE50-v1 |
| `test_result/v21_softmask/spd_detail_only_ml_w010_on_b0_val1000` | `SPD-DETAILONLY-ML-W010-QXS-VAL1000`：B0 + SPD Detail-only + high-confidence M_L (lambda_L=0.10) | development_negative | QXS-SAROPT / val@0 | 1000 | QXS-VAL1000-SPD-SCALE-SWEEP-v1 |
| `test_result/v21_softmask/spd_line_gated_detail_val1000` | `SPD-LINEGATE-P025-QXS-VAL1000`：B0 + SPD Detail-only + continuity-filtered line gate (p=0.25)<br>`SPD-LINEGATE-P050-QXS-VAL1000`：B0 + SPD Detail-only + continuity-filtered line gate (p=0.50)<br>`SPD-LINEGATE-P100-QXS-VAL1000`：B0 + SPD Detail-only + continuity-filtered line gate (p=1.00) | development_negative | QXS-SAROPT / val@0 | 1000 | QXS-VAL1000-SPD-SCALE-SWEEP-v1 |
| `test_result/fusion_p1/stable_detail_seed42_val1000` | `P1-STABLE-DETAIL-QXS-VAL1000`：B0 + P1 provenance-preserving stable-SAR Detail router | development_negative | QXS-SAROPT / val@0 | 1000 | QXS-VAL1000-SPD-SCALE-SWEEP-v1 |
| `test_result/fusion_p2a/source_preserving_seed42_val1000` | `P2A-SOURCE-PRESERVING-QXS-VAL1000`：B0 + P2-A source-preserving pseudo-sensing Detail adapter | development_negative | QXS-SAROPT / val@0 | 1000 | QXS-VAL1000-SPD-SCALE-SWEEP-v1 |
| `test_result/fusion_p2a/source_preserving_scale_sweep_val1000` | `P2A-SCALE025-QXS-VAL1000`：B0 + P2-A source-preserving residual (alpha=0.25)<br>`P2A-SCALE050-QXS-VAL1000`：B0 + P2-A source-preserving residual (alpha=0.50)<br>`P2A-SCALE075-QXS-VAL1000`：B0 + P2-A source-preserving residual (alpha=0.75) | development_selected<br>development_negative | QXS-SAROPT / val@0 | 1000 | QXS-VAL1000-SPD-SCALE-SWEEP-v1 |
| `test_result/mffa_v1/phase1_dtypefix_seed42_val1000` | `MFFA-V1-PHASE1-QXS-VAL1000`：MFFA v1 Phase-1 symmetric frequency autoencoder | development_passed | QXS-SAROPT / val@0 | 1000 | MFFA-V1-PHASE1-QXS-VAL1000-v1 |
| `test_result/ocaf_v2/independent_amplitude_seed42_val1000` | `OCAF-V2-INDEPENDENT-AMPLITUDE-QXS-VAL1000`：OCAF-v2 independent additive gates and explicit amplitude | development_negative | QXS-SAROPT / val@0 | 1000 | QXS-VAL1000-SPD-SCALE-SWEEP-v1 |
| `test_result/ocaf_v1/counterfactual_identity_seed42_val1000` | `OCAF-V1-COUNTERFACTUAL-IDENTITY-QXS-VAL1000`：OCAF-v1 optical-identity counterfactual SAR injection | development_negative | QXS-SAROPT / val@0 | 1000 | QXS-VAL1000-SPD-SCALE-SWEEP-v1 |
| `test_result/csr_v2/calibrated_routing_seed42_val1000` | `CSR-V2-CALIBRATED-ROUTING-QXS-VAL1000`：Detail-only SPD + calibrated competitive structure routing v2 | development_negative | QXS-SAROPT / val@0 | 1000 | QXS-VAL1000-SPD-SCALE-SWEEP-v1 |
| `test_result/csr_v1/detail_spd_seed42_val1000` | `CSR-V1-DETAIL-SPD-QXS-VAL1000`：Detail-only SPD + competitive visible-structure routing v1 | development_negative | QXS-SAROPT / val@0 | 1000 | QXS-VAL1000-SPD-SCALE-SWEEP-v1 |
| `test_result/optical_rescue/detail_spd_gradfix_seed42_val1000` | `OPTICAL-RESCUE-DETAIL-SPD-QXS-VAL1000`：Detail-only SPD + feature-space Optical Rescue | development_negative | QXS-SAROPT / val@0 | 1000 | QXS-VAL1000-SPD-SCALE-SWEEP-v1 |
| `test_result/mffa_v1/phase2_seed42_val1000_evalfix` | `MFFA-V1-PHASE2-QXS-VAL1000`：MFFA v1 frequency-factorized provenance-preserving fusion | development_negative | QXS-SAROPT / val@0 | 1000 | QXS-VAL1000-SPD-SCALE-SWEEP-v1 |

## 当前 SPD 主线目录

| 阶段 | 目录 | 说明 |
|---|---|---|
| 原始 SPD 训练 | `models/v21_softmask/spd_nomask_seed42` | Base+Detail SPD |
| 新主模型训练 | `models/v21_softmask/spd_detail_only_seed42` | 训练阶段删除 Base SPD |
| 原始 SPD 去噪确认 | `test_result/v21_softmask/spd_nomask_seed42_internal_test50` | Toronto internal50 |
| 新主模型去噪确认 | `test_result/v21_softmask/spd_detail_only_seed42_internal_test50` | Toronto internal50 |
| B0+完整SPD | `test_result/v21_softmask/spd_nomask_on_b0_test1000` | 正式 test1000 |
| 旧 Detail-only | `test_result/v21_softmask/spd_detail_only_on_b0_test1000` | 完整SPD训练后推理屏蔽Base |
| 新 Detail-only val | `test_result/v21_softmask/spd_detail_only_retrained_on_b0_val1000` | 新模型 validation1000 |
| 新 Detail-only test | `test_result/v21_softmask/spd_detail_only_retrained_on_b0_test1000` | 当前正式主结果 |
| 强度扫描 | `test_result/v21_softmask/spd_scale_sweep_b0_val1000` | 选择 Detail-only |
| 可视化比较 | `test_result/v21_softmask/visual_comparison` | 拼图和放大差分 |
| 五方法14指标汇总 | `test_result/v21_softmask/five_method_test1000_comparison` | 自动聚合正式test1000 |

## 未登记的工程输出

`tmp/` 下的 `*_smoke`、`*_dry_run` 目录只验证接口和单样本前向，不作为研究结果，不得与正式目录合表。
