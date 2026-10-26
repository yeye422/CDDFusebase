# 五方法正式 test1000：14项融合指标

协议：QXS-SAROPT test1000，PaperEvaluator，clamp→uint8。

| 方法 | EN↑ | MI↑ | FMI↑ | PSNR↑ | Qabf↑ | Nabf↓ | AG↑ | SF↑ | SD↑ | CC↑ | SCD↑ | SSIM↑ | MS_SSIM↑ | VIF↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 7.3290 | **4.6584** | 0.6946 | 13.5382 | **0.7138** | 0.1145 | **24.4244** | **44.3073** | **52.5678** | 0.5687 | 1.3866 | 0.5312 | 0.5737 | 0.3948 |
| B0 + SPD | **7.3557** | 3.7005 | 0.6553 | 14.0292 | 0.7015 | 0.0813 | 23.8746 | 43.0382 | 50.9323 | 0.5807 | 1.4951 | **0.5355** | 0.5810 | 0.3554 |
| B0 + Detail-only SPD | 7.3279 | 4.6451 | **0.6954** | 13.5991 | 0.7137 | 0.0946 | 24.3359 | 44.1715 | 52.4209 | 0.5675 | 1.3711 | 0.5323 | 0.5730 | **0.3953** |
| C0-minimax | 7.3023 | 2.7786 | 0.4358 | 14.0612 | 0.5905 | 0.0585 | 17.2336 | 31.7599 | 50.0409 | 0.5991 | 1.5011 | 0.5145 | 0.5931 | 0.2503 |
| C0-minimax + SPD | 7.3076 | 2.7574 | 0.4307 | **14.2545** | 0.6007 | **0.0556** | 17.1121 | 31.3657 | 48.7803 | **0.6094** | **1.5373** | 0.5204 | **0.5985** | 0.2516 |

## 数据来源

- B0：`test_result\reliability_v4\B0_clamp\metrics.csv`
- B0 + SPD：`test_result\v21_softmask\spd_nomask_on_b0_test1000\summary.json`
- B0 + Detail-only SPD：`test_result\v21_softmask\spd_detail_only_retrained_on_b0_test1000\summary.json`
- C0-minimax：`test_result\reliability_v5\frequency_minimax_seed42\metrics.csv`
- C0-minimax + SPD：`test_result\v21_softmask\spd_nomask_fusion_test1000_c0full\summary.json`