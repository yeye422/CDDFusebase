# 融合指标协议

`utils/Evaluator.py` now contains two evaluation profiles:

- `Evaluator`: the unchanged, eight-metric CDDFuse implementation used to
  reproduce the original repository's published values.
- `PaperEvaluator`: the fourteen metrics described in Section 4.3.2 of Wang
  Haolong's thesis.

后续光学–SAR 评测默认使用 `reference_v2_candidate_with_paper_fmi`。该协议采用 `PaperEvaluator.FMI` 的全图 DCT 互信息之和，其余 13 项调用 `ReferenceEvaluator`。参考指标的 MATLAB 一致性核验状态见 `reports/METRIC_AUDIT_20261101.md`。

复现历史 `paper` 协议时显式指定：

```bat
.venv\Scripts\python.exe test_OS.py --metric-profile paper
```

复现早期 CDDFuse/Dongying 的八项指标时使用 `--metric-profile legacy`。

## 历史 `paper` 协议的指标顺序与定义

The CSV columns follow the three thesis tables:

1. `EN`: base-2 entropy of the fused 8-bit luminance histogram.
2. `MI`: `MI(A,F) + MI(B,F)`, using a 256-by-256 joint histogram and base-2
   logarithms.
3. `FMI`: the same MI sum after a whole-image orthonormal 2-D DCT. DCT
   coefficients are jointly quantized to 256 levels for each source/fused pair.
4. `PSNR`: `10 log10(255^2 / MSE)`, where MSE is the mean of source/fused MSEs.
5. `Qabf`: Xydeas-Petrovic edge preservation with the constants printed in the
   thesis.
6. `Nabf`: artifact-weighted edge loss. A pixel is marked as an artifact when
   its fused gradient exceeds both source gradients.
7. `AG`: mean forward-difference gradient magnitude.
8. `SF`: row and column spatial frequencies normalized by the full image area.
9. `SD`: population standard deviation of fused luminance.
10. `CC`: mean Pearson correlation between each source and the fused image.
11. `SCD`: canonical cross-difference correlation sum.
12. `SSIM`: mean of source/fused SSIM values, using an 11-pixel Gaussian window,
    sigma 1.5, and data range 255.
13. `MS_SSIM`: mean source/fused five-scale SSIM with the standard Wang weights.
14. `VIF`: mean of the two source/fused VIF values.

## Reproducibility note

The thesis supplies equations but not the evaluation source that generated its
tables. `FMI`, `Nabf`, and `MS_SSIM` therefore require explicit implementation
choices documented above. They follow the thesis text and standard definitions,
but exact numerical equality with the thesis tables cannot be guaranteed without
the original evaluator and the original test image list.

