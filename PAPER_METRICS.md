# Paper-aligned fusion metrics

`utils/Evaluator.py` now contains two evaluation profiles:

- `Evaluator`: the unchanged, eight-metric CDDFuse implementation used to
  reproduce the original repository's published values.
- `PaperEvaluator`: the fourteen metrics described in Section 4.3.2 of Wang
  Haolong's thesis.

The Optical-SAR test uses `PaperEvaluator` by default:

```bat
.venv\Scripts\python.exe test_OS.py --metric-profile paper
```

Use `--metric-profile legacy` only when reproducing the earlier CDDFuse/Dongying
baseline.

## Metric order and definitions

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

