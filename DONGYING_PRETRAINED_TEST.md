# Dongying zero-shot test with CDDFuse pretrained weights

## Dataset

- Official source: https://github.com/XD-MG/DDHRNet
- Downloaded archive: `../datasets/DDHRNet_DATA.zip`
- Google Drive file ID: `1DAojDL2IjuJjW5fJLFCxj0cPqNOdHRzI`
- Archive size: 539,763,437 bytes
- SHA-256: `35a827fc402eb7267861214b5e6e80b470a7f03def6fb6cda81d3f25cfedc994`
- Extracted subset: `../datasets/Dongying`

The archive names the Dongying subset `shandong`. Extraction and pairing
validation found:

| Folder | Content | Count |
| --- | --- | ---: |
| `GF2` | RGB optical JPEG | 7,831 |
| `GF3` | VV SAR JPEG | 7,831 |
| `label` | land-cover label PNG | 7,831 |

All three folders have identical numeric stems from `1` through `7831`.

## Zero-shot protocol

- Checkpoint: `models/CDDFuse_IVF.pth`
- Test script: `test_OS.py`
- Device: NVIDIA GPU / CUDA
- Deterministic subset: numeric image IDs `1` through `150`
- Optical input: GF-2 RGB converted to YCrCb; only Y enters CDDFuse
- SAR input: GF-3 converted to one grayscale channel
- Input scaling: uint8 divided by 255
- Network weights: unchanged IVF pretrained weights
- RGB output: fused Y combined with the original GF-2 Cb and Cr channels
- Output normalization: the same per-image min-max normalization used by the
  official CDDFuse test scripts

Command:

```bat
.venv\Scripts\python.exe test_OS.py --metric-profile legacy --limit 150 --output-dir test_result\Dongying_IVF_150
```

## Mean metrics over 150 pairs

The values below use the original CDDFuse `legacy` metric profile. For the
fourteen thesis-aligned metrics, run `test_OS.py` with `--metric-profile paper`
(the current default); see `PAPER_METRICS.md` for definitions.

| EN | SD | SF | MI | SCD | VIF | Qabf | SSIM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 6.6700 | 34.8394 | 20.1362 | 2.2976 | 1.4637 | 0.6395 | 0.6245 | 1.0304 |

These are no-reference/source-referenced fusion metrics, not evidence that the
IVF checkpoint is optimal for Optical-SAR fusion. In particular, several of
these metrics can reward added high-frequency noise.

## Outputs

- `test_result/Dongying_IVF_150/gray`: fused luminance PNGs
- `test_result/Dongying_IVF_150/rgb`: fused RGB PNGs
- `test_result/Dongying_IVF_150/preview`: GF-2 / GF-3 / fused triptychs
- `test_result/Dongying_IVF_150/metrics.csv`: per-image values, mean, and std

## Initial observations

The pretrained IVF network runs successfully on Dongying without architecture
changes, but the domain shift is visually material:

1. Fine SAR speckle is strongly transferred into many fused outputs.
2. Bright SAR line responses can dominate optical luminance.
3. Black invalid/no-data wedges in GF-3 are also transferred into the fused
   image instead of being masked.
4. Optical chrominance is retained because Cb/Cr are copied directly, but this
   does not prevent luminance artifacts.

Therefore this run is a valid zero-shot diagnostic baseline, not a final
Optical-SAR baseline. Before training, invalid SAR regions should be explicitly
masked; after training, evaluation should separate stable structure transfer
from speckle leakage.
