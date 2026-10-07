# CDDFuse reproduction record

This checkout reproduces the official pretrained-model evaluation on Windows
with an NVIDIA GPU.

## Source revision

- Repository: https://github.com/Zhaozixiang1228/MMIF-CDDFuse
- Commit: `73681d28f6100461f1b214ecf77a93b495be227b`

## Verified environment

- Windows
- Python 3.8.10
- PyTorch 2.0.1+cu118
- NVIDIA GeForce RTX 4060 Ti (16 GB)
- The remaining direct dependencies are pinned by `requirements.txt`.

Create the environment from the repository root:

```bat
C:\Users\Administrator\AppData\Local\Programs\Python\Python38\python.exe -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.venv\Scripts\python.exe -m pip install -r requirements-repro.txt
```

## Compatibility fixes

The released pretrained checkpoints use the legacy keys `DIDF_Encoder` and
`DIDF_Decoder`, while the current test scripts originally expected
`CDDF_Encoder` and `CDDF_Decoder`. Both test scripts now accept either format.

The released scripts also pass floating-point arrays directly to the PNG
writer. Recent Pillow versions cannot save PNG images in floating-point mode,
so `utils/img_read_save.py` now clips, rounds, and converts outputs to `uint8`
before saving.

These fixes do not alter the network or pretrained weights.

## Pretrained evaluation

Run:

```bat
.venv\Scripts\python.exe test_IVF.py
.venv\Scripts\python.exe test_MIF.py
```

Verified IVF results:

| Dataset | EN | SD | SF | MI | SCD | VIF | Qabf | SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TNO | 7.12 | 46.00 | 13.15 | 2.19 | 1.76 | 0.77 | 0.54 | 1.03 |
| RoadScene | 7.44 | 54.67 | 16.36 | 2.30 | 1.81 | 0.69 | 0.52 | 0.98 |

The IVF values exactly match the official README.

Verified MIF results:

| Dataset / weights | EN | SD | SF | MI | SCD | VIF | Qabf | SSIM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MRI_CT / IVF | 4.83 | 88.59 | 33.83 | 2.24 | 1.74 | 0.50 | 0.59 | 1.31 |
| MRI_CT / MIF | 4.88 | 79.17 | 38.15 | 2.61 | 1.41 | 0.61 | 0.68 | 1.34 |
| MRI_PET / IVF | 4.23 | 81.69 | 28.04 | 1.87 | 1.82 | 0.66 | 0.65 | 1.46 |
| MRI_PET / MIF | 4.22 | 70.73 | 29.57 | 2.03 | 1.69 | 0.71 | 0.71 | 1.49 |
| MRI_SPECT / IVF | 3.91 | 71.82 | 20.66 | 1.90 | 1.87 | 0.65 | 0.68 | 1.45 |
| MRI_SPECT / MIF | 3.90 | 58.31 | 20.87 | 2.49 | 1.35 | 0.97 | 0.78 | 1.48 |

MIF values match the README within 0.01; the small difference is attributable
to output quantization and library-level numerical rounding.

Generated images are written under `test_result/`.

To verify that both training phases can execute a forward and backward pass
without downloading MSRS, run:

```bat
.venv\Scripts\python.exe smoke_test.py
```

## Training reproduction

Training requires the MSRS training set, which is not included in the CDDFuse
repository. Download it from https://github.com/Linfeng-Tang/MSRS and arrange
the inputs as:

```text
MSRS_train/
  ir/
  vi/
```

Then run:

```bat
.venv\Scripts\python.exe dataprocessing.py
.venv\Scripts\python.exe train.py
```

`dataprocessing.py` creates
`data/MSRS_train_imgsize_128_stride_200.h5`. The official configuration trains
for 120 epochs, switching from reconstruction/decomposition training to fusion
training after epoch 40, and writes the final checkpoint under `models/`.
