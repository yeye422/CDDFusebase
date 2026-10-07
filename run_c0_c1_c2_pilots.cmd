@echo off
setlocal

cd /d "%~dp0"

for %%V in (frequency_matched_baseline base_optical_detail base_optical_detail_refine) do (
  echo Training %%V
  .venv\Scripts\python.exe -u train_reliability_OS.py ^
    --variant %%V ^
    --training-policy fusion_only ^
    --data-root ..\datasets\QXSLAB_SAROPT ^
    --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
    --init-checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
    --output-dir models\reliability_v5 ^
    --run-name %%V_pilot ^
    --train-limit 800 ^
    --val-limit 200 ^
    --epochs 5 ^
    --phase1-epochs 0 ^
    --batch-size 8 ^
    --lr 1e-5 ^
    --decomp-weight 0 ^
    --optical-lowpass-sigma 2.0 ^
    --frequency-highpass-sigma 1.0 ^
    --frequency-midpass-sigma 2.0 ^
    --optical-high-weight 1.0 ^
    --structure-percentile 99 ^
    --structure-charbonnier-epsilon 0.001 ^
    --collapse-mae-threshold 0.01 ^
    --max-val-batches 0 ^
    --save-every 1 ^
    --seed 42
  if errorlevel 1 exit /b 1
)

echo C0/C1/C2 pilots completed successfully.
