@echo off
setlocal

cd /d "%~dp0"

.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant fixed_line_roa ^
  --training-policy fusion_only ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --init-checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
  --structure-calibration models\reliability_v5\fse_v2a_calibration.json ^
  --output-dir models\reliability_v5 ^
  --run-name fixed_line_roa_pilot ^
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
  --line-structure-weight 1.0 ^
  --line-reliability-threshold 0.5 ^
  --structure-percentile 99 ^
  --structure-charbonnier-epsilon 0.001 ^
  --minimax-temperature 0.1 ^
  --reference-batches 0 ^
  --collapse-mae-threshold 0.01 ^
  --max-val-batches 0 ^
  --save-every 5 ^
  --seed 42
if errorlevel 1 exit /b 1

.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant fixed_line_phase_bridge ^
  --training-policy fusion_only ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --init-checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
  --structure-calibration models\reliability_v5\fse_v2a_calibration.json ^
  --phase-calibration models\reliability_v5\phase_evidence_calibration.json ^
  --output-dir models\reliability_v5 ^
  --run-name fixed_line_phase_bridge_pilot ^
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
  --line-structure-weight 1.0 ^
  --line-reliability-threshold 0.5 ^
  --structure-percentile 99 ^
  --structure-charbonnier-epsilon 0.001 ^
  --minimax-temperature 0.1 ^
  --reference-batches 0 ^
  --collapse-mae-threshold 0.01 ^
  --max-val-batches 0 ^
  --save-every 5 ^
  --seed 42
if errorlevel 1 exit /b 1

echo C3a/C3b fixed-endpoint pilots completed successfully.
