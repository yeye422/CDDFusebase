@echo off
setlocal

cd /d "%~dp0"

.venv\Scripts\python.exe calibrate_structure_evidence_v2.py ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --sample-count 512 ^
  --pixels-per-image 2048 ^
  --flat-quantile 0.40 ^
  --threshold-quantile 0.995 ^
  --transition-quantile 0.95 ^
  --scales 3 5 9 ^
  --seed 42 ^
  --output models\reliability_v5\fse_v2a_calibration.json
if errorlevel 1 exit /b 1

.venv\Scripts\python.exe visualize_fixed_structure_evidence_v2.py ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --calibration models\reliability_v5\fse_v2a_calibration.json ^
  --split val ^
  --offset 0 ^
  --limit 24 ^
  --output-dir test_result\reliability_v5\fixed_structure_v2a_review
if errorlevel 1 exit /b 1

echo FSE-v2a calibration and review completed successfully.
