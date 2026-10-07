@echo off
setlocal

cd /d "%~dp0"

.venv\Scripts\python.exe calibrate_phase_evidence.py ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --fse-calibration models\reliability_v5\fse_v2a_calibration.json ^
  --sample-count 512 ^
  --pixels-per-image 2048 ^
  --flat-quantile 0.40 ^
  --threshold-quantile 0.995 ^
  --transition-quantile 0.95 ^
  --seed 42 ^
  --output models\reliability_v5\phase_evidence_calibration.json
if errorlevel 1 exit /b 1

.venv\Scripts\python.exe visualize_phase_evidence_comparison.py ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --fse-calibration models\reliability_v5\fse_v2a_calibration.json ^
  --phase-calibration models\reliability_v5\phase_evidence_calibration.json ^
  --split val ^
  --offset 0 ^
  --limit 24 ^
  --output-dir test_result\reliability_v5\phase_evidence_review
if errorlevel 1 exit /b 1

echo Phase antisymmetry comparison completed successfully.
