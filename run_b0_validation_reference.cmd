@echo off
setlocal

cd /d "%~dp0"

.venv\Scripts\python.exe test_OS.py ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --split val ^
  --limit 200 ^
  --checkpoint models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth ^
  --variant original ^
  --metric-profile paper ^
  --output-range-policy clamp ^
  --timing-warmup 0 ^
  --metrics-only ^
  --output-dir test_result\reliability_v5\validation_reference\B0_200

if errorlevel 1 exit /b 1

.venv\Scripts\python.exe summarize_c0_minimax_comparison.py
if errorlevel 1 exit /b 1

echo B0 reference and fixed-endpoint comparison completed successfully.
