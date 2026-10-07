@echo off
setlocal

cd /d "%~dp0"

.venv\Scripts\python.exe test_OS.py ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --split val ^
  --limit 200 ^
  --checkpoint models\reliability_v5\frequency_minimax_baseline_pilot\epoch_005.pth ^
  --variant frequency_minimax_baseline ^
  --metric-profile paper ^
  --output-range-policy clamp ^
  --timing-warmup 0 ^
  --metrics-only ^
  --output-dir test_result\reliability_v5\validation_minimax_final\frequency_minimax_baseline

if errorlevel 1 exit /b 1
.venv\Scripts\python.exe summarize_frequency_pilots.py ^
  --root test_result\reliability_v5\validation_minimax_final ^
  --output test_result\reliability_v5\validation_minimax_final\summary.csv
if errorlevel 1 exit /b 1
echo C0 minimax fixed-endpoint validation completed successfully.
