@echo off
setlocal

cd /d "%~dp0"

.venv\Scripts\python.exe test_OS.py ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --split val --limit 200 ^
  --checkpoint models\reliability_v5\fixed_line_roa_pilot\epoch_005.pth ^
  --variant fixed_line_roa ^
  --metric-profile paper --output-range-policy clamp ^
  --timing-warmup 0 --metrics-only ^
  --output-dir test_result\reliability_v5\c3_line_validation\fixed_line_roa
if errorlevel 1 exit /b 1

.venv\Scripts\python.exe test_OS.py ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --split val --limit 200 ^
  --checkpoint models\reliability_v5\fixed_line_phase_bridge_pilot\epoch_005.pth ^
  --variant fixed_line_phase_bridge ^
  --metric-profile paper --output-range-policy clamp ^
  --timing-warmup 0 --metrics-only ^
  --output-dir test_result\reliability_v5\c3_line_validation\fixed_line_phase_bridge
if errorlevel 1 exit /b 1

.venv\Scripts\python.exe summarize_c3_pilots.py
if errorlevel 1 exit /b 1

echo C3a/C3b validation completed successfully.
