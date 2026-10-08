@echo off
setlocal
cd /d "%~dp0"
set DATA_ROOT=..\datasets\QXSLAB_SAROPT
set SPLIT_MANIFEST=%DATA_ROOT%\splits\cddfuse_8000_1000_1000_seed20260924.json
set MODELS=models\reliability_v5
set OUT=test_result\reliability_v5\validation_d1_c0_constraints_final
for %%P in (
  "%MODELS%\frequency_minimax_baseline_pilot\epoch_005.pth"
  "%MODELS%\d1_c0_A0\epoch_005.pth"
  "%MODELS%\d1_c0_A1\epoch_005.pth"
  "%MODELS%\d1_c0_A2\epoch_005.pth"
  "%MODELS%\d1_c0_A3\epoch_005.pth"
  "%SPLIT_MANIFEST%"
) do (
  if not exist "%%~P" (
    echo Missing required checkpoint or manifest: %%~P
    exit /b 1
  )
)

echo Evaluate C0 fixed val200...
.venv\Scripts\python.exe -u test_OS.py ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --split val --limit 200 ^
  --checkpoint "%MODELS%\frequency_minimax_baseline_pilot\epoch_005.pth" ^
  --variant frequency_minimax_baseline ^
  --metric-profile paper --output-range-policy clamp ^
  --timing-warmup 0 --metrics-only ^
  --output-dir "%OUT%\C0"
if errorlevel 1 exit /b 1

echo Evaluate A0 fixed val200...
.venv\Scripts\python.exe -u test_OS.py ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --split val --limit 200 ^
  --checkpoint "%MODELS%\d1_c0_A0\epoch_005.pth" ^
  --variant frequency_minimax_baseline ^
  --metric-profile paper --output-range-policy clamp ^
  --timing-warmup 0 --metrics-only ^
  --output-dir "%OUT%\A0"
if errorlevel 1 exit /b 1

echo Evaluate A1 fixed val200...
.venv\Scripts\python.exe -u test_OS.py ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --split val --limit 200 ^
  --checkpoint "%MODELS%\d1_c0_A1\epoch_005.pth" ^
  --variant frequency_minimax_baseline ^
  --metric-profile paper --output-range-policy clamp ^
  --timing-warmup 0 --metrics-only ^
  --output-dir "%OUT%\A1"
if errorlevel 1 exit /b 1

echo Evaluate A2 fixed val200...
.venv\Scripts\python.exe -u test_OS.py ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --split val --limit 200 ^
  --checkpoint "%MODELS%\d1_c0_A2\epoch_005.pth" ^
  --variant frequency_minimax_baseline ^
  --metric-profile paper --output-range-policy clamp ^
  --timing-warmup 0 --metrics-only ^
  --output-dir "%OUT%\A2"
if errorlevel 1 exit /b 1

echo Evaluate A3 fixed val200...
.venv\Scripts\python.exe -u test_OS.py ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --split val --limit 200 ^
  --checkpoint "%MODELS%\d1_c0_A3\epoch_005.pth" ^
  --variant frequency_minimax_baseline ^
  --metric-profile paper --output-range-policy clamp ^
  --timing-warmup 0 --metrics-only ^
  --output-dir "%OUT%\A3"
if errorlevel 1 exit /b 1

echo Summarize D1 paired C0 comparisons...
.venv\Scripts\python.exe -u compare_d1_c0_constraints.py --root "%OUT%"
if errorlevel 1 exit /b 1
echo D1 evaluation complete: %OUT%\d1_vs_c0_summary.csv
