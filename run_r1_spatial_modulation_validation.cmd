@echo off
setlocal

cd /d "%~dp0"

set DATA_ROOT=..\datasets\QXSLAB_SAROPT
set SPLIT_MANIFEST=..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json
set C0_CHECKPOINT=models\reliability_v5\frequency_minimax_baseline_pilot\epoch_005.pth
set R1_CHECKPOINT=models\reliability_v5\r1_spatial_modulation_pilot\epoch_005.pth
set OUTPUT_ROOT=test_result\reliability_v5\validation_r1_final

if not exist "%C0_CHECKPOINT%" (
  echo Missing C0 checkpoint: %C0_CHECKPOINT%
  exit /b 1
)
if not exist "%R1_CHECKPOINT%" (
  echo Missing R1 checkpoint: %R1_CHECKPOINT%
  exit /b 1
)

echo [1/3] Evaluating C0 fixed endpoint on val200...
.venv\Scripts\python.exe test_OS.py ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --split val ^
  --limit 200 ^
  --checkpoint "%C0_CHECKPOINT%" ^
  --variant frequency_minimax_baseline ^
  --metric-profile paper ^
  --output-range-policy clamp ^
  --timing-warmup 0 ^
  --metrics-only ^
  --output-dir "%OUTPUT_ROOT%\frequency_minimax_baseline"

if errorlevel 1 exit /b 1

echo [2/3] Evaluating R1 fixed endpoint on the same val200...
.venv\Scripts\python.exe test_OS.py ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --split val ^
  --limit 200 ^
  --checkpoint "%R1_CHECKPOINT%" ^
  --variant r1_spatial_modulation ^
  --metric-profile paper ^
  --output-range-policy clamp ^
  --timing-warmup 0 ^
  --metrics-only ^
  --output-dir "%OUTPUT_ROOT%\r1_spatial_modulation"

if errorlevel 1 exit /b 1

echo [3/3] Comparing R1 directly against C0...
.venv\Scripts\python.exe compare_r1_to_c0.py ^
  --root "%OUTPUT_ROOT%" ^
  --output "%OUTPUT_ROOT%\r1_vs_c0.csv"

if errorlevel 1 exit /b 1
echo R1 fixed-endpoint validation completed successfully.
