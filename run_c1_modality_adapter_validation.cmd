@echo off
setlocal

cd /d "%~dp0"

set DATA_ROOT=..\datasets\QXSLAB_SAROPT
set SPLIT_MANIFEST=..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json
set MODELS_DIR=models\reliability_v5
set OUTPUT_ROOT=test_result\reliability_v5\validation_c1_adapters_final
set C0_CHECKPOINT=%MODELS_DIR%\frequency_minimax_baseline_pilot\epoch_005.pth
set C1_BASE_CHECKPOINT=%MODELS_DIR%\c1_base_adapter_pilot\epoch_005.pth
set C1_DETAIL_CHECKPOINT=%MODELS_DIR%\c1_detail_adapter_pilot\epoch_005.pth

for %%P in ("%C0_CHECKPOINT%" "%C1_BASE_CHECKPOINT%" "%C1_DETAIL_CHECKPOINT%") do (
  if not exist "%%~P" (
    echo Missing checkpoint: %%~P
    exit /b 1
  )
)

echo [1/4] C0 fixed endpoint on val200...
.venv\Scripts\python.exe -u test_OS.py ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --split val --limit 200 ^
  --checkpoint "%C0_CHECKPOINT%" ^
  --variant frequency_minimax_baseline ^
  --metric-profile paper --output-range-policy clamp ^
  --timing-warmup 0 --metrics-only ^
  --output-dir "%OUTPUT_ROOT%\frequency_minimax_baseline"
if errorlevel 1 exit /b 1

echo [2/4] C1 Base-only adapter on the same val200...
.venv\Scripts\python.exe -u test_OS.py ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --split val --limit 200 ^
  --checkpoint "%C1_BASE_CHECKPOINT%" ^
  --variant c1_base_adapter ^
  --metric-profile paper --output-range-policy clamp ^
  --timing-warmup 0 --metrics-only ^
  --output-dir "%OUTPUT_ROOT%\c1_base_adapter"
if errorlevel 1 exit /b 1

echo [3/4] C1 Detail-only adapter on the same val200...
.venv\Scripts\python.exe -u test_OS.py ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --split val --limit 200 ^
  --checkpoint "%C1_DETAIL_CHECKPOINT%" ^
  --variant c1_detail_adapter ^
  --metric-profile paper --output-range-policy clamp ^
  --timing-warmup 0 --metrics-only ^
  --output-dir "%OUTPUT_ROOT%\c1_detail_adapter"
if errorlevel 1 exit /b 1

echo [4/4] Paired C0 / C1 Base / C1 Detail comparisons...
.venv\Scripts\python.exe -u compare_c1_to_c0.py ^
  --root "%OUTPUT_ROOT%"
if errorlevel 1 exit /b 1

echo C1 fixed-endpoint evaluation complete. Read c1_vs_c0_summary.csv.
