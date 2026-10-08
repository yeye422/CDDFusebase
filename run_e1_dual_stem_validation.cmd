@echo off
setlocal

cd /d "%~dp0"

set DATA_ROOT=..\datasets\QXSLAB_SAROPT
set SPLIT_MANIFEST=..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json
set C0_CHECKPOINT=models\reliability_v5\frequency_minimax_baseline_pilot\epoch_005.pth
set E1_CHECKPOINT=models\reliability_v5\e1_dual_stem_pilot\epoch_005.pth
set OUTPUT_ROOT=test_result\reliability_v5\validation_e1_dual_stem_final

if not exist "%C0_CHECKPOINT%" (
  echo Missing C0 checkpoint: %C0_CHECKPOINT%
  exit /b 1
)
if not exist "%E1_CHECKPOINT%" (
  echo Missing E1 checkpoint: %E1_CHECKPOINT%
  exit /b 1
)

echo [1/3] C0 same val200 endpoint...
.venv\Scripts\python.exe -u test_OS.py ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --split val --limit 200 ^
  --checkpoint "%C0_CHECKPOINT%" ^
  --variant frequency_minimax_baseline ^
  --metric-profile paper --output-range-policy clamp ^
  --timing-warmup 0 --metrics-only ^
  --output-dir "%OUTPUT_ROOT%\frequency_minimax_baseline"
if errorlevel 1 exit /b 1

echo [2/3] E1 dual stem on matched val200...
.venv\Scripts\python.exe -u test_OS.py ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --split val --limit 200 ^
  --checkpoint "%E1_CHECKPOINT%" ^
  --variant e1_dual_stem ^
  --metric-profile paper --output-range-policy clamp ^
  --timing-warmup 0 --metrics-only ^
  --output-dir "%OUTPUT_ROOT%\e1_dual_stem"
if errorlevel 1 exit /b 1

echo [3/3] Matched paired C0/E1 comparison...
.venv\Scripts\python.exe -u compare_e1_to_c0.py ^
  --root "%OUTPUT_ROOT%"
if errorlevel 1 exit /b 1

echo E1 validation complete. Read e1_vs_c0_summary.csv.
