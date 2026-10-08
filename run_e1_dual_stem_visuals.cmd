@echo off
setlocal

cd /d "%~dp0"

set DATA_ROOT=..\datasets\QXSLAB_SAROPT
set SPLIT_MANIFEST=..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json
set C0_CHECKPOINT=models\reliability_v5\frequency_minimax_baseline_pilot\epoch_005.pth
set E1_CHECKPOINT=models\reliability_v5\e1_dual_stem_pilot\epoch_005.pth
set OUTPUT_ROOT=test_result\reliability_v5\e1_qualitative_val24

if not exist "%C0_CHECKPOINT%" exit /b 1
if not exist "%E1_CHECKPOINT%" exit /b 1

echo [1/2] Matched C0 visual outputs...
.venv\Scripts\python.exe -u test_OS.py ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --split val --limit 24 ^
  --checkpoint "%C0_CHECKPOINT%" ^
  --variant frequency_minimax_baseline ^
  --metric-profile paper --output-range-policy clamp --timing-warmup 0 ^
  --output-dir "%OUTPUT_ROOT%\frequency_minimax_baseline"
if errorlevel 1 exit /b 1

echo [2/2] Matched E1 visual outputs...
.venv\Scripts\python.exe -u test_OS.py ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --split val --limit 24 ^
  --checkpoint "%E1_CHECKPOINT%" ^
  --variant e1_dual_stem ^
  --metric-profile paper --output-range-policy clamp --timing-warmup 0 ^
  --output-dir "%OUTPUT_ROOT%\e1_dual_stem"
if errorlevel 1 exit /b 1

echo E1 visual outputs ready.
