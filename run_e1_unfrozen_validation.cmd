@echo off
setlocal
cd /d "%~dp0"
set DATA_ROOT=..\datasets\QXSLAB_SAROPT
set SPLIT_MANIFEST=..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json
set MODEL_DIR=models\reliability_v5
set C0_CHECKPOINT=%MODEL_DIR%\frequency_minimax_baseline_pilot\epoch_005.pth
set E1_CHECKPOINT=%MODEL_DIR%\e1_encoder_unfrozen_pilot\epoch_005.pth
set C0_ENCODER_CHECKPOINT=%MODEL_DIR%\c0_encoder_unfrozen_control\epoch_005.pth
set C0_FINETUNE_CHECKPOINT=%MODEL_DIR%\c0_full_finetune_control\epoch_005.pth
set E1_E2E_CHECKPOINT=%MODEL_DIR%\e1_end2end_pilot\epoch_005.pth
set OUTPUT_ROOT=test_result\reliability_v5\validation_e1_unfrozen_final

for %%P in ("%C0_CHECKPOINT%" "%E1_CHECKPOINT%" "%SPLIT_MANIFEST%") do (
  if not exist "%%~P" (
    echo Missing required checkpoint or manifest: %%~P
    exit /b 1
  )
)

echo [1/4] Evaluate fixed C0 val200...
.venv\Scripts\python.exe -u test_OS.py ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --split val --limit 200 ^
  --checkpoint "%C0_CHECKPOINT%" ^
  --variant frequency_minimax_baseline ^
  --metric-profile paper --output-range-policy clamp ^
  --timing-warmup 0 --metrics-only ^
  --output-dir "%OUTPUT_ROOT%\frequency_minimax_baseline"
if errorlevel 1 exit /b 1

echo [2/4] Evaluate E1 unfrozen Encoder val200...
.venv\Scripts\python.exe -u test_OS.py ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --split val --limit 200 ^
  --checkpoint "%E1_CHECKPOINT%" ^
  --variant e1_dual_stem ^
  --metric-profile paper --output-range-policy clamp ^
  --timing-warmup 0 --metrics-only ^
  --output-dir "%OUTPUT_ROOT%\e1_encoder_unfrozen"
if errorlevel 1 exit /b 1

if exist "%C0_ENCODER_CHECKPOINT%" (
  echo [3/5] Evaluate matched C0 encoder-only control...
  .venv\Scripts\python.exe -u test_OS.py ^
    --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
    --split val --limit 200 ^
    --checkpoint "%C0_ENCODER_CHECKPOINT%" ^
    --variant frequency_minimax_baseline ^
    --metric-profile paper --output-range-policy clamp ^
    --timing-warmup 0 --metrics-only ^
    --output-dir "%OUTPUT_ROOT%\c0_encoder_unfrozen"
  if errorlevel 1 exit /b 1
) else (
  echo C0 matched encoder-only control not available yet, skipping.
)

if exist "%C0_FINETUNE_CHECKPOINT%" (
  echo [3/4] Evaluate optional C0 full-finetune control...
  .venv\Scripts\python.exe -u test_OS.py ^
    --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
    --split val --limit 200 ^
    --checkpoint "%C0_FINETUNE_CHECKPOINT%" ^
    --variant frequency_minimax_baseline ^
    --metric-profile paper --output-range-policy clamp ^
    --timing-warmup 0 --metrics-only ^
    --output-dir "%OUTPUT_ROOT%\c0_full_finetune"
  if errorlevel 1 exit /b 1
) else (
  echo C0 full-finetune control not available yet, skipping.
)

if exist "%E1_E2E_CHECKPOINT%" (
  echo [4/4] Evaluate optional E1 end-to-end...
  .venv\Scripts\python.exe -u test_OS.py ^
    --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
    --split val --limit 200 ^
    --checkpoint "%E1_E2E_CHECKPOINT%" ^
    --variant e1_dual_stem ^
    --metric-profile paper --output-range-policy clamp ^
    --timing-warmup 0 --metrics-only ^
    --output-dir "%OUTPUT_ROOT%\e1_end2end"
  if errorlevel 1 exit /b 1
) else (
  echo E1 end-to-end checkpoint not available yet, skipping.
)

echo Compare available results on identical val200...
.venv\Scripts\python.exe -u compare_e1_unfrozen.py --root "%OUTPUT_ROOT%"
if errorlevel 1 exit /b 1
echo E1 unfrozen evaluation complete.
