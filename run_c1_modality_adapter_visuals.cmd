@echo off
setlocal

cd /d "%~dp0"

set DATA_ROOT=..\datasets\QXSLAB_SAROPT
set SPLIT_MANIFEST=..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json
set MODELS_DIR=models\reliability_v5
set OUTPUT_ROOT=test_result\reliability_v5\c1_qualitative_val24

if not exist "%MODELS_DIR%\frequency_minimax_baseline_pilot\epoch_005.pth" exit /b 1
if not exist "%MODELS_DIR%\c1_base_adapter_pilot\epoch_005.pth" exit /b 1
if not exist "%MODELS_DIR%\c1_detail_adapter_pilot\epoch_005.pth" exit /b 1

echo [1/3] C0 val24 previews...
.venv\Scripts\python.exe -u test_OS.py ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --split val --limit 24 ^
  --checkpoint "%MODELS_DIR%\frequency_minimax_baseline_pilot\epoch_005.pth" ^
  --variant frequency_minimax_baseline ^
  --metric-profile paper --output-range-policy clamp --timing-warmup 0 ^
  --output-dir "%OUTPUT_ROOT%\frequency_minimax_baseline"
if errorlevel 1 exit /b 1

echo [2/3] C1 Base val24 previews...
.venv\Scripts\python.exe -u test_OS.py ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --split val --limit 24 ^
  --checkpoint "%MODELS_DIR%\c1_base_adapter_pilot\epoch_005.pth" ^
  --variant c1_base_adapter ^
  --metric-profile paper --output-range-policy clamp --timing-warmup 0 ^
  --output-dir "%OUTPUT_ROOT%\c1_base_adapter"
if errorlevel 1 exit /b 1

echo [3/3] C1 Detail val24 previews...
.venv\Scripts\python.exe -u test_OS.py ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --split val --limit 24 ^
  --checkpoint "%MODELS_DIR%\c1_detail_adapter_pilot\epoch_005.pth" ^
  --variant c1_detail_adapter ^
  --metric-profile paper --output-range-policy clamp --timing-warmup 0 ^
  --output-dir "%OUTPUT_ROOT%\c1_detail_adapter"
if errorlevel 1 exit /b 1
echo C1 qualitative images saved under %OUTPUT_ROOT%
