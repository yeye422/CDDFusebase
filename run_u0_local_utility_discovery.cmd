@echo off
setlocal

cd /d "%~dp0"

set DATA_ROOT=..\datasets\QXSLAB_SAROPT
set SPLIT_MANIFEST=..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json
set C0_CHECKPOINT=models\reliability_v5\frequency_minimax_baseline_pilot\epoch_005.pth

if not exist "%C0_CHECKPOINT%" (
  echo Missing C0 fixed endpoint: %C0_CHECKPOINT%
  exit /b 1
)
if not exist "%SPLIT_MANIFEST%" (
  echo Missing split manifest: %SPLIT_MANIFEST%
  exit /b 1
)

echo [1/2] Checking U0 intervention invariants...
set PYTHONPATH=%CD%;%PYTHONPATH%
.venv\Scripts\python.exe tests\test_u0_local_utility.py
if errorlevel 1 exit /b 1

echo [2/2] U0 discovery audit on fixed train24...
.venv\Scripts\python.exe -u audit_u0_local_utility.py ^
  --checkpoint "%C0_CHECKPOINT%" ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --split train ^
  --offset 0 ^
  --limit 24 ^
  --grid 4 ^
  --eps 0.10 0.20 ^
  --ramp 8 ^
  --min-tile-size 48 ^
  --min-sar-valid 0.90 ^
  --direction-tolerance 0.0002 ^
  --output-dir test_result\reliability_v5\u0_local_utility_train24

if errorlevel 1 exit /b 1
echo U0 discovery finished. Use the separate validation runner only if discovery is informative.
