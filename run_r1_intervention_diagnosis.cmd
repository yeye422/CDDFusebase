@echo off
setlocal

cd /d "%~dp0"

set DATA_ROOT=..\datasets\QXSLAB_SAROPT
set SPLIT_MANIFEST=..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json
set R1_CHECKPOINT=models\reliability_v5\r1_spatial_modulation_pilot\epoch_005.pth
set OUTPUT_DIR=test_result\reliability_v5\r1_spatial_modulation_diagnosis_val200

if not exist "%R1_CHECKPOINT%" (
  echo Missing R1 checkpoint: %R1_CHECKPOINT%
  exit /b 1
)
if not exist "%SPLIT_MANIFEST%" (
  echo Missing split manifest: %SPLIT_MANIFEST%
  exit /b 1
)

.venv\Scripts\python.exe -u diagnose_r1_interventions.py ^
  --checkpoint "%R1_CHECKPOINT%" ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --split val ^
  --offset 0 ^
  --limit 200 ^
  --seed 20261008 ^
  --smooth-kernel 5 ^
  --output-dir "%OUTPUT_DIR%"

if errorlevel 1 exit /b 1
echo R1 causal intervention diagnosis completed successfully.
echo Results: %OUTPUT_DIR%
