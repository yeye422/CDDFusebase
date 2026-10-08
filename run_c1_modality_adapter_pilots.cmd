@echo off
setlocal

cd /d "%~dp0"
set PYTHONPATH=%CD%;%PYTHONPATH%

set DATA_ROOT=..\datasets\QXSLAB_SAROPT
set SPLIT_MANIFEST=..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json
set C0_RUN=models\reliability_v5\frequency_minimax_baseline_pilot
set C0_CHECKPOINT=%C0_RUN%\epoch_005.pth
set C0_CONFIG=%C0_RUN%\config.json

if not exist "%C0_CHECKPOINT%" (
  echo Missing C0 checkpoint: %C0_CHECKPOINT%
  exit /b 1
)
if not exist "%C0_CONFIG%" (
  echo Missing C0 reference config: %C0_CONFIG%
  exit /b 1
)
if not exist "%SPLIT_MANIFEST%" (
  echo Missing split manifest: %SPLIT_MANIFEST%
  exit /b 1
)

echo [1/4] C1 unit tests...
.venv\Scripts\python.exe tests\test_c1_modality_adapter.py
if errorlevel 1 exit /b 1

echo [2/4] Frozen C0 real-checkpoint identity (Base and Detail)...
.venv\Scripts\python.exe -u verify_c1_c0_identity.py ^
  --checkpoint "%C0_CHECKPOINT%" ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --limit 4 --tolerance 1e-6
if errorlevel 1 exit /b 1

echo [3/4] Train isolated C1 Base adapter pilot...
.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant c1_base_adapter ^
  --init-checkpoint "%C0_CHECKPOINT%" ^
  --frequency-reference-config "%C0_CONFIG%" ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --output-dir models\reliability_v5 ^
  --run-name c1_base_adapter_pilot ^
  --train-limit 800 --val-limit 200 ^
  --epochs 5 --phase1-epochs 0 ^
  --batch-size 8 --lr 1e-4 ^
  --decomp-weight 0 ^
  --optical-lowpass-sigma 2.0 ^
  --frequency-highpass-sigma 1.0 ^
  --frequency-midpass-sigma 2.0 ^
  --optical-high-weight 1.0 ^
  --structure-percentile 99 ^
  --structure-charbonnier-epsilon 0.001 ^
  --minimax-temperature 0.1 ^
  --c1-adapter-alpha 0.10 ^
  --c1-dev-weight 0.01 ^
  --clip-grad 0.01 ^
  --collapse-mae-threshold 0.01 ^
  --max-val-batches 0 --save-every 5 --seed 42
if errorlevel 1 exit /b 1

echo [4/4] Train isolated C1 Detail adapter pilot...
.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant c1_detail_adapter ^
  --init-checkpoint "%C0_CHECKPOINT%" ^
  --frequency-reference-config "%C0_CONFIG%" ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --output-dir models\reliability_v5 ^
  --run-name c1_detail_adapter_pilot ^
  --train-limit 800 --val-limit 200 ^
  --epochs 5 --phase1-epochs 0 ^
  --batch-size 8 --lr 1e-4 ^
  --decomp-weight 0 ^
  --optical-lowpass-sigma 2.0 ^
  --frequency-highpass-sigma 1.0 ^
  --frequency-midpass-sigma 2.0 ^
  --optical-high-weight 1.0 ^
  --structure-percentile 99 ^
  --structure-charbonnier-epsilon 0.001 ^
  --minimax-temperature 0.1 ^
  --c1-adapter-alpha 0.10 ^
  --c1-dev-weight 0.01 ^
  --clip-grad 0.01 ^
  --collapse-mae-threshold 0.01 ^
  --max-val-batches 0 --save-every 5 --seed 42
if errorlevel 1 exit /b 1

echo C1 Base and Detail isolated pilots completed.
