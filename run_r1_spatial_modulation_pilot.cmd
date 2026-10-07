@echo off
setlocal

cd /d "%~dp0"
set PYTHONPATH=%CD%;%PYTHONPATH%

set DATA_ROOT=..datasetsQXSLAB_SAROPT
set SPLIT_MANIFEST=..datasetsQXSLAB_SAROPTsplitscddfuse_8000_1000_1000_seed20260924.json
set C0_RUN=modelseliability_v5requency_minimax_baseline_pilot
set C0_CHECKPOINT=%C0_RUN%est.pth
set C0_CONFIG=%C0_RUN%config.json

if not exist "%C0_CHECKPOINT%" (
  echo Missing C0 checkpoint: %C0_CHECKPOINT%
  exit /b 1
)
if not exist "%C0_CONFIG%" (
  echo Missing C0 config: %C0_CONFIG%
  exit /b 1
)
if not exist "%SPLIT_MANIFEST%" (
  echo Missing split manifest: %SPLIT_MANIFEST%
  exit /b 1
)

echo [1/3] Running R1 unit checks...
.venvScriptspython.exe tests	est_r1_spatial_modulation.py
if errorlevel 1 exit /b 1

echo [2/3] Verifying zero-initialized R1 is identical to C0 checkpoint...
.venvScriptspython.exe -u verify_r1_c0_identity.py ^
  --checkpoint "%C0_CHECKPOINT%" ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --split val ^
  --limit 4 ^
  --tolerance 1e-6
if errorlevel 1 exit /b 1

echo [3/3] Running R1 800/200 pilot...
.venvScriptspython.exe -u train_reliability_OS.py ^
  --variant r1_spatial_modulation ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --init-checkpoint "%C0_CHECKPOINT%" ^
  --frequency-reference-config "%C0_CONFIG%" ^
  --output-dir modelseliability_v5 ^
  --run-name r1_spatial_modulation_pilot ^
  --train-limit 800 ^
  --val-limit 200 ^
  --epochs 5 ^
  --phase1-epochs 0 ^
  --batch-size 8 ^
  --lr 1e-4 ^
  --decomp-weight 0 ^
  --optical-lowpass-sigma 2.0 ^
  --frequency-highpass-sigma 1.0 ^
  --frequency-midpass-sigma 2.0 ^
  --optical-high-weight 1.0 ^
  --structure-percentile 99 ^
  --structure-charbonnier-epsilon 0.001 ^
  --minimax-temperature 0.1 ^
  --reference-batches 0 ^
  --r1-modulation-alpha 0.25 ^
  --r1-dev-weight 0.01 ^
  --clip-grad 0.01 ^
  --collapse-mae-threshold 0.01 ^
  --max-val-batches 0 ^
  --save-every 5 ^
  --seed 42

if errorlevel 1 exit /b 1
echo R1 spatial-modulation pilot completed successfully.
