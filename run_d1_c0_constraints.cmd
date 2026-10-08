@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%CD%;%PYTHONPATH%
if "%~1"=="" goto usage

set D1=%~1
set DECOMP=
set RECON=
if /I "%D1%"=="A0" (
  set DECOMP=0
  set RECON=0
)
if /I "%D1%"=="A1" (
  set DECOMP=0.05
  set RECON=0
)
if /I "%D1%"=="A2" (
  set DECOMP=0
  set RECON=0.05
)
if /I "%D1%"=="A3" (
  set DECOMP=0.05
  set RECON=0.05
)
if not defined DECOMP goto usage

set DATA_ROOT=..\datasets\QXSLAB_SAROPT
set SPLIT_MANIFEST=%DATA_ROOT%\splits\cddfuse_8000_1000_1000_seed20260924.json
set C0_RUN=models\reliability_v5\frequency_minimax_baseline_pilot
set C0_CHECKPOINT=%C0_RUN%\epoch_005.pth
set C0_CONFIG=%C0_RUN%\config.json
for %%P in ("%C0_CHECKPOINT%" "%C0_CONFIG%" "%SPLIT_MANIFEST%") do (
  if not exist "%%~P" (
    echo Missing experiment input: %%~P
    exit /b 1
  )
)

echo D1-%D1% C0 Encoder-only decomp=%DECOMP% recon=%RECON%...
.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant frequency_minimax_baseline ^
  --training-policy encoder_only ^
  --run-name d1_c0_%D1% ^
  --lr 1e-5 --c0-embed-lr 1e-4 ^
  --init-checkpoint "%C0_CHECKPOINT%" ^
  --frequency-reference-config "%C0_CONFIG%" ^
  --data-root "%DATA_ROOT%" --split-manifest "%SPLIT_MANIFEST%" ^
  --output-dir models\reliability_v5 ^
  --train-limit 800 --val-limit 200 ^
  --epochs 5 --phase1-epochs 0 --batch-size 8 ^
  --decomp-weight 0 ^
  --d1-decomp-weight %DECOMP% ^
  --d1-recon-weight %RECON% ^
  --d1-calibration-batches 4 ^
  --optical-lowpass-sigma 2.0 ^
  --frequency-highpass-sigma 1.0 ^
  --frequency-midpass-sigma 2.0 ^
  --optical-high-weight 1.0 ^
  --structure-percentile 99 ^
  --structure-charbonnier-epsilon 0.001 ^
  --minimax-temperature 0.1 ^
  --clip-grad 0.01 --collapse-mae-threshold 0.01 ^
  --max-val-batches 0 --save-every 1 --seed 42
if errorlevel 1 exit /b 1
echo D1-%D1% training complete.
exit /b 0

:usage
echo Usage: %~nx0 A0 ^| A1 ^| A2 ^| A3
echo A0=Minimax, A1=+decomp, A2=+recon, A3=+both
exit /b 2
