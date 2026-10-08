@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%CD%;%PYTHONPATH%
set DATA_ROOT=..\datasets\QXSLAB_SAROPT
set SPLIT_MANIFEST=..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json
set C0_RUN=models\reliability_v5\frequency_minimax_baseline_pilot
set C0_CHECKPOINT=%C0_RUN%\epoch_005.pth
set C0_CONFIG=%C0_RUN%\config.json
for %%P in ("%C0_CHECKPOINT%" "%C0_CONFIG%" "%SPLIT_MANIFEST%") do (
  if not exist "%%~P" (
    echo Missing experiment input: %%~P
    exit /b 1
  )
)

echo === E1 end-to-end ===
echo E1 full-network finetune from C0...
.venv\Scripts\python.exe -u train_reliability_OS.py ^
  --variant e1_dual_stem ^
  --training-policy all ^
  --run-name e1_end2end_pilot ^
  --lr 1e-5 ^
  --e1-stem-lr 1e-4 ^
  --e1-fusion-lr 1e-6 ^
  --e1-stem-dev-weight 0 ^
  --init-checkpoint "%C0_CHECKPOINT%" ^
  --frequency-reference-config "%C0_CONFIG%" ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --output-dir models\reliability_v5 ^
  --train-limit 800 --val-limit 200 ^
  --epochs 5 --phase1-epochs 0 ^
  --batch-size 8 ^
  --decomp-weight 0 ^
  --optical-lowpass-sigma 2.0 ^
  --frequency-highpass-sigma 1.0 ^
  --frequency-midpass-sigma 2.0 ^
  --optical-high-weight 1.0 ^
  --structure-percentile 99 ^
  --structure-charbonnier-epsilon 0.001 ^
  --minimax-temperature 0.1 ^
  --reference-batches 0 ^
  --clip-grad 0.01 ^
  --collapse-mae-threshold 0.01 ^
  --max-val-batches 0 ^
  --save-every 5 ^
  --seed 42
if errorlevel 1 exit /b 1
echo Training complete: e1_end2end_pilot
