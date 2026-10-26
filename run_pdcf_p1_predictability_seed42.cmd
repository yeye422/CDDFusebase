@echo off
setlocal

cd /d "%~dp0"

.venv\Scripts\python.exe -u train_pdcf_predictability.py ^
  --c0-checkpoint models\reliability_v5\frequency_minimax_seed42\best.pth ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --output-dir models\pdcf_p1 ^
  --run-name predictability_seed42 ^
  --epochs 10 ^
  --batch-size 8 ^
  --patch-size 128 ^
  --lr 1e-4 ^
  --weight-decay 1e-4 ^
  --hidden-channels 32 ^
  --predictor-blocks 2 ^
  --smooth-l1-beta 0.1 ^
  --statistics-batches 0 ^
  --seed 42

if errorlevel 1 exit /b 1
echo PDCF P1 predictability seed42 training completed successfully.

