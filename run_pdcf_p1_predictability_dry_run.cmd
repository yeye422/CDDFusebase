@echo off
setlocal

cd /d "%~dp0"

.venv\Scripts\python.exe -u train_pdcf_predictability.py ^
  --c0-checkpoint models\reliability_v5\frequency_minimax_seed42\best.pth ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --batch-size 2 ^
  --patch-size 128 ^
  --train-limit 24 ^
  --val-limit 24 ^
  --dry-run

if errorlevel 1 exit /b 1
echo PDCF P1 predictability dry-run completed successfully.

