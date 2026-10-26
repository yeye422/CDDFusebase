@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

.venv\Scripts\python.exe -u train_mffa_v1_phase2.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --phase1-checkpoint "models\mffa_v1\phase1_dtypefix_seed42\best.pth" ^
  --frequency-reference-config "models\reliability_v5\frequency_minimax_seed42\config.json" ^
  --output-dir "models\mffa_v1" ^
  --run-name "phase2_seed42" ^
  --epochs 5 ^
  --batch-size 4 ^
  --patch-size 128 ^
  --lr 1e-5 ^
  --clip-grad 0.01 ^
  --train-limit 8000 ^
  --val-limit 1000 ^
  --num-workers 0 ^
  --seed 42 ^
  --amp

if errorlevel 1 exit /b %errorlevel%

echo.
echo MFFA v1 Phase-2 fusion training completed.
echo Results: models\mffa_v1\phase2_seed42
endlocal
