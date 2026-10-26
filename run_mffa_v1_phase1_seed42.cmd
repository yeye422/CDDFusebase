@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

.venv\Scripts\python.exe -u train_mffa_v1_phase1.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --output-dir "models\mffa_v1" ^
  --run-name "phase1_dtypefix_seed42" ^
  --epochs 20 ^
  --batch-size 4 ^
  --patch-size 128 ^
  --lr 1e-4 ^
  --band-weight 0.25 ^
  --ssim-weight 0.5 ^
  --max-relative-sar 0.05 ^
  --train-limit 8000 ^
  --val-limit 1000 ^
  --num-workers 0 ^
  --seed 42 ^
  --amp

if errorlevel 1 exit /b %errorlevel%

echo.
echo MFFA v1 Phase-1 training completed.
echo Results: models\mffa_v1\phase1_dtypefix_seed42
endlocal
