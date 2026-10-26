@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

.venv\Scripts\python.exe -u train_fusion_p1.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --init-checkpoint "models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth" ^
  --output-dir "models\fusion_p1" ^
  --run-name "stable_detail_seed42" ^
  --epochs 5 ^
  --batch-size 8 ^
  --patch-size 128 ^
  --lr 1e-4 ^
  --perturb-sigma 0.03 ^
  --teacher-weight 1.0 ^
  --fusion-weight 0.25 ^
  --consistency-weight 0.05 ^
  --baseline-anchor-weight 0.05 ^
  --train-limit 8000 ^
  --val-limit 1000 ^
  --num-workers 0 ^
  --seed 42 ^
  --amp

if errorlevel 1 exit /b %errorlevel%

echo.
echo P1 stable Detail fusion training completed.
echo Results: models\fusion_p1\stable_detail_seed42
endlocal
