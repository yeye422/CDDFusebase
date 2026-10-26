@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

.venv\Scripts\python.exe -u train_fusion_p2a.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --init-checkpoint "models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth" ^
  --output-dir "models\fusion_p2a" ^
  --run-name "source_preserving_seed42" ^
  --epochs 6 ^
  --head-warmup-epochs 1 ^
  --batch-size 8 ^
  --patch-size 128 ^
  --lr 1e-4 ^
  --head-lr 1e-4 ^
  --max-relative-correction 0.03 ^
  --optical-feature-weight 0.25 ^
  --sar-feature-weight 0.25 ^
  --fusion-weight 0.25 ^
  --invariance-weight 0.05 ^
  --baseline-anchor-weight 0.05 ^
  --residual-weight 0.01 ^
  --perturb-sigma 0.03 ^
  --train-limit 8000 ^
  --val-limit 1000 ^
  --num-workers 0 ^
  --seed 42 ^
  --amp

if errorlevel 1 exit /b %errorlevel%

echo.
echo P2-A source-preserving fusion training completed.
echo Results: models\fusion_p2a\source_preserving_seed42
endlocal
