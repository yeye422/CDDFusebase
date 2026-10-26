@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

.venv\Scripts\python.exe -u train_optical_rescue.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --b0-checkpoint "models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth" ^
  --spd-checkpoint "models\v21_softmask\spd_detail_only_seed42\best.pth" ^
  --output-dir "models\optical_rescue" ^
  --run-name "detail_spd_gradfix_seed42" ^
  --epochs 6 ^
  --head-warmup-epochs 1 ^
  --batch-size 8 ^
  --patch-size 128 ^
  --lr 1e-4 ^
  --head-lr 1e-4 ^
  --max-relative-correction 0.005 ^
  --train-limit 8000 ^
  --val-limit 1000 ^
  --num-workers 0 ^
  --seed 42 ^
  --amp

if errorlevel 1 exit /b %errorlevel%

echo.
echo Optical Rescue training completed.
echo Results: models\optical_rescue\detail_spd_gradfix_seed42
endlocal
