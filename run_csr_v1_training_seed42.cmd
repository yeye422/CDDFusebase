@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

.venv\Scripts\python.exe -u train_csr_v1.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --b0-checkpoint "models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth" ^
  --spd-checkpoint "models\v21_softmask\spd_detail_only_seed42\best.pth" ^
  --output-dir "models\csr_v1" ^
  --run-name "detail_spd_seed42" ^
  --epochs 8 ^
  --batch-size 8 ^
  --patch-size 128 ^
  --lr 1e-4 ^
  --max-mid-replacement 0.10 ^
  --max-high-replacement 0.03 ^
  --mid-sigma 3.0 ^
  --high-sigma 1.0 ^
  --train-limit 8000 ^
  --val-limit 1000 ^
  --num-workers 0 ^
  --seed 42 ^
  --amp

if errorlevel 1 exit /b %errorlevel%

echo.
echo CSR-v1 training completed.
echo Results: models\csr_v1\detail_spd_seed42
endlocal
