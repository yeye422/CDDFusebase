@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

.venv\Scripts\python.exe -u train_ocaf_v2.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --b0-checkpoint "models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth" ^
  --spd-checkpoint "models\v21_softmask\spd_detail_only_seed42\best.pth" ^
  --output-dir "models\ocaf_v2" ^
  --run-name "independent_amplitude_seed42" ^
  --epochs 10 ^
  --batch-size 6 ^
  --patch-size 128 ^
  --lr 1e-4 ^
  --max-common-amplitude 0.0078431373 ^
  --max-exclusive-mid-amplitude 0.0058823529 ^
  --max-exclusive-high-amplitude 0.0039215686 ^
  --target-common-rms 0.0019607843 ^
  --target-exclusive-rms 0.0009803922 ^
  --perturb-sigma 0.03 ^
  --train-limit 8000 ^
  --val-limit 1000 ^
  --num-workers 0 ^
  --seed 42 ^
  --amp

if errorlevel 1 exit /b %errorlevel%

echo.
echo OCAF-v2 training completed.
echo Results: models\ocaf_v2\independent_amplitude_seed42
endlocal
