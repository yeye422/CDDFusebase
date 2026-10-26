@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

.venv\Scripts\python.exe -u train_spd.py ^
  --spd-variant selective_detail ^
  --selective-scale-kernels 3 5 9 ^
  --selective-correction-highpass-kernel 5 ^
  --noisy-dir "D:\#imageInfusion\datasets\SAR despeckling filters dataset\SAR despeckling filters dataset\Main folder\Noisy" ^
  --reference-dir "D:\#imageInfusion\datasets\SAR despeckling filters dataset\SAR despeckling filters dataset\Main folder\GTruth" ^
  --val-noisy-dir "D:\#imageInfusion\datasets\SAR despeckling filters dataset\SAR despeckling filters dataset\Main folder\Noisy_val" ^
  --val-reference-dir "D:\#imageInfusion\datasets\SAR despeckling filters dataset\SAR despeckling filters dataset\Main folder\GTruth_val" ^
  --official-val-test-fraction 0.5 ^
  --init-checkpoint "models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth" ^
  --output-dir "models\v21_softmask" ^
  --run-name "spd_selective_detail_seed42" ^
  --epochs 30 ^
  --batch-size 8 ^
  --patch-size 128 ^
  --lr 1e-4 ^
  --base-feature-weight 0 ^
  --detail-feature-weight 0.05 ^
  --consistency-weight 0.1 ^
  --consistency-sigma 0.03 ^
  --invalid-threshold -1 ^
  --num-workers 0 ^
  --seed 42 ^
  --amp

if errorlevel 1 exit /b %errorlevel%

echo.
echo Selective-Detail SPD training completed.
echo Results: models\v21_softmask\spd_selective_detail_seed42
endlocal
