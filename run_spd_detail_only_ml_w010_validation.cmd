@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

echo [1/2] Evaluating Detail-only SPD + M_L w0.10 on SAR internal test50...
.venv\Scripts\python.exe evaluate_spd.py ^
  --noisy-dir "D:\#imageInfusion\datasets\SAR despeckling filters dataset\SAR despeckling filters dataset\Main folder\Noisy_val" ^
  --reference-dir "D:\#imageInfusion\datasets\SAR despeckling filters dataset\SAR despeckling filters dataset\Main folder\GTruth_val" ^
  --spd-checkpoint "models\v21_softmask\spd_detail_only_ml_w010_seed42\best.pth" ^
  --encoder-checkpoint "models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth" ^
  --split test ^
  --official-val-test-fraction 0.5 ^
  --tile-size 128 ^
  --tile-overlap 32 ^
  --save-images 8 ^
  --output-dir "test_result\v21_softmask\spd_detail_only_ml_w010_seed42_internal_test50"
if errorlevel 1 exit /b %errorlevel%

echo [2/2] Evaluating Detail-only SPD + M_L w0.10 with B0 on QXS validation1000...
.venv\Scripts\python.exe compare_spd_fusion.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --split val ^
  --offset 0 ^
  --limit 1000 ^
  --c0-checkpoint "models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth" ^
  --spd-checkpoint "models\v21_softmask\spd_detail_only_ml_w010_seed42\best.pth" ^
  --spd-base-scale 1 ^
  --spd-detail-scale 1 ^
  --save-images 8 ^
  --output-dir "test_result\v21_softmask\spd_detail_only_ml_w010_on_b0_val1000"
if errorlevel 1 exit /b %errorlevel%

echo.
echo Detail-only SPD + M_L w0.10 validation completed. Formal test1000 was not used.
endlocal
