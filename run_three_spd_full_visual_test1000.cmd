@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

set OUT1=test_result\v21_softmask\spd_nomask_on_b0_test1000_fullviz_v1
set OUT2=test_result\v21_softmask\spd_detail_only_retrained_on_b0_test1000_fullviz_v1
set OUT3=test_result\v21_softmask\spd_nomask_on_c0_test1000_fullviz_v1

if exist "%OUT1%" (
  echo ERROR: %OUT1% already exists. Refusing to overwrite.
  exit /b 2
)
if exist "%OUT2%" (
  echo ERROR: %OUT2% already exists. Refusing to overwrite.
  exit /b 2
)
if exist "%OUT3%" (
  echo ERROR: %OUT3% already exists. Refusing to overwrite.
  exit /b 2
)

echo [1/3] B0 + full SPD: test1000 with complete visual outputs...
.venv\Scripts\python.exe compare_spd_fusion.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --split test --offset 0 --limit 1000 ^
  --c0-checkpoint "models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth" ^
  --spd-checkpoint "models\v21_softmask\spd_nomask_seed42\best.pth" ^
  --spd-base-scale 1 --spd-detail-scale 1 ^
  --save-images 1000 --save-methods e2_feature ^
  --output-dir "%OUT1%"
if errorlevel 1 exit /b %errorlevel%

echo [2/3] B0 + retrained Detail-only SPD: test1000 with complete visual outputs...
.venv\Scripts\python.exe compare_spd_fusion.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --split test --offset 0 --limit 1000 ^
  --c0-checkpoint "models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth" ^
  --spd-checkpoint "models\v21_softmask\spd_detail_only_seed42\best.pth" ^
  --spd-base-scale 1 --spd-detail-scale 1 ^
  --save-images 1000 --save-methods e2_feature ^
  --output-dir "%OUT2%"
if errorlevel 1 exit /b %errorlevel%

echo [3/3] C0-minimax + full SPD: test1000 with complete visual outputs...
.venv\Scripts\python.exe compare_spd_fusion.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --split test --offset 0 --limit 1000 ^
  --c0-checkpoint "models\reliability_v5\frequency_minimax_seed42\epoch_005.pth" ^
  --spd-checkpoint "models\v21_softmask\spd_nomask_seed42\best.pth" ^
  --spd-base-scale 1 --spd-detail-scale 1 ^
  --save-images 1000 --save-methods e2_feature ^
  --output-dir "%OUT3%"
if errorlevel 1 exit /b %errorlevel%

echo.
echo All three complete-visualization test1000 runs finished.
echo Each output contains images/e2_feature, rgb/e2_feature, preview/e2_feature,
echo per_image.csv, and summary.json.
endlocal
