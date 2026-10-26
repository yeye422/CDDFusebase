@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

.venv\Scripts\python.exe sweep_spd_feature_scales.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --split val ^
  --offset 0 ^
  --limit 1000 ^
  --b0-checkpoint "models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth" ^
  --spd-checkpoint "models\v21_softmask\spd_nomask_seed42\best.pth" ^
  --output-dir "test_result\v21_softmask\spd_scale_sweep_b0_val1000"

if errorlevel 1 exit /b %errorlevel%

echo.
echo SPD Base/Detail scale sweep completed on validation 1000.
echo Results: test_result\v21_softmask\spd_scale_sweep_b0_val1000
endlocal
