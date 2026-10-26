@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

.venv\Scripts\python.exe -u evaluate_optical_rescue.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --split val ^
  --offset 0 ^
  --limit 1000 ^
  --b0-checkpoint "models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth" ^
  --spd-checkpoint "models\v21_softmask\spd_detail_only_seed42\best.pth" ^
  --rescue-checkpoint "models\optical_rescue\detail_spd_gradfix_seed42\best.pth" ^
  --save-images 8 ^
  --output-dir "test_result\optical_rescue\detail_spd_gradfix_seed42_val1000"

if errorlevel 1 exit /b %errorlevel%

echo.
echo Optical Rescue validation1000 completed. Formal test was not used.
endlocal
