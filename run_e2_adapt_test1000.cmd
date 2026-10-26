@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

.venv\Scripts\python.exe compare_spd_fusion.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --split test ^
  --offset 0 ^
  --limit 1000 ^
  --c0-checkpoint "models\v21_softmask\e2_adapt_full_seed42\best.pth" ^
  --spd-checkpoint "models\v21_softmask\spd_nomask_seed42\best.pth" ^
  --save-images 1000 ^
  --output-dir "test_result\v21_softmask\e2_adapt_full_seed42_test1000"

if errorlevel 1 exit /b %errorlevel%

echo.
echo E2-Adapt formal test1000 completed.
echo Final E2-Adapt metrics are under methods.e2_feature in summary.json.
endlocal
