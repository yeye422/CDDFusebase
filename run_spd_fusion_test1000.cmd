@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

.venv\Scripts\python.exe compare_spd_fusion.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --split test ^
  --offset 0 ^
  --limit 1000 ^
  --c0-checkpoint "models\reliability_v5\frequency_minimax_seed42\epoch_005.pth" ^
  --spd-checkpoint "models\v21_softmask\spd_nomask_seed42\best.pth" ^
  --save-images 1000 ^
  --output-dir "test_result\v21_softmask\spd_nomask_fusion_test1000_c0full"

if errorlevel 1 exit /b %errorlevel%

echo.
echo SPD formal test1000 evaluation completed.
echo Results: test_result\v21_softmask\spd_nomask_fusion_test1000_c0full
endlocal
