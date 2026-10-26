@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

.venv\Scripts\python.exe -u evaluate_fusion_p2a.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --split val ^
  --offset 0 ^
  --limit 1000 ^
  --b0-checkpoint "models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth" ^
  --p2a-checkpoint "models\fusion_p2a\source_preserving_seed42\best.pth" ^
  --save-images 8 ^
  --output-dir "test_result\fusion_p2a\source_preserving_seed42_val1000"

if errorlevel 1 exit /b %errorlevel%

echo.
echo P2-A validation1000 completed. Formal test1000 was not used.
endlocal
