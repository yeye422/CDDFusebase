@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

.venv\Scripts\python.exe -u evaluate_mffa_v1_phase2.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --split val ^
  --offset 0 ^
  --limit 1000 ^
  --b0-checkpoint "models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth" ^
  --mffa-checkpoint "models\mffa_v1\phase2_seed42\best.pth" ^
  --save-images 8 ^
  --output-dir "test_result\mffa_v1\phase2_seed42_val1000_evalfix"

if errorlevel 1 exit /b %errorlevel%

echo.
echo MFFA v1 Phase-2 validation completed. Formal test1000 was not used.
endlocal
