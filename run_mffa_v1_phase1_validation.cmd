@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

.venv\Scripts\python.exe -u evaluate_mffa_v1_phase1.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --split val ^
  --offset 0 ^
  --limit 1000 ^
  --checkpoint "models\mffa_v1\phase1_dtypefix_seed42\best.pth" ^
  --save-images 8 ^
  --output-dir "test_result\mffa_v1\phase1_dtypefix_seed42_val1000"

if errorlevel 1 exit /b %errorlevel%

echo.
echo MFFA v1 Phase-1 validation completed. Fusion training was not run.
endlocal
