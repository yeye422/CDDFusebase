@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

.venv\Scripts\python.exe -u audit_fusion_feature_p0.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --split val ^
  --offset 0 ^
  --limit 200 ^
  --checkpoint "models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth" ^
  --perturb-sigma 0.03 ^
  --local-radius 2 ^
  --edge-quantile 0.80 ^
  --flat-quantile 0.50 ^
  --save-maps 8 ^
  --seed 42 ^
  --output-dir "audit_result\fusion_feature_p0\b0_val200_seed42"

if errorlevel 1 exit /b %errorlevel%

echo.
echo Frozen-feature P0 audit completed.
echo Results: audit_result\fusion_feature_p0\b0_val200_seed42
endlocal
