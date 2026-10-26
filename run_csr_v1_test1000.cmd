@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

echo Run only after CSR-v1 validation settings are frozen and accepted.
.venv\Scripts\python.exe -u evaluate_csr_v1.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --split test ^
  --offset 0 ^
  --limit 1000 ^
  --b0-checkpoint "models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth" ^
  --spd-checkpoint "models\v21_softmask\spd_detail_only_seed42\best.pth" ^
  --csr-checkpoint "models\csr_v1\detail_spd_seed42\best.pth" ^
  --save-images 1000 ^
  --output-dir "test_result\csr_v1\detail_spd_seed42_test1000"

if errorlevel 1 exit /b %errorlevel%

echo.
echo CSR-v1 formal test1000 completed. Register it before rendering the index.
endlocal
