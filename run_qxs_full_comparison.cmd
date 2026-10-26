@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

echo [1/2] Evaluating official CDDFuse zero-shot with clamp on test1000...
.venv\Scripts\python.exe test_OS.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --split test ^
  --offset 0 ^
  --limit 1000 ^
  --checkpoint "models\CDDFuse_IVF.pth" ^
  --variant original ^
  --metric-profile paper ^
  --output-range-policy clamp ^
  --timing-warmup 10 ^
  --metrics-only ^
  --output-dir "test_result\comparison_strict\cddfuse_zero_shot_clamp_test1000"
if errorlevel 1 exit /b %errorlevel%

echo [2/2] Building full 14-metric and paired edge reports...
.venv\Scripts\python.exe build_qxs_full_comparison.py ^
  --zero-shot-clamp "test_result\comparison_strict\cddfuse_zero_shot_clamp_test1000\metrics.csv" ^
  --output-dir "test_result\comparison_strict\full_qxs_test1000"
if errorlevel 1 exit /b %errorlevel%

echo.
echo Full strict comparison completed.
echo Results: test_result\comparison_strict\full_qxs_test1000
endlocal
