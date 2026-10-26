@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

.venv\Scripts\python.exe -u sweep_spd_line_gated_detail.py ^
  --data-root "..\datasets\QXSLAB_SAROPT" ^
  --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" ^
  --split val ^
  --offset 0 ^
  --limit 1000 ^
  --b0-checkpoint "models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth" ^
  --spd-checkpoint "models\v21_softmask\spd_detail_only_seed42\best.pth" ^
  --line-calibration "..\datasets\SAR despeckling filters dataset\derived_masks_v1\calibration.json" ^
  --confidence-floor 0.50 ^
  --min-area 32 ^
  --min-extent 16 ^
  --min-elongation 2.0 ^
  --large-area 128 ^
  --preserve-strengths 0 0.25 0.5 1.0 ^
  --save-images 8 ^
  --output-dir "test_result\v21_softmask\spd_line_gated_detail_val1000"

if errorlevel 1 exit /b %errorlevel%

echo.
echo Validation-only line-gated Detail sweep completed.
echo Results: test_result\v21_softmask\spd_line_gated_detail_val1000
endlocal
