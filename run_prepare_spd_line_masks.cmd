@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

set ROOT=D:\#imageInfusion\datasets\SAR despeckling filters dataset\SAR despeckling filters dataset\Main folder
set OUT=..\datasets\SAR despeckling filters dataset\derived_masks_v1

echo [1/3] Calibrating joint line evidence on training pairs...
.venv\Scripts\python.exe prepare_spd_line_masks.py calibrate ^
  --noisy-dir "%ROOT%\Noisy" ^
  --reference-dir "%ROOT%\GTruth" ^
  --sample-count 512 ^
  --pixels-per-image 2048 ^
  --flat-quantile 0.40 ^
  --threshold-quantile 0.995 ^
  --transition-quantile 0.95 ^
  --scales 3 5 9 ^
  --seed 42 ^
  --output "%OUT%\calibration.json"
if errorlevel 1 exit /b %errorlevel%

echo [2/3] Generating training masks and audit previews...
.venv\Scripts\python.exe prepare_spd_line_masks.py generate ^
  --noisy-dir "%ROOT%\Noisy" ^
  --reference-dir "%ROOT%\GTruth" ^
  --calibration "%OUT%\calibration.json" ^
  --output-dir "%OUT%\train" ^
  --preview-count 24 ^
  --perturb-sigma 0.03 ^
  --seed 42
if errorlevel 1 exit /b %errorlevel%

echo [3/3] Generating validation masks and audit previews...
.venv\Scripts\python.exe prepare_spd_line_masks.py generate ^
  --noisy-dir "%ROOT%\Noisy_val" ^
  --reference-dir "%ROOT%\GTruth_val" ^
  --calibration "%OUT%\calibration.json" ^
  --output-dir "%OUT%\validation" ^
  --preview-count 24 ^
  --perturb-sigma 0.03 ^
  --seed 4242
if errorlevel 1 exit /b %errorlevel%

echo.
echo SPD line-mask preparation completed.
echo Results: %OUT%
endlocal
