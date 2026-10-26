@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

set ROOT=D:\#imageInfusion\datasets\SAR despeckling filters dataset\SAR despeckling filters dataset\Main folder
set OUT=..\datasets\SAR despeckling filters dataset\derived_masks_v1

echo [1/2] Regenerating training masks with high-confidence mapping...
.venv\Scripts\python.exe prepare_spd_line_masks.py generate ^
  --noisy-dir "%ROOT%\Noisy" ^
  --reference-dir "%ROOT%\GTruth" ^
  --calibration "%OUT%\calibration.json" ^
  --output-dir "%OUT%\train" ^
  --confidence-floor 0.5 ^
  --preview-count 24 ^
  --perturb-sigma 0.03 ^
  --seed 42
if errorlevel 1 exit /b %errorlevel%

echo [2/2] Regenerating validation masks with high-confidence mapping...
.venv\Scripts\python.exe prepare_spd_line_masks.py generate ^
  --noisy-dir "%ROOT%\Noisy_val" ^
  --reference-dir "%ROOT%\GTruth_val" ^
  --calibration "%OUT%\calibration.json" ^
  --output-dir "%OUT%\validation" ^
  --confidence-floor 0.5 ^
  --preview-count 24 ^
  --perturb-sigma 0.03 ^
  --seed 4242
if errorlevel 1 exit /b %errorlevel%

echo.
echo High-confidence line masks completed.
echo Training masks: %OUT%\train\line_joint_high
echo Validation masks: %OUT%\validation\line_joint_high
endlocal
