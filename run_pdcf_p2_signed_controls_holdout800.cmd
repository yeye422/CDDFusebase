@echo off
setlocal

cd /d "%~dp0"

.venv\Scripts\python.exe -u evaluate_pdcf_signed_controls.py ^
  --c0-checkpoint models\reliability_v5\frequency_minimax_seed42\best.pth ^
  --p1-checkpoint models\pdcf_p1\predictability_seed42\best.pth ^
  --predictable-checkpoint models\pdcf_p2\predictable_residual_seed42_pilot1000_gradfix\best.pth ^
  --raw-checkpoint models\pdcf_p2\raw_evidence_seed42_pilot1000_gradfix\best.pth ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --split val ^
  --calibration-offset 0 ^
  --calibration-limit 200 ^
  --evaluation-offset 200 ^
  --evaluation-limit 800 ^
  --patch-size 256 ^
  --signed-scale -0.05 ^
  --unsharp-sigma 1.0 ^
  --calibration-initial-beta 0.01 ^
  --calibration-iterations 4 ^
  --calibration-tolerance 0.01 ^
  --bootstrap-repetitions 5000 ^
  --bootstrap-seed 20261022 ^
  --local-block-size 32 ^
  --local-gradient-budget 0.05 ^
  --save-previews 16 ^
  --output-dir test_result\pdcf_p2\signed_controls_seed42_val800_offset200_v1

if errorlevel 1 exit /b 1
echo PDCF signed controls holdout800 completed successfully.
