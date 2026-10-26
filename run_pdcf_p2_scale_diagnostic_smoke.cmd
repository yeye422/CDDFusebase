@echo off
setlocal

cd /d "%~dp0"

.venv\Scripts\python.exe -u sweep_pdcf_p2_scales.py ^
  --c0-checkpoint models\reliability_v5\frequency_minimax_seed42\best.pth ^
  --p1-checkpoint models\pdcf_p1\predictability_seed42\best.pth ^
  --predictable-checkpoint models\pdcf_p2\predictable_residual_seed42_pilot1000_gradfix\best.pth ^
  --raw-checkpoint models\pdcf_p2\raw_evidence_seed42_pilot1000_gradfix\best.pth ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --split val ^
  --offset 0 ^
  --limit 1 ^
  --patch-size 256 ^
  --save-previews 1 ^
  --output-dir tmp\pdcf_p2_scale_diagnostic_smoke

if errorlevel 1 exit /b 1
echo PDCF P2 scale diagnostic smoke completed successfully.

