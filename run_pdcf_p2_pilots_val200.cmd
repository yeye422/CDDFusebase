@echo off
setlocal

cd /d "%~dp0"

.venv\Scripts\python.exe -u evaluate_pdcf_single_path.py ^
  --c0-checkpoint models\reliability_v5\frequency_minimax_seed42\best.pth ^
  --p1-checkpoint models\pdcf_p1\predictability_seed42\best.pth ^
  --predictable-checkpoint models\pdcf_p2\predictable_residual_seed42_pilot1000_gradfix\best.pth ^
  --raw-checkpoint models\pdcf_p2\raw_evidence_seed42_pilot1000_gradfix\best.pth ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --split val ^
  --offset 0 ^
  --limit 200 ^
  --patch-size 256 ^
  --save-previews 8 ^
  --output-dir test_result\pdcf_p2\matched_pilots_seed42_val200_gradfix

if errorlevel 1 exit /b 1
echo PDCF P2 matched pilot val200 evaluation completed successfully.
