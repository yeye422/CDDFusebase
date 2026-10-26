@echo off
setlocal

cd /d "%~dp0"

for %%M in (predictable_residual raw_evidence) do (
  .venv\Scripts\python.exe -u train_pdcf_single_path.py ^
    --input-mode %%M ^
    --c0-checkpoint models\reliability_v5\frequency_minimax_seed42\best.pth ^
    --p1-checkpoint models\pdcf_p1\predictability_seed42\best.pth ^
    --data-root ..\datasets\QXSLAB_SAROPT ^
    --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
    --output-dir models\pdcf_p2 ^
    --run-name dry_run_%%M ^
    --batch-size 2 ^
    --train-limit 24 ^
    --val-limit 24 ^
    --max-train-batches 1 ^
    --max-val-batches 1 ^
    --dry-run
  if errorlevel 1 exit /b 1
)

echo PDCF P2 dual dry-run completed successfully.

