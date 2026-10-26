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
    --run-name %%M_seed42_pilot1000_gradfix ^
    --epochs 5 ^
    --batch-size 8 ^
    --patch-size 128 ^
    --train-limit 1000 ^
    --val-limit 200 ^
    --lr 1e-4 ^
    --weight-decay 1e-4 ^
    --hidden-channels 64 ^
    --blocks 2 ^
    --max-relative-correction 0.25 ^
    --anchor-weight 0 ^
    --correction-weight 0 ^
    --seed 42
  if errorlevel 1 exit /b 1
)

echo PDCF P2 matched pilots completed successfully.
