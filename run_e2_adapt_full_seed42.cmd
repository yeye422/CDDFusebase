@echo off
setlocal
cd /d "D:\#imageInfusion\CDDFuse"

set COMMON=--data-root "..\datasets\QXSLAB_SAROPT" --split-manifest "..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json" --c0-checkpoint "models\reliability_v5\frequency_minimax_seed42\epoch_005.pth" --spd-checkpoint "models\v21_softmask\spd_nomask_seed42\best.pth" --frequency-reference-config "models\reliability_v5\frequency_minimax_seed42\config.json" --output-dir "models\v21_softmask" --epochs 5 --batch-size 8 --patch-size 128 --lr 1e-5 --train-limit 8000 --val-limit 1000 --seed 42 --amp

echo Training E2-Adapt on 8000 train / 1000 validation pairs...
.venv\Scripts\python.exe -u train_spd_fusion_adapt.py --variant e2_adapt --run-name e2_adapt_full_seed42 %COMMON%
if errorlevel 1 exit /b %errorlevel%

echo.
echo Full E2-Adapt training completed.
echo Test split was not evaluated.
endlocal
