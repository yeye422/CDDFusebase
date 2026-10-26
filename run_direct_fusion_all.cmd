@echo off
setlocal EnableExtensions

rem Train R0-R3 and evaluate every T1 checkpoint on the frozen validation split.
rem Usage:
rem   run_direct_fusion_all.cmd pilot [run_tag]
rem   run_direct_fusion_all.cmd full  [run_tag]
rem
rem pilot: fixed 1000 training pairs and first 200 validation pairs.
rem full:  all training pairs and all validation pairs.
rem The held-out test split is intentionally not evaluated by this script.

cd /d "%~dp0"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=pilot"

if /I "%MODE%"=="pilot" goto :configure_pilot
if /I "%MODE%"=="full" goto :configure_full

echo ERROR: unknown mode "%MODE%".
echo Usage: %~nx0 pilot [run_tag]
echo        %~nx0 full  [run_tag]
exit /b 2

:configure_pilot
set "TRAIN_LIMIT=1000"
set "VAL_LIMIT=200"
set "EVAL_LIMIT=200"
set "DEFAULT_TAG=direct_pilot_seed42_20261023"
goto :configured

:configure_full
set "TRAIN_LIMIT=0"
set "VAL_LIMIT=0"
set "EVAL_LIMIT=300"
set "DEFAULT_TAG=direct_full_seed42_20261023"
goto :configured

:configured
set "RUN_TAG=%~2"
if "%RUN_TAG%"=="" set "RUN_TAG=%DEFAULT_TAG%"

set "PYTHON=.venv\Scripts\python.exe"
set "DATA_ROOT=..\datasets\Dongying"
set "INIT_CHECKPOINT=models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth"
set "MODEL_ROOT=models\direct_fusion_matrix\%RUN_TAG%"
set "EVAL_ROOT=test_result\direct_fusion_matrix\%RUN_TAG%\val"

if not exist "%PYTHON%" (
    echo ERROR: Python executable not found: %PYTHON%
    exit /b 3
)
if not exist "%DATA_ROOT%\GF2" (
    echo ERROR: Optical directory not found: %DATA_ROOT%\GF2
    exit /b 3
)
if not exist "%DATA_ROOT%\GF3" (
    echo ERROR: SAR directory not found: %DATA_ROOT%\GF3
    exit /b 3
)
if not exist "%INIT_CHECKPOINT%" (
    echo ERROR: initialization checkpoint not found: %INIT_CHECKPOINT%
    exit /b 3
)
if exist "%MODEL_ROOT%" echo Continuing existing model root: %MODEL_ROOT%
if exist "%EVAL_ROOT%" echo Continuing existing evaluation root: %EVAL_ROOT%

echo ============================================================
echo Direct-fusion matrix
echo mode:       %MODE%
echo run tag:    %RUN_TAG%
echo train limit:%TRAIN_LIMIT%
echo val limit:  %VAL_LIMIT%
echo model root: %MODEL_ROOT%
echo eval root:  %EVAL_ROOT%
echo ============================================================

for %%V in (r0 r1 r2 r3) do (
    call :train_variant %%V
    if errorlevel 1 goto :failed
)

for %%V in (r0 r1 r2 r3) do (
    call :validate_variant %%V
    if errorlevel 1 goto :failed
)

echo.
echo [SUMMARY] collecting all validation metrics
"%PYTHON%" summarize_direct_fusion_matrix.py ^
  --eval-root "%EVAL_ROOT%" ^
  --output "%EVAL_ROOT%\summary.csv"
if errorlevel 1 goto :failed

echo ============================================================
echo All R0-R3 training and validation evaluations completed.
echo Models:      %MODEL_ROOT%
echo Evaluations: %EVAL_ROOT%
echo Summary:     %EVAL_ROOT%\summary.csv
echo Select the final checkpoint from validation results before
echo running any held-out or formal test protocol.
echo ============================================================
exit /b 0

:train_variant
set "VARIANT=%~1"
if exist "%MODEL_ROOT%\%VARIANT%\epoch_006.pth" (
    echo.
    echo [SKIP TRAIN] %VARIANT% already has epoch_006.pth
    exit /b 0
)
set "LATEST_CHECKPOINT="
if exist "%MODEL_ROOT%\%VARIANT%" (
    for /f "delims=" %%C in ('dir /b /a-d /o:n "%MODEL_ROOT%\%VARIANT%\epoch_*.pth" 2^>nul') do set "LATEST_CHECKPOINT=%MODEL_ROOT%\%VARIANT%\%%C"
)
if defined LATEST_CHECKPOINT (
    set "RECOVERY_OPTION=--resume"
    set RECOVERY_VALUE="%LATEST_CHECKPOINT%"
) else (
    set "RECOVERY_OPTION=--allow-incomplete-restart"
    set "RECOVERY_VALUE="
)
echo.
echo [TRAIN] %VARIANT%
if defined LATEST_CHECKPOINT echo [RESUME] %LATEST_CHECKPOINT%
"%PYTHON%" train_direct_fusion.py ^
  --variant %VARIANT% ^
  --data-root "%DATA_ROOT%" ^
  --optical-dir GF2 ^
  --sar-dir GF3 ^
  --test-count 150 ^
  --val-count 300 ^
  --train-limit %TRAIN_LIMIT% ^
  --val-limit %VAL_LIMIT% ^
  --patch-size 256 ^
  --batch-size 2 ^
  --accumulation-steps 4 ^
  --num-workers 0 ^
  --invalid-threshold 0 ^
  --min-valid-fraction 0.90 ^
  --seed 42 ^
  --t0-epochs 1 ^
  --t1-epochs 5 ^
  --t0-decoder-lr 1e-4 ^
  --t1-fusion-lr 1e-4 ^
  --t1-decoder-lr 2e-5 ^
  --weight-decay 1e-4 ^
  --clip-grad 1 ^
  --gradient-weight 10 ^
  --reconstruction-weight 0.1 ^
  --init-checkpoint "%INIT_CHECKPOINT%" ^
  --output-dir "%MODEL_ROOT%" ^
  --run-name %VARIANT% ^
  --device cuda ^
  %RECOVERY_OPTION% %RECOVERY_VALUE%
if errorlevel 1 exit /b 1
exit /b 0

:validate_variant
set "VARIANT=%~1"
echo.
echo [VALIDATE] %VARIANT% epochs 002-006
for %%E in (002 003 004 005 006) do (
    call :validate_checkpoint %VARIANT% %%E
    if errorlevel 1 exit /b 1
)
exit /b 0

:validate_checkpoint
set "VARIANT=%~1"
set "EPOCH=%~2"
if exist "%EVAL_ROOT%\%VARIANT%_epoch_%EPOCH%\metrics.csv" if exist "%EVAL_ROOT%\%VARIANT%_epoch_%EPOCH%\run_metadata.json" (
    echo [SKIP VALIDATE] %VARIANT% epoch_%EPOCH%.pth already evaluated
    exit /b 0
)
echo [VALIDATE] %VARIANT% epoch_%EPOCH%.pth
"%PYTHON%" test_OS.py ^
  --variant %VARIANT% ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%MODEL_ROOT%\%VARIANT%\split_manifest.json" ^
  --split val ^
  --limit %EVAL_LIMIT% ^
  --checkpoint "%MODEL_ROOT%\%VARIANT%\epoch_%EPOCH%.pth" ^
  --output-dir "%EVAL_ROOT%\%VARIANT%_epoch_%EPOCH%" ^
  --metrics-only ^
  --metric-profile paper ^
  --output-range-policy clamp ^
  --timing-warmup 10
if errorlevel 1 exit /b 1
exit /b 0

:failed
echo.
echo ERROR: pipeline stopped after a failed command.
exit /b 1
