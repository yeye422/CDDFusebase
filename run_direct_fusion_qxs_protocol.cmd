@echo off
setlocal EnableExtensions

rem Strict QXS protocol for direct-fusion R0-R3.
rem
rem Stage 1 - train all controls and evaluate validation only:
rem   run_direct_fusion_qxs_protocol.cmd train-val [run_tag]
rem
rem Matched extension - add three T1 epochs to every R0-R3 model:
rem   run_direct_fusion_qxs_protocol.cmd extend-val [run_tag]
rem
rem Stage 2 - after freezing exactly one validation-selected checkpoint:
rem   run_direct_fusion_qxs_protocol.cmd test run_tag variant epoch
rem Example:
rem   run_direct_fusion_qxs_protocol.cmd test qxs_direct_seed42_v1 r3 004

cd /d "%~dp0"

set "MODE=%~1"
if "%MODE%"=="" goto :usage

set "PYTHON=.venv\Scripts\python.exe"
set "DATA_ROOT=..\datasets\QXSLAB_SAROPT"
set "SPLIT_MANIFEST=..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json"
set "INIT_CHECKPOINT=models\qxslab_saropt_from_ivf\CDDFuse_OS_20260924_012025\best.pth"

if not exist "%PYTHON%" (
    echo ERROR: Python executable not found: %PYTHON%
    exit /b 3
)
if not exist "%DATA_ROOT%" (
    echo ERROR: QXS data root not found: %DATA_ROOT%
    exit /b 3
)
if not exist "%SPLIT_MANIFEST%" (
    echo ERROR: fixed split manifest not found: %SPLIT_MANIFEST%
    exit /b 3
)
if not exist "%INIT_CHECKPOINT%" (
    echo ERROR: initialization checkpoint not found: %INIT_CHECKPOINT%
    exit /b 3
)

if /I "%MODE%"=="train-val" goto :train_val
if /I "%MODE%"=="extend-val" goto :extend_val
if /I "%MODE%"=="test" goto :formal_test
goto :usage

:train_val
set "RUN_TAG=%~2"
if "%RUN_TAG%"=="" set "RUN_TAG=qxs_direct_seed42_v1"
set "MODEL_ROOT=models\direct_fusion_qxs\%RUN_TAG%"
set "EVAL_ROOT=test_result\direct_fusion_qxs\%RUN_TAG%\val"
set "TARGET_EPOCH=006"
set "T1_EPOCHS=5"
set "EPOCH_LIST=002 003 004 005 006"
set "EXPECTED_RESULTS=20"

echo ============================================================
echo QXS direct-fusion train/validation protocol
echo run tag:    %RUN_TAG%
echo train:      fixed train8000
echo validation: fixed val1000
echo test:       disabled in this stage
echo model root: %MODEL_ROOT%
echo eval root:  %EVAL_ROOT%
echo ============================================================

if exist "%MODEL_ROOT%" echo Continuing existing model root: %MODEL_ROOT%
if exist "%EVAL_ROOT%" echo Continuing existing evaluation root: %EVAL_ROOT%

for %%V in (r0 r1 r2 r3) do (
    call :train_variant %%V
    if errorlevel 1 goto :failed
)

for %%V in (r0 r1 r2 r3) do (
    call :validate_variant %%V
    if errorlevel 1 goto :failed
)

echo.
echo [SUMMARY] collecting all QXS val1000 metrics
"%PYTHON%" summarize_direct_fusion_matrix.py ^
  --eval-root "%EVAL_ROOT%" ^
  --output "%EVAL_ROOT%\summary.csv"
if errorlevel 1 goto :failed

echo ============================================================
echo QXS train8000 and val1000 stages completed.
echo Summary: %EVAL_ROOT%\summary.csv
echo No test image has been evaluated.
echo Freeze one variant and epoch using validation results, then run:
echo %~nx0 test %RUN_TAG% variant epoch
echo Example: %~nx0 test %RUN_TAG% r3 004
echo ============================================================
exit /b 0

:extend_val
set "RUN_TAG=%~2"
if "%RUN_TAG%"=="" set "RUN_TAG=qxs_direct_seed42_v1"
set "MODEL_ROOT=models\direct_fusion_qxs\%RUN_TAG%"
set "EVAL_ROOT=test_result\direct_fusion_qxs\%RUN_TAG%\val_extended"
set "TARGET_EPOCH=009"
set "T1_EPOCHS=8"
set "EPOCH_LIST=007 008 009"
set "EXPECTED_RESULTS=12"

if not exist "%MODEL_ROOT%\r0\epoch_006.pth" (
    echo ERROR: base train-val stage is incomplete for %RUN_TAG%.
    echo Run: %~nx0 train-val %RUN_TAG%
    exit /b 5
)

echo ============================================================
echo QXS direct-fusion matched extension
echo run tag:    %RUN_TAG%
echo models:     R0, R1, R2, R3
echo extension:  three additional T1 epochs
echo checkpoints: epoch_007 through epoch_009
echo validation: fixed val1000, metrics only
echo test:       disabled in this stage
echo ============================================================

if exist "%EVAL_ROOT%" echo Continuing existing extension evaluation: %EVAL_ROOT%

for %%V in (r0 r1 r2 r3) do (
    call :train_variant %%V
    if errorlevel 1 goto :failed
)

for %%V in (r0 r1 r2 r3) do (
    call :validate_variant %%V
    if errorlevel 1 goto :failed
)

echo.
echo [SUMMARY] collecting extended QXS val1000 metrics
"%PYTHON%" summarize_direct_fusion_matrix.py ^
  --eval-root "%EVAL_ROOT%" ^
  --output "%EVAL_ROOT%\summary.csv" ^
  --expected-count %EXPECTED_RESULTS%
if errorlevel 1 goto :failed

echo ============================================================
echo Matched extension and val1000 evaluation completed.
echo Summary: %EVAL_ROOT%\summary.csv
echo No test image has been evaluated.
echo ============================================================
exit /b 0

:train_variant
set "VARIANT=%~1"
if exist "%MODEL_ROOT%\%VARIANT%\epoch_%TARGET_EPOCH%.pth" (
    echo.
    echo [SKIP TRAIN] %VARIANT% already has epoch_%TARGET_EPOCH%.pth
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
echo [TRAIN QXS] %VARIANT%
if defined LATEST_CHECKPOINT echo [RESUME] %LATEST_CHECKPOINT%
"%PYTHON%" train_direct_fusion.py ^
  --variant %VARIANT% ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --train-limit 0 ^
  --val-limit 0 ^
  --patch-size 256 ^
  --batch-size 2 ^
  --accumulation-steps 4 ^
  --num-workers 0 ^
  --invalid-threshold 0 ^
  --min-valid-fraction 0.90 ^
  --seed 42 ^
  --t0-epochs 1 ^
  --t1-epochs %T1_EPOCHS% ^
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
echo [VALIDATE QXS] %VARIANT% epochs %EPOCH_LIST% on fixed val1000
for %%E in (%EPOCH_LIST%) do (
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
echo [VALIDATE QXS] %VARIANT% epoch_%EPOCH%.pth
"%PYTHON%" test_OS.py ^
  --variant %VARIANT% ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --split val ^
  --limit 1000 ^
  --checkpoint "%MODEL_ROOT%\%VARIANT%\epoch_%EPOCH%.pth" ^
  --output-dir "%EVAL_ROOT%\%VARIANT%_epoch_%EPOCH%" ^
  --metrics-only ^
  --metric-profile paper ^
  --output-range-policy clamp ^
  --timing-warmup 10
if errorlevel 1 exit /b 1
exit /b 0

:formal_test
set "RUN_TAG=%~2"
set "SELECTED_VARIANT=%~3"
set "SELECTED_EPOCH=%~4"
if "%RUN_TAG%"=="" goto :test_usage
if "%SELECTED_VARIANT%"=="" goto :test_usage
if "%SELECTED_EPOCH%"=="" goto :test_usage

if /I "%SELECTED_VARIANT%"=="r0" goto :variant_ok
if /I "%SELECTED_VARIANT%"=="r1" goto :variant_ok
if /I "%SELECTED_VARIANT%"=="r2" goto :variant_ok
if /I "%SELECTED_VARIANT%"=="r3" goto :variant_ok
echo ERROR: selected variant must be r0, r1, r2, or r3.
exit /b 2

:variant_ok
set "MODEL_ROOT=models\direct_fusion_qxs\%RUN_TAG%"
set "EVAL_ROOT=test_result\direct_fusion_qxs\%RUN_TAG%\val"
set "SELECTED_CHECKPOINT=%MODEL_ROOT%\%SELECTED_VARIANT%\epoch_%SELECTED_EPOCH%.pth"
set "TEST_ROOT=test_result\direct_fusion_qxs\%RUN_TAG%\formal_test"

if not exist "%EVAL_ROOT%\summary.csv" (
    echo ERROR: validation summary is missing: %EVAL_ROOT%\summary.csv
    echo Complete train-val and freeze the checkpoint before formal testing.
    exit /b 5
)
if not exist "%EVAL_ROOT%\%SELECTED_VARIANT%_epoch_%SELECTED_EPOCH%\metrics.csv" if not exist "test_result\direct_fusion_qxs\%RUN_TAG%\val_extended\%SELECTED_VARIANT%_epoch_%SELECTED_EPOCH%\metrics.csv" (
    echo ERROR: selected checkpoint has no completed val1000 evaluation.
    exit /b 5
)
if not exist "%SELECTED_CHECKPOINT%" (
    echo ERROR: selected checkpoint not found: %SELECTED_CHECKPOINT%
    exit /b 5
)
if exist "%TEST_ROOT%" (
    echo ERROR: formal test output already exists: %TEST_ROOT%
    echo The strict protocol permits only one frozen test1000 evaluation.
    exit /b 6
)

echo ============================================================
echo FORMAL QXS TEST1000
echo run tag:    %RUN_TAG%
echo variant:    %SELECTED_VARIANT%
echo epoch:      %SELECTED_EPOCH%
echo checkpoint: %SELECTED_CHECKPOINT%
echo output:     %TEST_ROOT%
echo ============================================================

"%PYTHON%" test_OS.py ^
  --variant %SELECTED_VARIANT% ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --split test ^
  --limit 1000 ^
  --checkpoint "%SELECTED_CHECKPOINT%" ^
  --output-dir "%TEST_ROOT%" ^
  --metric-profile paper ^
  --output-range-policy clamp ^
  --timing-warmup 10
if errorlevel 1 goto :failed

echo ============================================================
echo Formal test1000 completed exactly once.
echo Register the result in EXPERIMENT_REGISTRY.json, then run:
echo .venv\Scripts\python.exe maintain_experiment_index.py --check --render
echo ============================================================
exit /b 0

:test_usage
echo Usage: %~nx0 test run_tag variant epoch
echo Example: %~nx0 test qxs_direct_seed42_v1 r3 004
exit /b 2

:usage
echo Usage:
echo   %~nx0 train-val [run_tag]
echo   %~nx0 extend-val [run_tag]
echo   %~nx0 test run_tag variant epoch
exit /b 2

:failed
echo.
echo ERROR: QXS protocol stopped after a failed command.
exit /b 1
