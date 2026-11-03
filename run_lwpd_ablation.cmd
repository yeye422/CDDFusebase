@echo off
setlocal EnableExtensions

rem LWPD cumulative component ablation for QXS-SAROPT.
rem
rem Train E1-E3 and evaluate E1-E4 on the fixed val1000 split:
rem   run_lwpd_ablation.cmd train-val
rem
rem After choosing exactly one variant from validation, run formal test1000:
rem   run_lwpd_ablation.cmd test e1
rem   run_lwpd_ablation.cmd test e2
rem   run_lwpd_ablation.cmd test e3
rem   run_lwpd_ablation.cmd test e4

cd /d "%~dp0"

set "MODE=%~1"
set "PYTHON=.venv\Scripts\python.exe"
set "DATA_ROOT=..\datasets\QXSLAB_SAROPT"
set "SPLIT_MANIFEST=..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json"
set "INIT_CHECKPOINT=models\direct_fusion_qxs\qxs_direct_seed42_v1\r3\lowest_val_loss_diagnostic.pth"
set "MODEL_ROOT=models\lwpd_qxs_ablation_v1"
set "VAL_ROOT=test_result\lwpd_qxs_ablation_v1\val1000"
set "TEST_ROOT=test_result\lwpd_qxs_ablation_v1\test1000"
set "E4_CHECKPOINT=models\lwpd_qxs_full_v1\e4_seed42\epoch_016.pth"

if not exist "%PYTHON%" (
    echo ERROR: Python executable not found: %PYTHON%
    exit /b 3
)
if not exist "%DATA_ROOT%" (
    echo ERROR: Dataset root not found: %DATA_ROOT%
    exit /b 3
)
if not exist "%SPLIT_MANIFEST%" (
    echo ERROR: Split manifest not found: %SPLIT_MANIFEST%
    exit /b 3
)
if not exist "%INIT_CHECKPOINT%" (
    echo ERROR: R3 initialization checkpoint not found: %INIT_CHECKPOINT%
    exit /b 3
)

if /I "%MODE%"=="train-val" goto :train_val
if /I "%MODE%"=="test" goto :formal_test
goto :usage

:train_val
echo ============================================================
echo LWPD cumulative ablation: E1-E4
echo Train: fixed train8000, seed 42, 16 epochs
echo Eval:  fixed val1000, paper metrics, clamp policy
echo Test:  disabled in this stage
echo ============================================================

for %%V in (e1 e2 e3) do (
    call :train_variant %%V
    if errorlevel 1 goto :failed
)

for %%V in (e1 e2 e3 e4) do (
    call :validate_variant %%V
    if errorlevel 1 goto :failed
)

echo ============================================================
echo LWPD E1-E4 training and val1000 evaluation completed.
echo.
echo Compare these files in cumulative order:
echo   E1 - R3 : fixed Haar decomposition
echo   E2 - E1 : directional context gates
echo   E3 - E2 : controlled SAR recovery
echo   E4 - E3 : phase-symmetry evidence
echo.
echo Validation outputs:
echo   %VAL_ROOT%\e1_seed42\metrics.csv
echo   %VAL_ROOT%\e2_seed42\metrics.csv
echo   %VAL_ROOT%\e3_seed42\metrics.csv
echo   %VAL_ROOT%\e4_seed42\metrics.csv
echo.
echo Do not evaluate every variant on test1000. Select one using val1000,
echo then run: %~nx0 test e1_or_e2_or_e3_or_e4
echo ============================================================
exit /b 0

:train_variant
set "SHORT=%~1"
set "VARIANT=lwpd_%SHORT%"
set "RUN_NAME=%SHORT%_seed42"
set "RUN_DIR=%MODEL_ROOT%\%RUN_NAME%"
set "FINAL_CHECKPOINT=%RUN_DIR%\epoch_016.pth"

if exist "%FINAL_CHECKPOINT%" (
    echo.
    echo [SKIP TRAIN] %VARIANT% already has %FINAL_CHECKPOINT%
    exit /b 0
)

set "LATEST_CHECKPOINT="
if exist "%RUN_DIR%" (
    for /f "delims=" %%C in ('dir /b /a-d /o:n "%RUN_DIR%\epoch_*.pth" 2^>nul') do set "LATEST_CHECKPOINT=%RUN_DIR%\%%C"
)

set "RESUME_OPTION="
if defined LATEST_CHECKPOINT (
    echo.
    echo [RESUME TRAIN] %VARIANT% from %LATEST_CHECKPOINT%
    set RESUME_OPTION=--resume "%LATEST_CHECKPOINT%"
) else (
    if exist "%RUN_DIR%" (
        dir /b "%RUN_DIR%" 2^>nul | findstr . >nul
        if not errorlevel 1 (
            echo ERROR: Non-empty run directory has no resumable epoch checkpoint:
            echo   %RUN_DIR%
            exit /b 5
        )
    )
    echo.
    echo [TRAIN] %VARIANT%
)

"%PYTHON%" -u train_lwpd.py ^
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
  --warmup-epochs 2 ^
  --local-epochs 4 ^
  --joint-epochs 10 ^
  --warmup-lwpd-lr 1e-4 ^
  --warmup-fusion-lr 5e-5 ^
  --local-lwpd-lr 1e-4 ^
  --local-fusion-lr 1e-4 ^
  --joint-lwpd-lr 5e-5 ^
  --joint-fusion-lr 5e-5 ^
  --decoder-lr 2e-5 ^
  --encoder-tail-lr 1e-5 ^
  --encoder-early-lr 2e-6 ^
  --weight-decay 1e-4 ^
  --clip-grad 1 ^
  --gradient-weight 10 ^
  --reconstruction-weight 0.1 ^
  --init-checkpoint "%INIT_CHECKPOINT%" ^
  --output-dir "%MODEL_ROOT%" ^
  --run-name "%RUN_NAME%" ^
  --device cuda ^
  %RESUME_OPTION%
exit /b %errorlevel%

:validate_variant
set "SHORT=%~1"
set "RUN_NAME=%SHORT%_seed42"
if /I "%SHORT%"=="e4" (
    set "CHECKPOINT=%E4_CHECKPOINT%"
) else (
    set "CHECKPOINT=%MODEL_ROOT%\%RUN_NAME%\epoch_016.pth"
)
set "OUTPUT_DIR=%VAL_ROOT%\%RUN_NAME%"

if not exist "%CHECKPOINT%" (
    echo ERROR: Validation checkpoint not found: %CHECKPOINT%
    exit /b 6
)
if exist "%OUTPUT_DIR%\metrics.csv" (
    echo.
    echo [SKIP VAL] %SHORT% already has %OUTPUT_DIR%\metrics.csv
    exit /b 0
)

echo.
echo [VAL1000] %SHORT% using %CHECKPOINT%
"%PYTHON%" test_OS.py ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --split val ^
  --variant auto ^
  --checkpoint "%CHECKPOINT%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --offset 0 ^
  --limit 1000 ^
  --metrics-only ^
  --metric-profile paper ^
  --output-range-policy clamp ^
  --timing-warmup 10
exit /b %errorlevel%

:formal_test
set "SHORT=%~2"
if "%SHORT%"=="" goto :usage
if /I not "%SHORT%"=="e1" if /I not "%SHORT%"=="e2" if /I not "%SHORT%"=="e3" if /I not "%SHORT%"=="e4" (
    echo ERROR: Test variant must be e1, e2, e3, or e4.
    exit /b 2
)

set "RUN_NAME=%SHORT%_seed42"
if /I "%SHORT%"=="e4" (
    set "CHECKPOINT=%E4_CHECKPOINT%"
) else (
    set "CHECKPOINT=%MODEL_ROOT%\%RUN_NAME%\epoch_016.pth"
)
set "OUTPUT_DIR=%TEST_ROOT%\%RUN_NAME%"

if not exist "%CHECKPOINT%" (
    echo ERROR: Formal-test checkpoint not found: %CHECKPOINT%
    exit /b 6
)
if exist "%OUTPUT_DIR%\metrics.csv" (
    echo ERROR: Formal test output already exists: %OUTPUT_DIR%\metrics.csv
    echo Refusing to repeat held-out test evaluation.
    exit /b 7
)

echo ============================================================
echo Formal LWPD test1000
echo Variant:    %SHORT%
echo Checkpoint: %CHECKPOINT%
echo Output:     %OUTPUT_DIR%
echo ============================================================

"%PYTHON%" test_OS.py ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --split test ^
  --variant auto ^
  --checkpoint "%CHECKPOINT%" ^
  --output-dir "%OUTPUT_DIR%" ^
  --offset 0 ^
  --limit 1000 ^
  --metric-profile paper ^
  --output-range-policy clamp ^
  --timing-warmup 10
if errorlevel 1 goto :failed

echo ============================================================
echo Formal test1000 completed: %OUTPUT_DIR%\metrics.csv
echo ============================================================
exit /b 0

:usage
echo Usage:
echo   %~nx0 train-val
echo   %~nx0 test e1
echo   %~nx0 test e2
echo   %~nx0 test e3
echo   %~nx0 test e4
echo.
echo train-val trains E1-E3 and evaluates E1-E4 on val1000.
echo test evaluates exactly one validation-selected variant on test1000.
exit /b 2

:failed
echo.
echo ERROR: LWPD ablation command failed with exit code %errorlevel%.
exit /b %errorlevel%

