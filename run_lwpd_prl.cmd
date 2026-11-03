@echo off
setlocal EnableExtensions

rem PR-LWPD validation protocol.
rem
rem Train the matched E2-PRL candidate and evaluate fixed val1000:
rem   run_lwpd_prl.cmd train-val
rem
rem Run formal test1000 only after validation and multi-seed review:
rem   run_lwpd_prl.cmd test

cd /d "%~dp0"

set "MODE=%~1"
set "PYTHON=.venv\Scripts\python.exe"
set "DATA_ROOT=..\datasets\QXSLAB_SAROPT"
set "SPLIT_MANIFEST=..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json"
set "INIT_CHECKPOINT=models\direct_fusion_qxs\qxs_direct_seed42_v1\r3\lowest_val_loss_diagnostic.pth"
set "MODEL_ROOT=models\lwpd_qxs_prl_v1"
set "RUN_NAME=e2_prl_seed42"
set "RUN_DIR=%MODEL_ROOT%\%RUN_NAME%"
set "FINAL_CHECKPOINT=%RUN_DIR%\epoch_016.pth"
set "VAL_DIR=test_result\lwpd_qxs_prl_v1\e2_prl_seed42_val1000"
set "TEST_DIR=test_result\lwpd_qxs_prl_v1\e2_prl_seed42_test1000"
set "E2_METRICS=test_result\lwpd_qxs_ablation_v1\val1000\e2_seed42\metrics.csv"

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
echo E2-PRL matched training and val1000 evaluation
echo Variant: lwpd_e2_prl
echo Train:   fixed train8000, seed 42, 16 epochs
echo Eval:    fixed val1000, paper metrics, clamp policy
echo Test:    disabled
echo ============================================================

if exist "%FINAL_CHECKPOINT%" (
    echo [SKIP TRAIN] Final checkpoint already exists: %FINAL_CHECKPOINT%
    goto :validate
)

set "LATEST_CHECKPOINT="
if exist "%RUN_DIR%" (
    for /f "delims=" %%C in ('dir /b /a-d /o:n "%RUN_DIR%\epoch_*.pth" 2^>nul') do set "LATEST_CHECKPOINT=%RUN_DIR%\%%C"
)

set "RESUME_OPTION="
if defined LATEST_CHECKPOINT (
    echo [RESUME] %LATEST_CHECKPOINT%
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
)

"%PYTHON%" -u train_lwpd.py ^
  --variant lwpd_e2_prl ^
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
  --local-lwpd-lr 1e-4 ^
  --joint-lwpd-lr 5e-5 ^
  --warmup-wavelet-lr 1e-5 ^
  --local-wavelet-lr 1e-5 ^
  --joint-wavelet-lr 5e-6 ^
  --warmup-fusion-lr 5e-5 ^
  --local-fusion-lr 1e-4 ^
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
if errorlevel 1 goto :failed

:validate
if not exist "%FINAL_CHECKPOINT%" (
    echo ERROR: Final checkpoint not found: %FINAL_CHECKPOINT%
    exit /b 6
)

if exist "%VAL_DIR%\metrics.csv" (
    echo [SKIP VAL] Existing metrics: %VAL_DIR%\metrics.csv
) else (
    "%PYTHON%" test_OS.py ^
      --data-root "%DATA_ROOT%" ^
      --split-manifest "%SPLIT_MANIFEST%" ^
      --split val ^
      --variant auto ^
      --checkpoint "%FINAL_CHECKPOINT%" ^
      --output-dir "%VAL_DIR%" ^
      --offset 0 ^
      --limit 1000 ^
      --metrics-only ^
      --metric-profile paper ^
      --output-range-policy clamp ^
      --timing-warmup 10
    if errorlevel 1 goto :failed
)

if not exist "%E2_METRICS%" (
    echo ERROR: Matched E2 baseline metrics not found: %E2_METRICS%
    exit /b 6
)

"%PYTHON%" compare_lwpd_prl.py ^
  --baseline "%E2_METRICS%" ^
  --candidate "%VAL_DIR%\metrics.csv" ^
  --output-dir "%VAL_DIR%\comparison_vs_e2"
if errorlevel 1 goto :failed

echo ============================================================
echo E2-PRL val1000 evaluation completed.
echo Metrics: %VAL_DIR%\metrics.csv
echo Summary: %VAL_DIR%\comparison_vs_e2\SUMMARY.md
echo No test image was evaluated.
echo ============================================================
exit /b 0

:formal_test
if not exist "%FINAL_CHECKPOINT%" (
    echo ERROR: Formal-test checkpoint not found: %FINAL_CHECKPOINT%
    exit /b 6
)
if not exist "%VAL_DIR%\comparison_vs_e2\SUMMARY.md" (
    echo ERROR: Validation comparison is missing.
    echo Run first: %~nx0 train-val
    exit /b 6
)
if exist "%TEST_DIR%\metrics.csv" (
    echo ERROR: Formal test output already exists: %TEST_DIR%\metrics.csv
    echo Refusing to repeat held-out test evaluation.
    exit /b 7
)

echo WARNING: Continue only if validation selected E2-PRL before test1000.
"%PYTHON%" test_OS.py ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --split test ^
  --variant auto ^
  --checkpoint "%FINAL_CHECKPOINT%" ^
  --output-dir "%TEST_DIR%" ^
  --offset 0 ^
  --limit 1000 ^
  --metric-profile paper ^
  --output-range-policy clamp ^
  --timing-warmup 10
if errorlevel 1 goto :failed

echo Formal test1000 completed: %TEST_DIR%\metrics.csv
echo Update EXPERIMENT_REGISTRY.json before using this result.
exit /b 0

:usage
echo Usage:
echo   %~nx0 train-val
echo   %~nx0 test
exit /b 2

:failed
set "FAIL_CODE=%errorlevel%"
echo ERROR: E2-PRL command failed with exit code %FAIL_CODE%.
exit /b %FAIL_CODE%

