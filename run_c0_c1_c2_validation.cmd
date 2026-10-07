@echo off
setlocal

cd /d "%~dp0"

for %%V in (frequency_matched_baseline base_optical_detail base_optical_detail_refine) do (
  echo Evaluating fixed final checkpoint for %%V
  .venv\Scripts\python.exe test_OS.py ^
    --data-root ..\datasets\QXSLAB_SAROPT ^
    --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
    --split val ^
    --limit 200 ^
    --checkpoint models\reliability_v5\%%V_pilot\epoch_005.pth ^
    --variant %%V ^
    --metric-profile paper ^
    --output-range-policy clamp ^
    --timing-warmup 0 ^
    --metrics-only ^
    --output-dir test_result\reliability_v5\validation_final\%%V
  if errorlevel 1 exit /b 1
)

echo C0/C1/C2 fixed-endpoint validation completed successfully.
