@echo off
setlocal

cd /d "%~dp0"

for %%E in (001 002 003 004 005 006 007 008 009 010) do (
  for %%G in (0.00 0.10 0.15 0.20 0.25 0.35 0.50) do (
    echo Evaluating epoch %%E at residual scale %%G
    .venv\Scripts\python.exe test_OS.py ^
      --data-root ..\datasets\QXSLAB_SAROPT ^
      --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
      --split val ^
      --limit 200 ^
      --checkpoint models\reliability_v4\A1prime_pilot\epoch_%%E.pth ^
      --variant residual_only ^
      --force-residual-scale %%G ^
      --metric-profile paper ^
      --output-range-policy clamp ^
      --timing-warmup 0 ^
      --metrics-only ^
      --output-dir test_result\reliability_v4\sweep\A1prime_e%%E_g%%G
    if errorlevel 1 exit /b 1
  )
)

echo Sweep completed successfully.
