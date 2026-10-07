@echo off
setlocal

cd /d "%~dp0"

set DATA_ROOT=..datasetsQXSLAB_SAROPT
set SPLIT_MANIFEST=..datasetsQXSLAB_SAROPTsplitscddfuse_8000_1000_1000_seed20260924.json
set C0_CHECKPOINT=modelseliability_v5requency_minimax_baseline_pilotepoch_005.pth
set R1_CHECKPOINT=modelseliability_v51_spatial_modulation_pilotepoch_005.pth
set OUTPUT_ROOT=test_resulteliability_v5alidation_r1_final

if not exist "%C0_CHECKPOINT%" (
  echo Missing C0 checkpoint: %C0_CHECKPOINT%
  exit /b 1
)
if not exist "%R1_CHECKPOINT%" (
  echo Missing R1 checkpoint: %R1_CHECKPOINT%
  exit /b 1
)

echo [1/3] Evaluating C0 fixed endpoint on val200...
.venvScriptspython.exe test_OS.py ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --split val ^
  --limit 200 ^
  --checkpoint "%C0_CHECKPOINT%" ^
  --variant frequency_minimax_baseline ^
  --metric-profile paper ^
  --output-range-policy clamp ^
  --timing-warmup 0 ^
  --metrics-only ^
  --output-dir "%OUTPUT_ROOT%requency_minimax_baseline"

if errorlevel 1 exit /b 1

echo [2/3] Evaluating R1 fixed endpoint on the same val200...
.venvScriptspython.exe test_OS.py ^
  --data-root "%DATA_ROOT%" ^
  --split-manifest "%SPLIT_MANIFEST%" ^
  --split val ^
  --limit 200 ^
  --checkpoint "%R1_CHECKPOINT%" ^
  --variant r1_spatial_modulation ^
  --metric-profile paper ^
  --output-range-policy clamp ^
  --timing-warmup 0 ^
  --metrics-only ^
  --output-dir "%OUTPUT_ROOT%1_spatial_modulation"

if errorlevel 1 exit /b 1

echo [3/3] Summarizing C0 versus R1...
.venvScriptspython.exe summarize_frequency_pilots.py ^
  --root "%OUTPUT_ROOT%" ^
  --output "%OUTPUT_ROOT%summary.csv"

if errorlevel 1 exit /b 1
echo R1 fixed-endpoint validation completed successfully.
