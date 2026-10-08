@echo off
setlocal
cd /d "%~dp0"
set PYTHONPATH=%CD%;%PYTHONPATH%
set ROOT=..\datasets\QXSLAB_SAROPT
set SPLIT=%ROOT%\splits\cddfuse_8000_1000_1000_seed20260924.json
set ROOT_OUT=test_result\reliability_v5\d0_c0_first
set C0=models\reliability_v5\frequency_minimax_baseline_pilot\epoch_005.pth
set E1=models\reliability_v5\e1_dual_stem_pilot\epoch_005.pth
set E1_UNF=models\reliability_v5\e1_encoder_unfrozen_pilot\epoch_005.pth
set C0_UNF=models\reliability_v5\c0_encoder_unfrozen_control\epoch_005.pth

for %%P in ("%C0%" "%E1%" "%E1_UNF%" "%C0_UNF%" "%SPLIT%") do (
    if not exist "%%~P" (
        echo Missing fixed experimental input: %%~P
        exit /b 1
    )
)

echo [1/2] D0 diagnosis on fixed val24...
.venv\Scripts\python.exe -u audit_d0_c0_decomposition.py ^
  --data-root "%ROOT%" --split-manifest "%SPLIT%" ^
  --split val --limit 24 --noise-std 0.02 --seed 42 ^
  --model "C0=%C0%" "E1_stem=%E1%" "E1_unfrozen=%E1_UNF%" "C0_unfrozen=%C0_UNF%" ^
  --output-dir "%ROOT_OUT%\val24"
if errorlevel 1 exit /b 1

echo [2/2] D0 diagnosis on fixed val200...
.venv\Scripts\python.exe -u audit_d0_c0_decomposition.py ^
  --data-root "%ROOT%" --split-manifest "%SPLIT%" ^
  --split val --limit 200 --noise-std 0.02 --seed 42 ^
  --model "C0=%C0%" "E1_stem=%E1%" "E1_unfrozen=%E1_UNF%" "C0_unfrozen=%C0_UNF%" ^
  --output-dir "%ROOT_OUT%\val200"
if errorlevel 1 exit /b 1

echo D0 saved summary.csv per_image.csv channel_correlations.csv metadata.json
