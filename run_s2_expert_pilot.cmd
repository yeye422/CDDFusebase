@echo off
setlocal
cd /d "%~dp0" || exit /b 1

if /i "%~1"=="--help" goto usage
if not "%~2"=="" goto invalid_args
if not "%~1"=="" if /i not "%~1"=="--with-routes" goto invalid_args

set "PYTHON=.venv\Scripts\python.exe"
set "MODELS=models\s2_expert_pilot_v2"
set "RESULTS=test_result\s2_expert_pilot_v2"
set "REPORT=%RESULTS%\comparison_seed42_val64.csv"
if not exist "%PYTHON%" (
    echo Python environment missing: %PYTHON% 1>&2
    exit /b 1
)

for %%M in (e2_ft conv dense top2) do call :train %%M || exit /b 1
for %%M in (e2_ft conv dense top2) do call :evaluate %%M || exit /b 1
if not exist "%REPORT%" "%PYTHON%" -u run_s2_expert_pilot.py compare || exit /b 1
echo Comparison: %REPORT%
if /i "%~1"=="--with-routes" (
    for %%M in (dense top2) do call :routes %%M || exit /b 1
)
exit /b 0

:train
if exist "%MODELS%\%~1_seed42\final.pth" (
    echo Reusing checkpoint: %~1
    exit /b 0
)
"%PYTHON%" -u run_s2_expert_pilot.py train --mode %~1
exit /b %errorlevel%

:evaluate
set "EVAL=%RESULTS%\%~1_seed42_val64"
if exist "%EVAL%\metrics.csv" if exist "%EVAL%\run_metadata.json" (
    echo Reusing evaluation: %~1
    exit /b 0
)
"%PYTHON%" -u run_s2_expert_pilot.py eval --mode %~1
exit /b %errorlevel%

:routes
set "ROUTES=%RESULTS%\%~1_seed42_routes_val64"
if exist "%ROUTES%\routes.csv" if exist "%ROUTES%\summary.json" (
    echo Reusing route diagnostics: %~1
    exit /b 0
)
"%PYTHON%" -u diagnose_s2_expert_routes.py --mode %~1 --limit 64
exit /b %errorlevel%

:invalid_args
echo Usage: %~nx0 [--with-routes] 1>&2
exit /b 2

:usage
echo Usage: %~nx0 [--with-routes]
echo Runs the four S2 pilot modes, evaluates val64 and writes the comparison.
echo --with-routes also runs Dense and Top-2 route diagnostics.
exit /b 0
