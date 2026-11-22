@echo off
setlocal
pushd "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Missing .venv\Scripts\python.exe
    popd
    exit /b 1
)
if "%~1"=="" (
    ".venv\Scripts\python.exe" -u diagnose_rct_predictive_transfer.py all
) else (
    ".venv\Scripts\python.exe" -u diagnose_rct_predictive_transfer.py %*
)
set "RCT_EXIT_CODE=%ERRORLEVEL%"
popd
exit /b %RCT_EXIT_CODE%
