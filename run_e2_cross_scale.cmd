@echo off
setlocal
cd /d "%~dp0" || exit /b 1
rem Prefer the CUDA environment verified for this workstation. Explicit overrides win.
if not defined E2_PYTHON if exist "D:\anaconda\envs\lightglue_env\python.exe" set "E2_PYTHON=D:\anaconda\envs\lightglue_env\python.exe"
if not defined E2_PYTHON set "E2_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%E2_PYTHON%" (
  echo Python executable not found: "%E2_PYTHON%"
  exit /b 1
)
echo [E2] Python: "%E2_PYTHON%"
"%E2_PYTHON%" -u run_e2_cross_scale.py %*
exit /b %errorlevel%
