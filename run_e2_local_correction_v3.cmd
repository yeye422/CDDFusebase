@echo off
setlocal
cd /d "%~dp0"
if not defined E2_PYTHON if exist "D:\anaconda\envs\lightglue_env\python.exe" set "E2_PYTHON=D:\anaconda\envs\lightglue_env\python.exe"
if not defined E2_PYTHON set "E2_PYTHON=%~dp0.venv\Scripts\python.exe"
"%E2_PYTHON%" -u run_e2_local_correction_v3.py %*
exit /b %errorlevel%
