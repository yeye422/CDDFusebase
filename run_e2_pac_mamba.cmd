@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -u run_e2_pac_mamba.py %*
exit /b %errorlevel%
