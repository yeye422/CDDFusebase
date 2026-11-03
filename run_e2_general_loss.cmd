@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -u run_e2_general_loss.py %*
exit /b %errorlevel%
