@echo off
setlocal
cd /d "%~dp0" || exit /b 1
.venv\Scripts\python.exe -u run_e2_local_correspondence.py %*
exit /b %errorlevel%
