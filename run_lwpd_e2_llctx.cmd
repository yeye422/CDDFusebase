@echo off
setlocal
cd /d "%~dp0"
.venv\Scripts\python.exe -u run_lwpd_e2_llctx.py %*
exit /b %errorlevel%
