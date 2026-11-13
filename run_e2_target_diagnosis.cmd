@echo off
setlocal
cd /d "%~dp0"
rem Default reuses saved images. Optional --raw-baseline permits one FP32 scene only.
rem No training or test split.
.venv\Scripts\python.exe diagnose_e2_targets.py %*
exit /b %errorlevel%
