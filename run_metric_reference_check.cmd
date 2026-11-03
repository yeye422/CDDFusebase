@echo off
setlocal
cd /d "%~dp0"
rem Synthetic unit check only; never loads a dataset or model checkpoint.
.venv\Scripts\python.exe -m unittest tests.test_reference_metrics tests.test_paper_metrics
if errorlevel 1 exit /b 1
"D:\MATLAB\R2024a\bin\matlab.exe" -batch "addpath('tests/metric_reference'); check_matlab_oracle('reports/metric_audit_20261101_v2');"
if errorlevel 1 exit /b 1
echo All Python and MATLAB synthetic reference checks passed.
exit /b 0
