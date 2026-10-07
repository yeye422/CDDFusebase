@echo off
setlocal

cd /d "%~dp0"

.venv\Scripts\python.exe test_OS.py ^
  --data-root ..\datasets\QXSLAB_SAROPT ^
  --split-manifest ..\datasets\QXSLAB_SAROPT\splits\cddfuse_8000_1000_1000_seed20260924.json ^
  --split val --limit 200 ^
  --checkpoint models\reliability_v5\srr_detail_v6_ranking_pilot\epoch_005.pth ^
  --variant srr_detail ^
  --metric-profile paper --output-range-policy clamp ^
  --timing-warmup 0 --metrics-only ^
  --output-dir test_result\reliability_v5\srr_detail_v6_ranking_validation

if errorlevel 1 exit /b 1
echo SRR DetailFuse V6 ranking validation completed successfully.
