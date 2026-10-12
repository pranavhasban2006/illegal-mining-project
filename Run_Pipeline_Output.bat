@echo off
echo Running Sentinel-2 Pipeline...
set PYTHONPATH=src
.\.venv\Scripts\python.exe -m mining_detector.pipeline
echo.
echo Pipeline Finished! Opening outputs folder...
explorer.exe "%~dp0outputs"
pause
