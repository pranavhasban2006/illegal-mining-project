@echo off
REM Double-click this file to start the Illegal Mining Detector app.
REM It runs scripts\run_app.ps1, which sets everything up and starts the app.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_app.ps1"
if errorlevel 1 (
    echo.
    echo The app did not start. Read the message above, then press any key to close this window.
    pause >nul
)
