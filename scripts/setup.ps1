# One-time setup for the Illegal Mining Detector. Safe to run more than once.
#
# What it does, in order:
#   1. Creates the .venv folder with Python 3.11 if it is missing.
#   2. Installs the packages in requirements.txt (or the app packages, if that file is missing).
#   3. Downloads and prepares the global mining dataset (skipped if already there).
#   4. Runs the test suite.
#
# Run from PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\setup.ps1

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectDir
$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"

function Step($text) { Write-Host ""; Write-Host "== $text" -ForegroundColor Cyan }

Step "1/4 Python environment"
if (Test-Path $VenvPython) {
    Write-Host "Found .venv, reusing it."
} else {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Python 3.11 was not found. Install it from https://www.python.org/downloads/ (tick 'Add to PATH'), then run this again." -ForegroundColor Red
        exit 1
    }
    Write-Host "Created .venv."
}

Step "2/4 Installing packages"
& $VenvPython -m pip install --quiet --upgrade pip
if (Test-Path "requirements.txt") {
    & $VenvPython -m pip install --quiet -r requirements.txt
} else {
    Write-Host "requirements.txt not found, installing the app packages directly."
    & $VenvPython -m pip install --quiet streamlit folium streamlit-folium pandas numpy pillow matplotlib geopandas pyogrio shapely requests pytest
}
if ($LASTEXITCODE -ne 0) {
    Write-Host "Package installation failed. Check your internet connection and run this again." -ForegroundColor Red
    exit 1
}

Step "3/4 Mining dataset"
& $VenvPython "scripts\fetch_datasets.py" mining
if ($LASTEXITCODE -ne 0) {
    Write-Host "The mining dataset could not be prepared. The app still runs with the sample data." -ForegroundColor Yellow
}

Step "4/4 Tests"
& $VenvPython -m pytest -q
$testsOk = ($LASTEXITCODE -eq 0)
if (-not $testsOk) {
    Write-Host "Some tests failed. Read the output above." -ForegroundColor Yellow
}

Write-Host ""
if ($testsOk) {
    Write-Host "Setup complete - next: scripts\run_app.bat" -ForegroundColor Green
} else {
    Write-Host "Setup finished with failing tests. Fix them before relying on the results." -ForegroundColor Yellow
    exit 1
}
