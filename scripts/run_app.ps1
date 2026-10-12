# Start the Illegal Mining Detector web app.
#
# What it does, in order:
#   1. Goes to the project folder (the folder above this script).
#   2. Creates the .venv folder with Python 3.11 if it is missing.
#   3. Installs any missing app packages.
#   4. Starts the app and opens it in your browser.
#
# Run it with scripts\run_app.bat (double-click), or from PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\run_app.ps1

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectDir

$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "Creating the Python environment (.venv) with Python 3.11..."
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Could not find Python 3.11. Install it from https://www.python.org/downloads/ and run this again."
        exit 1
    }
}

# Install the packages the app needs. pip skips anything already installed.
$Packages = @("streamlit", "folium", "streamlit-folium", "pandas", "numpy", "pillow", "matplotlib")
$Missing = @()
foreach ($pkg in $Packages) {
    & $VenvPython -m pip show $pkg *> $null
    if ($LASTEXITCODE -ne 0) { $Missing += $pkg }
}
if ($Missing.Count -gt 0) {
    Write-Host "Installing missing packages: $($Missing -join ', ')"
    & $VenvPython -m pip install --quiet @Missing
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Package installation failed. Check your internet connection and try again."
        exit 1
    }
}

# Regenerate the sample run if it is missing, so the demo always works.
$SampleReport = Join-Path $ProjectDir "tests\experience\fixtures\sample_run\report.json"
if (-not (Test-Path $SampleReport)) {
    & $VenvPython "tests\experience\fixtures\make_fixtures.py"
}

Write-Host ""
Write-Host "Starting the app. Your browser should open. To stop it, close this window or press Ctrl+C."
Write-Host ""
& $VenvPython -m streamlit run "app\Home.py" --server.headless=false --browser.gatherUsageStats=false
