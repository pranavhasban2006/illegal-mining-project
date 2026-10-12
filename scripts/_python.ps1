# Shared helper for setup.ps1 and run_app.ps1. Dot-source it, then call New-ProjectVenv.
#
# Makes the .venv folder with Python 3.11. Tries, in order:
#   1. The Python launcher (py -3.11), if Python 3.11 is installed.
#   2. uv, which downloads Python 3.11 by itself if it is missing. uv is installed with pip
#      from whatever Python 3 the launcher finds, so no manual install is needed.

function New-ProjectVenv {
    param([string]$ProjectDir)

    # The callers use "Stop", but the launcher and uv print messages to stderr, which would
    # abort the script before the fallback runs. Use "Continue" here and check exit codes.
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        return (Invoke-VenvSetup -ProjectDir $ProjectDir)
    } finally {
        $ErrorActionPreference = $previousPreference
    }
}

function Invoke-VenvSetup {
    param([string]$ProjectDir)

    $venvDir = Join-Path $ProjectDir ".venv"

    py -3.11 -m venv $venvDir *> $null
    if ($LASTEXITCODE -eq 0 -and (Test-Path (Join-Path $venvDir "Scripts\python.exe"))) {
        return $true
    }

    Write-Host "Python 3.11 was not found by the launcher. Using uv to get it (one-time download, about 30 MB)."
    if (Test-Path $venvDir) { Remove-Item -Recurse -Force $venvDir }
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv --seed --python 3.11 $venvDir
    } else {
        py -3 -m pip install --quiet --user uv
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Could not install uv. Check your internet connection, or install Python 3.11 from https://www.python.org/downloads/ and run this again." -ForegroundColor Red
            return $false
        }
        py -3 -m uv venv --seed --python 3.11 $venvDir
    }
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path (Join-Path $venvDir "Scripts\python.exe"))) {
        Write-Host "Could not create the Python 3.11 environment. Install Python 3.11 from https://www.python.org/downloads/ and run this again." -ForegroundColor Red
        return $false
    }
    return $true
}
