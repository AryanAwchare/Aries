# One-time setup for Windows (PowerShell).
# Creates a Python 3.12 venv (recommended — torch/Kronos/xgboost wheels are
# NOT yet available for Python 3.14 on Windows), installs deps, sets up the
# DB and runs the test suite.
#
# python-3.12: if not installed, edit $PY to the path of a 3.12 interpreter.
$ErrorActionPreference = "Stop"
$PY = "python3.12"

if (-not (Get-Command $PY -ErrorAction SilentlyContinue)) {
    $PY = "py -3.12"
}

& $PY -c "import sys; assert sys.version_info[:2] >= (3,10), 'Python 3.10+ required'; print(sys.version)"
if (-not $?) { exit 1 }

if (-not (Test-Path ".venv")) {
    & $PY -m venv .venv
    Write-Host "Created .venv"
}

& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
if (-not $?) { exit 1 }

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Copied .env.example -> .env (fill in credentials)"
}

& ".\.venv\Scripts\python.exe" -m app.cli setup-db
Write-Host "--------------------------------------------------"
Write-Host "Setup complete. Next:"
Write-Host "  .\.venv\Scripts\python.exe -m app.cli backtest --synthetic"
Write-Host "  .\.venv\Scripts\python.exe -m pytest tests -q"
Write-Host "  .\.venv\Scripts\python.exe -m uvicorn app.main:app --reload"