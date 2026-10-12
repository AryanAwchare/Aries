# ============================================================
# Aries — Paper Trading Quickstart (PowerShell)
# Usage: .\scripts\quickstart.ps1 [-Mode <backtest|replay|server>]
# Default mode: replay (full paper-trading dry-run on synthetic data)
# ============================================================
param(
    [ValidateSet("backtest","replay","server","train","test")]
    [string]$Mode = "replay"
)

$ErrorActionPreference = "Stop"
$PY = ".\.venv\Scripts\python.exe"

if (-not (Test-Path $PY)) {
    Write-Host "⚠️  No .venv found. Running setup first..." -ForegroundColor Yellow
    .\scripts\setup.ps1
}

Write-Host ""
Write-Host "════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ARIES — Gold (XAUUSD) AI Trading Platform" -ForegroundColor Cyan
Write-Host "  Mode: $Mode" -ForegroundColor Cyan
Write-Host "════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

switch ($Mode) {
    "backtest" {
        Write-Host "▶ Running hybrid backtest on synthetic data..." -ForegroundColor Green
        & $PY -m app.cli backtest --synthetic --days 730
    }
    "replay" {
        Write-Host "▶ Setting up DB..." -ForegroundColor Green
        & $PY -m app.cli setup-db
        Write-Host "▶ Running paper trading replay on synthetic data..." -ForegroundColor Green
        & $PY -m app.cli replay --synthetic --days 180
    }
    "train" {
        Write-Host "▶ Running walk-forward training + calibration..." -ForegroundColor Green
        & $PY -m app.cli walk-forward --synthetic --days 1800 --calibrate
    }
    "test" {
        Write-Host "▶ Running full test suite..." -ForegroundColor Green
        & $PY -m pytest tests -q
    }
    "server" {
        Write-Host "▶ Setting up DB..." -ForegroundColor Green
        & $PY -m app.cli setup-db
        Write-Host "▶ Starting FastAPI server on http://localhost:8000 ..." -ForegroundColor Green
        Write-Host "   Docs: http://localhost:8000/docs" -ForegroundColor DarkGray
        Write-Host "   Press Ctrl+C to stop." -ForegroundColor DarkGray
        & $PY -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    }
}

Write-Host ""
Write-Host "✅ Done." -ForegroundColor Green
