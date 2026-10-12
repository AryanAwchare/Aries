"""
Download XAUUSD M15 OHLCV data for Aries training.

Usage:
    python scripts/download_data.py                    # yfinance (default)
    python scripts/download_data.py --source yfinance  # explicit
    python scripts/download_data.py --source mt5       # MT5 Python API
    python scripts/download_data.py --start 2019-01-01 --end 2025-12-31

Output: data/raw/XAUUSD_M15.csv  (UTC-indexed, ready for Aries CSVProvider)
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data" / "raw"
OUT_FILE = OUT_DIR / "XAUUSD_M15.csv"


def download_yfinance(start: str, end: str) -> None:
    """Download via yfinance (free, ~1-2 years of M15 available from Yahoo)."""
    try:
        import yfinance as yf
    except ImportError:
        print("Installing yfinance...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
        import yfinance as yf

    import pandas as pd

    print(f"Downloading XAUUSD M15 from Yahoo Finance ({start} -> {end})...")
    ticker = yf.Ticker("GC=F")  # Gold futures — closest proxy to XAUUSD

    # Yahoo only provides ~60 days of 15-minute data at once; loop year-by-year
    # and use daily for earlier data, resampling to M15 is not ideal.
    # For longer history, we download daily then resample (lower quality but ok for
    # initial training).

    # Try M15 first (limited to ~60 days from Yahoo):
    df_15m = ticker.history(period="60d", interval="15m", auto_adjust=True)
    df_15m.index = df_15m.index.tz_convert("UTC")
    df_15m = df_15m.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df_15m.index.name = "timestamp"

    # Get longer history as daily and resample to pseudo-M15 (fill-forward each day's OHLC to 96 bars/day)
    print("  Fetching 5-year daily data for longer history (will upsample to M15)...")
    df_d = ticker.history(start=start, end=end, interval="1d", auto_adjust=True)
    df_d.index = df_d.index.tz_localize("UTC") if df_d.index.tz is None else df_d.index.tz_convert("UTC")
    df_d = df_d.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df_d.index.name = "timestamp"

    # Upsample daily → 15-min by forward-filling OHLC within each day
    # This is approximate but gives the trainer enough structural variety
    df_up = df_d.resample("15min").ffill()
    # Add small noise so bars aren't identical (prevents overfitting to flat regions)
    import numpy as np
    rng = np.random.default_rng(42)
    for col in ["open", "high", "low", "close"]:
        df_up[col] = df_up[col] * (1 + rng.normal(0, 0.0003, len(df_up)))
    df_up["volume"] = (df_up["volume"] / 96).clip(lower=1).astype(int)

    # Combine: prefer real 15m bars where available, else use upsampled daily
    combined = pd.concat([df_up, df_15m])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined = combined.sort_index()

    # Ensure UTC-aware timestamps in the output
    if combined.index.tz is None:
        combined.index = combined.index.tz_localize("UTC")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_FILE)
    print(f"  [OK] Saved {len(combined):,} bars  ->  {OUT_FILE}")
    print(f"     Date range: {combined.index[0]}  ->  {combined.index[-1]}")
    print()
    print("NOTE: The first few years are upsampled from daily data (lower quality).")
    print("      The last ~60 days are real 15-minute bars from Yahoo Finance.")
    print("      For best training quality, use MT5 export or HistData.com.")


def download_mt5(start: str, end: str) -> None:
    """Download via MetaTrader 5 Python API (requires MT5 terminal installed)."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        print("ERROR: MetaTrader5 package not installed.")
        print("Install with:  .venv\\Scripts\\pip install MetaTrader5")
        sys.exit(1)

    import pandas as pd

    if not mt5.initialize():
        print(f"ERROR: MT5 initialize() failed: {mt5.last_error()}")
        print("Make sure MetaTrader 5 terminal is running and logged in.")
        sys.exit(1)

    print(f"Downloading XAUUSD M15 via MT5 ({start} → {end})...")

    from_date = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    to_date = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)

    rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_M15, from_date, to_date)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        print("ERROR: No data returned. Check that XAUUSD is available in your MT5 terminal.")
        sys.exit(1)

    df = pd.DataFrame(rates)
    df["timestamp"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("timestamp")
    df = df.rename(columns={"tick_volume": "volume"})[["open", "high", "low", "close", "volume"]]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE)
    print(f"  [OK] Saved {len(df):,} bars  ->  {OUT_FILE}")
    print(f"     Date range: {df.index[0]}  ->  {df.index[-1]}")


def verify_file() -> None:
    """Print a quick sanity check on the CSV."""
    import pandas as pd
    df = pd.read_csv(OUT_FILE, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    print()
    print("=" * 52)
    print("  File verification")
    print("=" * 52)
    print(f"  Rows:     {len(df):,}")
    print(f"  Start:    {df.index.min()}")
    print(f"  End:      {df.index.max()}")
    print(f"  Columns:  {list(df.columns)}")
    print(f"  NaN rows: {df.isna().any(axis=1).sum()}")
    print(f"  Price range: {df['close'].min():.2f} – {df['close'].max():.2f}")
    print("=" * 52)
    print()
    print("Next: train the meta-model on this data:")
    print("  .venv\\Scripts\\python.exe -m app.cli walk-forward --days 1825 --calibrate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download XAUUSD M15 data for Aries training")
    parser.add_argument("--source", choices=["yfinance", "mt5"], default="yfinance",
                        help="Data source (default: yfinance)")
    parser.add_argument("--start", default="2020-01-01", help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"), help="End date YYYY-MM-DD")
    args = parser.parse_args()

    if args.source == "yfinance":
        download_yfinance(args.start, args.end)
    elif args.source == "mt5":
        download_mt5(args.start, args.end)

    verify_file()


if __name__ == "__main__":
    main()
