"""Ingest HISTDATA XAUUSD M1 files and resample to M15 OHLCV.

Reads every ``HISTDATA_COM_MT_XAUUSD_M1<yymm>`` folder under a downloads
directory, concatenates the ``DAT_MT_XAUUSD_M1_<year>.csv`` files into one
UTC-indexed M1 frame, resamples to M15 and writes the CSVProvider-format
file ``data/raw/XAUUSD_M15.csv``.

Usage:
    python scripts/ingest_histdata.py                          # default Downloads
    python scripts/ingest_histdata.py --src C:/path/to/Downloads
    python scripts/ingest_histdata.py --out data/raw
"""
from __future__ import annotations

import argparse
import glob
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data" / "raw"
OUT_FILE = OUT_DIR / "XAUUSD_M15.csv"

COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]


def discover_csv_paths(src_dir: Path) -> list[Path]:
    """Return every HISTDATA ``DAT_MT_XAUUSD_M1_*.csv`` file underneath ``src_dir``."""
    matches = []
    for pattern in (
        "*/DAT_MT_XAUUSD_M1_*.csv",
        "DAT_MT_XAUUSD_M1_*.csv",
        "*/*/DAT_MT_XAUUSD_M1_*.csv",
    ):
        matches += glob.glob(str(src_dir / pattern))
    seen: set[str] = set()
    out: list[Path] = []
    for m in sorted(matches):
        if m not in seen:
            seen.add(m)
            out.append(Path(m))
    out.sort(key=lambda p: p.name)
    return out


def read_histdata(path: Path) -> pd.DataFrame:
    """Parse one HISTDATA M1 CSV into a UTC-tz-aware, close-typed frame."""
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        dtype={"open": np.float64, "high": np.float64, "low": np.float64, "close": np.float64},
    )
    ts = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M", errors="coerce")
    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"]).drop(columns=["date", "time"])
    df = df.set_index("timestamp").sort_index()
    df.index = df.index.tz_localize("UTC")
    return df


def resample_m15(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate M1 -> M15. Tick count per bucket is used as a volume proxy
    because HISTDATA MT files carry no real tick volume (all zeros)."""
    df["_n"] = 1
    agg = df.resample("15min").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "_n": "sum"}
    )
    agg = agg.dropna(subset=["open"]).rename(columns={"_n": "tick_count"})
    # The HISTDATA volume column is 0 throughout; substitute the tick proxy.
    agg["volume"] = agg["tick_count"].astype(int)
    agg = agg[["open", "high", "low", "close", "volume"]].round(3)
    return agg


def validate(df: pd.DataFrame) -> None:
    print(f"  bars:            {len(df):,}")
    print(f"  span:            {df.index.min()} -> {df.index.max()}")
    print(f"  years:           {sorted({ts.year for ts in df.index})}")
    print(f"  volume min/max:  {int(df['volume'].min())} / {int(df['volume'].max())} (proxy = ticks/15min)")
    per_year = df.groupby(df.index.year).size()
    for y, n in per_year.items():
        print(f"    {y}: {n:>9,}")
    gaps = df.index.to_series().diff().dropna()
    big = int((gaps > pd.Timedelta(minutes=45)).sum())
    print(f"  gaps >45min:     {big}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Resample HISTDATA XAUUSD M1 -> M15 (CSVProvider format).")
    ap.add_argument("--src", default="C:/Users/dell/Downloads", help="directory to glob for HISTDATA folders")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="output directory (default <repo>/data/raw)")
    args = ap.parse_args(argv)

    src_dir = Path(args.src)
    csv_paths = discover_csv_paths(src_dir)
    if not csv_paths:
        print(f"ERROR: no DAT_MT_XAUUSD_M1_*.csv found under {src_dir}")
        return 1
    print(f"Found {len(csv_paths)} HISTDATA file(s):")
    for p in csv_paths:
        print(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)")

    frames = [read_histdata(p) for p in csv_paths]
    mega = pd.concat(frames, axis=0)
    mega = mega[~mega.index.duplicated(keep="last")].sort_index()
    print(f"M1 rows before resample: {len(mega):,}")

    out = resample_m15(mega)
    args.out.mkdir(parents=True, exist_ok=True)
    out_df = out.reset_index()
    out_df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    out_df.to_csv(args.out / "XAUUSD_M15.csv", index=False)
    print(f"Wrote {args.out / 'XAUUSD_M15.csv'}")

    validate(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())