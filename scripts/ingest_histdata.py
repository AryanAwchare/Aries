"""Ingest HISTDATA XAUUSD M1 files and emit a CSVProvider-format OHLCV file.

Reads every ``HISTDATA_COM_MT_XAUUSD_M1<yymm>`` folder under a downloads
directory, concatenates the ``DAT_MT_XAUUSD_M1_<year>.csv`` files into one
UTC-indexed M1 frame, optionally resamples to a coarser timeframe and writes
``data/raw/XAUUSD_<TIMEFRAME>.csv``.

Usage:
    python scripts/ingest_histdata.py                          # default Downloads -> M15
    python scripts/ingest_histdata.py --timeframe M1           # keep raw M1 bars
    python scripts/ingest_histdata.py --src C:/path/to/Downloads
    python scripts/ingest_histdata.py --out data/raw

NOTE on volume: HISTDATA MT files carry no real tick volume (the column is 0
throughout). For the coarse timeframes we substitute the number of M1 bars per
bucket (a tick-activity proxy). At M1 each bucket holds exactly one bar, so the
proxy degenerates to a constant ``1`` and the volume features become neutral.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "data" / "raw"

COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]

# Resample alias per supported timeframe. M1 is a no-op passthrough.
RESAMPLE_RULES: dict[str, str | None] = {
    "M1": None,
    "M5": "5min",
    "M15": "15min",
    "M30": "30min",
    "H1": "h",
}


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


def resample(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Aggregate raw M1 into ``timeframe``.

    Tick count per bucket is used as a volume proxy because HISTDATA MT files
    carry no real tick volume (all zeros). At M1 no aggregation happens and the
    proxy is the constant ``1``.
    """
    rule = RESAMPLE_RULES.get(timeframe.upper())
    if rule is None and timeframe.upper() != "M1":
        raise ValueError(f"unsupported timeframe: {timeframe!r}; known {sorted(RESAMPLE_RULES)}")
    if rule is None:  # M1 passthrough
        out = df.copy()
        out["volume"] = 1
        out = out[["open", "high", "low", "close", "volume"]].round(3)
        return out
    df = df.copy()
    df["_n"] = 1
    agg = df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "_n": "sum"}
    )
    agg = agg.dropna(subset=["open"]).rename(columns={"_n": "tick_count"})
    agg["volume"] = agg["tick_count"].astype(int)
    agg = agg[["open", "high", "low", "close", "volume"]].round(3)
    return agg


def validate(df: pd.DataFrame, timeframe: str) -> None:
    gap_limit = pd.Timedelta(minutes=5) if timeframe.upper() == "M1" else pd.Timedelta(minutes=45)
    print(f"  bars:            {len(df):,}")
    print(f"  span:            {df.index.min()} -> {df.index.max()}")
    print(f"  years:           {sorted({ts.year for ts in df.index})}")
    print(f"  volume min/max:  {int(df['volume'].min())} / {int(df['volume'].max())}")
    per_year = df.groupby(df.index.year).size()
    for y, n in per_year.items():
        print(f"    {y}: {n:>10,}")
    gaps = df.index.to_series().diff().dropna()
    big = int((gaps > gap_limit).sum())
    print(f"  gaps >{gap_limit}: {big}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Ingest HISTDATA XAUUSD M1 -> OHLCV CSV (CSVProvider format).")
    ap.add_argument("--src", default="C:/Users/dell/Downloads", help="directory to glob for HISTDATA folders")
    ap.add_argument("--out", type=Path, default=OUT_DIR, help="output directory (default <repo>/data/raw)")
    ap.add_argument("--timeframe", default="M15", help="target timeframe (M1, M5, M15, M30, H1)")
    args = ap.parse_args(argv)

    timeframe = args.timeframe.upper()
    out_file = args.out / f"XAUUSD_{timeframe}.csv"

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

    out = resample(mega, timeframe)
    args.out.mkdir(parents=True, exist_ok=True)
    out_df = out.reset_index()
    out_df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    out_df.to_csv(out_file, index=False)
    print(f"Wrote {out_file}")

    validate(out, timeframe)
    return 0


if __name__ == "__main__":
    sys.exit(main())
