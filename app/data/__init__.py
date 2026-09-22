"""Market data layer: OHLCV providers + CFTC COT ingestion.

The execution bridge (MetaApi) doubling as a market-data source keeps the
number of external dependencies low; a CSV provider lets the backtest and
replay harness feed historical bars through the exact same code path.
"""
from __future__ import annotations