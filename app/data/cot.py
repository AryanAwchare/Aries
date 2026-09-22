"""CFTC COT (Commitments of Traders) ingestion.

Weekly aggregated institutional positioning for gold futures — a context
feature for the meta-model (not a standalone signal). Pulls the public
disaggregated-COT text archive and normalises the gold rows.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from zipfile import ZipFile

import pandas as pd

from ..logging import get_logger

log = get_logger(__name__)

COT_ARCHIVE_URL = "https://www.cftc.gov/files/dea/history/fut_disagg_txt_all.zip"

# Columns in the disaggregated futures-only report (selected set)
_COT_COLUMNS = [
    "market_and_exchange_names",
    "report_date_as_yyyy_mm_dd",
    "producer_merchant_processor_user_positions_long_all",
    "producer_merchant_processor_user_positions_short_all",
    "money_manager_positions_long_all",
    "money_manager_positions_short_all",
    "swap_dealer_positions_long_all",
    "swap_dealer_positions_short_all",
    "open_interest_all",
]

GOLD_NAME = "GOLD - COMMODITY EXCHANGE INC."


@dataclass
class COTSnapshot:
    report_date: datetime
    open_interest: float
    money_manager_long: float
    money_manager_short: float
    producer_long: float
    producer_short: float
    swap_long: float
    swap_short: float

    @property
    def money_manager_net(self) -> float:
        return self.money_manager_long - self.money_manager_short

    @property
    def producer_net(self) -> float:
        return self.producer_long - self.producer_short


class COTClient:
    """Downloads + normalises the CFTC disaggregated COT history."""

    def __init__(self, archive_url: str = COT_ARCHIVE_URL, http_timeout: float = 60.0) -> None:
        self.archive_url = archive_url
        self._http_timeout = http_timeout

    async def _download(self) -> bytes:
        import httpx

        async with httpx.AsyncClient(timeout=self._http_timeout, follow_redirects=True) as client:
            resp = await client.get(self.archive_url)
            resp.raise_for_status()
            return resp.content

    @staticmethod
    def _parse(bytes_io: BytesIO) -> pd.DataFrame:
        with ZipFile(bytes_io) as zf:
            name = next(n for n in zf.namelist() if n.endswith(".txt"))
            with zf.open(name) as fh:
                df = pd.read_csv(fh, encoding="iso-8859-1", low_memory=False)
        return df

    async def history(self) -> pd.DataFrame:
        """Return the COT history filtered to gold, columns prefixed ``cot_``."""
        raw = self._parse(BytesIO(await self._download()))
        df = raw[raw["market_and_exchange_names"].fillna("").str.upper() == GOLD_NAME]
        df = df[_COT_COLUMNS].copy()
        df["report_date_as_yyyy_mm_dd"] = pd.to_datetime(df["report_date_as_yyyy_mm_dd"])
        df = df.rename(columns={"report_date_as_yyyy_mm_dd": "cot_date"})
        for col in df.columns:
            if col not in ("cot_date", "market_and_exchange_names"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for src, dst in [
            ("producer_merchant_processor_user_positions_long_all", "cot_producer_long"),
            ("producer_merchant_processor_user_positions_short_all", "cot_producer_short"),
            ("money_manager_positions_long_all", "cot_mm_long"),
            ("money_manager_positions_short_all", "cot_mm_short"),
            ("swap_dealer_positions_long_all", "cot_swap_long"),
            ("swap_dealer_positions_short_all", "cot_swap_short"),
            ("open_interest_all", "cot_open_interest"),
        ]:
            df[dst] = df[src]
        df = df.drop(columns=[c for c in _COT_COLUMNS if c != "market_and_exchange_names"])
        df["cot_mm_net"] = df["cot_mm_long"] - df["cot_mm_short"]
        df["cot_producer_net"] = df["cot_producer_long"] - df["cot_producer_short"]
        df = df.sort_values("cot_date").set_index("cot_date")
        log.info("COT history loaded: %d weekly rows", len(df))
        return df

    @staticmethod
    def positioning_zscore(cot: pd.DataFrame, window: int = 52) -> pd.Series:
        """How extreme is current money-manager net positioning vs. its 1y history."""
        if cot.empty:
            return pd.Series([0.0])
        mm = cot["cot_mm_net"].rolling(window, min_periods=max(4, window // 2)).agg(["mean", "std"])
        return (cot["cot_mm_net"] - mm["mean"]) / mm["std"].replace(0, 1)

    @staticmethod
    def snapshot(cot: pd.DataFrame) -> COTSnapshot | None:
        if cot.empty:
            return None
        row = cot.iloc[-1]
        return COTSnapshot(
            report_date=row.name.to_pydatetime(),
            open_interest=float(row["cot_open_interest"]),
            money_manager_long=float(row["cot_mm_long"]),
            money_manager_short=float(row["cot_mm_short"]),
            producer_long=float(row["cot_producer_long"]),
            producer_short=float(row["cot_producer_short"]),
            swap_long=float(row["cot_swap_long"]),
            swap_short=float(row["cot_swap_short"]),
        )


def cot_feature_vector(cot: pd.DataFrame) -> dict[str, float]:
    """Feature dict for one point in time (latest report)."""
    snap = COTClient.snapshot(cot)
    z = COTClient.positioning_zscore(cot)
    if snap is None:
        return {
            "cot_mm_net_z": 0.0,
            "cot_mm_net_oi": 0.0,
            "cot_producer_net_oi": 0.0,
        }
    return {
        "cot_mm_net_z": float(z.iloc[-1] if not z.empty else 0.0),
        "cot_mm_net_oi": float(snap.money_manager_net / max(snap.open_interest, 1)),
        "cot_producer_net_oi": float(snap.producer_net / max(snap.open_interest, 1)),
    }