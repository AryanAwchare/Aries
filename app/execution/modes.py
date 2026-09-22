"""Trading-mode resolution. The platform never defaults to live."""
from __future__ import annotations

from ..config import TradingMode


def resolve_execution_client(settings) -> object:
    """Pick the ExecutionClient for the active trading mode.

    * backtest/replay → used only for historical loops (no broker client)
    * paper           → PaperBroker against the demo account's market data
    * prop_eval / live → MetaApiClient against prop / live credentials

    The client itself carries NO mode — mode is enforced here + by the risk
    engine's profile gate.
    """
    from .base import MetaApiClient, PaperBroker

    mode = settings.trading_mode
    if mode in (TradingMode.PAPER, TradingMode.BACKTEST, TradingMode.REPLAY):
        return PaperBroker()
    if not settings.metaapi_token:
        raise RuntimeError("TRADING_MODE requires METAAPI_TOKEN to be set")
    account_id = settings.metas_api_prop_account_id if mode is TradingMode.PROP_EVAL else settings.metas_api_demo_account_id
    return MetaApiClient(settings.metaapi_token, account_id)


def ensure_safe_mode(settings) -> TradingMode:
    """Hard guard: an empty/unset trading mode must resolve to paper, never live."""
    return settings.trading_mode if settings.trading_mode is not None else TradingMode.PAPER