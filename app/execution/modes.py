"""Trading-mode resolution. The platform never defaults to live."""
from __future__ import annotations

from ..config import TradingMode


def resolve_execution_client(settings, session_factory=None) -> object:
    """Pick the ExecutionClient for the active trading mode + driver.

    * backtest/replay → used only for historical loops (no broker client)
    * paper           → PaperBroker by default, or EaBroker if EXECUTION_DRIVER=ea
    * prop_eval / live → driver-dependent: EaBroker (local MT5) or MetaApiClient

    The client itself carries NO mode — mode is enforced here + by the risk
    engine's profile gate.
    """
    mode = settings.trading_mode

    if settings.execution_driver == "ea":
        from .ea_bridge import EaBroker
        if session_factory is None:
            raise RuntimeError("EXECUTION_DRIVER=ea requires a session_factory")
        return EaBroker(settings, session_factory)

    if mode in (TradingMode.PAPER, TradingMode.BACKTEST, TradingMode.REPLAY):
        return PaperBroker()
    if not settings.metaapi_token:
        raise RuntimeError("TRADING_MODE requires METAAPI_TOKEN to be set")
    account_id = settings.metas_api_prop_account_id if mode is TradingMode.PROP_EVAL else settings.metas_api_demo_account_id
    return MetaApiClient(settings.metaapi_token, account_id)


def ensure_safe_mode(settings) -> TradingMode:
    """Hard guard: an empty/unset trading mode must resolve to paper, never live."""
    return settings.trading_mode if settings.trading_mode is not None else TradingMode.PAPER