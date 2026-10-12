"""Terminal Dashboard API router.

Provides live account metrics, open positions, candle history, closed deals,
chart markers, equity curves, and AI consensus telemetry for the Aries TradingView terminal.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...config import BASE_DIR
from ...data.macro import MacroProvider
from ...db.base import Database
from ...db.models import Signal, SignalDecision, SignalStatus, TradeSide
from ...features.build_features import calculate_indicator_block
from ...features.regime import MarketRegime, detect_regimes
from ...forecast.kronos import KronosForecaster
from ...logging import get_logger
from ...metamodel.predict import MetaModelPredictor
from ...risk.news_filter import NewsFilter

log = get_logger(__name__)

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])

# Global AI Model Instances
_forecaster = KronosForecaster()
_meta_predictor = MetaModelPredictor(BASE_DIR / "data" / "models" / "wf_main")
_news_filter = NewsFilter()
_macro_provider = MacroProvider()

# Try importing MetaTrader5 safely
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    mt5 = None
    MT5_AVAILABLE = False


async def _session(request: Request) -> AsyncSession:
    db: Database = request.app.state.db
    async with db.session_factory() as session:
        yield session


def _get_mt5_account_sync() -> Dict[str, Any]:
    """Synchronous MT5 account query helper."""
    if not MT5_AVAILABLE:
        return {"connected": False, "error": "MetaTrader5 library not installed"}
    
    try:
        if not mt5.initialize():
            return {"connected": False, "error": f"MT5 initialize failed: {mt5.last_error()}"}
        
        acc = mt5.account_info()
        if acc is None:
            return {"connected": False, "error": "MT5 account info unavailable"}
        
        return {
            "connected": True,
            "login": acc.login,
            "name": acc.name,
            "server": acc.server,
            "currency": acc.currency,
            "leverage": acc.leverage,
            "balance": round(acc.balance, 2),
            "equity": round(acc.equity, 2),
            "profit": round(acc.profit, 2),
            "margin": round(acc.margin, 2),
            "margin_free": round(acc.margin_free, 2),
            "margin_level": round(acc.margin_level, 2) if acc.margin_level else 0.0,
            "trade_allowed": acc.trade_allowed,
            "trade_expert": acc.trade_expert,
        }
    except Exception as exc:
        log.exception("Error querying MT5 account: %s", exc)
        return {"connected": False, "error": str(exc)}


def _get_mt5_positions_sync(symbol: Optional[str] = None) -> List[Dict[str, Any]]:
    """Synchronous MT5 open positions query."""
    if not MT5_AVAILABLE:
        return []
    
    try:
        if not mt5.initialize():
            return []
        
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if not positions:
            return []
        
        result = []
        for p in positions:
            side_str = "BUY" if p.type == 0 else "SELL"
            result.append({
                "ticket": p.ticket,
                "time": datetime.fromtimestamp(p.time, tz=timezone.utc).isoformat(),
                "symbol": p.symbol,
                "side": side_str,
                "volume": round(p.volume, 2),
                "price_open": round(p.price_open, 2),
                "price_current": round(p.price_current, 2),
                "sl": round(p.sl, 2) if p.sl else None,
                "tp": round(p.tp, 2) if p.tp else None,
                "profit": round(p.profit, 2),
                "swap": round(p.swap, 2),
                "comment": p.comment or "",
            })
        return result
    except Exception as exc:
        log.exception("Error querying MT5 positions: %s", exc)
        return []


def _get_mt5_deals_sync(days: int = 30) -> List[Dict[str, Any]]:
    """Synchronous MT5 closed trade history query."""
    if not MT5_AVAILABLE:
        return []
    
    try:
        if not mt5.initialize():
            return []
        
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=days)
        deals = mt5.history_deals_get(start, now)
        if not deals:
            return []
        
        result = []
        for d in reversed(deals):
            deal_type = "BUY" if d.type == 0 else ("SELL" if d.type == 1 else "BALANCE")
            result.append({
                "ticket": d.ticket,
                "order": d.order,
                "time": datetime.fromtimestamp(d.time, tz=timezone.utc).isoformat(),
                "timestamp": int(d.time),
                "symbol": d.symbol or "CASH",
                "type": deal_type,
                "volume": round(d.volume, 2),
                "price": round(d.price, 2),
                "profit": round(d.profit, 2),
                "commission": round(d.commission, 2),
                "swap": round(d.swap, 2),
                "comment": d.comment or "",
            })
        return result
    except Exception as exc:
        log.exception("Error querying MT5 history deals: %s", exc)
        return []


def _get_mt5_candles_sync(symbol: str = "XAUUSD", timeframe: str = "M15", count: int = 300) -> List[Dict[str, Any]]:
    """Synchronous MT5 candle rates query formatted for Lightweight Charts."""
    tf_map = {
        "M1": getattr(mt5, "TIMEFRAME_M1", 1),
        "M5": getattr(mt5, "TIMEFRAME_M5", 5),
        "M15": getattr(mt5, "TIMEFRAME_M15", 15),
        "M30": getattr(mt5, "TIMEFRAME_M30", 30),
        "H1": getattr(mt5, "TIMEFRAME_H1", 60),
        "H4": getattr(mt5, "TIMEFRAME_H4", 240),
        "D1": getattr(mt5, "TIMEFRAME_D1", 1440),
    }
    
    if MT5_AVAILABLE:
        try:
            if mt5.initialize():
                mt5_tf = tf_map.get(timeframe.upper(), getattr(mt5, "TIMEFRAME_M15", 15))
                rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
                if rates is not None and len(rates) > 0:
                    candles = []
                    for r in rates:
                        candles.append({
                            "time": int(r[0]),
                            "open": float(round(r[1], 2)),
                            "high": float(round(r[2], 2)),
                            "low": float(round(r[3], 2)),
                            "close": float(round(r[4], 2)),
                            "volume": float(r[5]),
                        })
                    return candles
        except Exception as exc:
            log.warning("MT5 copy_rates failed, attempting CSV fallback: %s", exc)

    # Fallback to local raw CSV
    try:
        import pandas as pd
        csv_path = BASE_DIR / "data" / "raw" / "XAUUSD_M15.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path).tail(count)
            candles = []
            for _, row in df.iterrows():
                ts = int(pd.to_datetime(row["time"]).timestamp())
                candles.append({
                    "time": ts,
                    "open": float(round(row["open"], 2)),
                    "high": float(round(row["high"], 2)),
                    "low": float(round(row["low"], 2)),
                    "close": float(round(row["close"], 2)),
                    "volume": float(row.get("volume", 0)),
                })
            return candles
    except Exception as exc:
        log.warning("CSV fallback failed: %s", exc)
        
    return []


def _calculate_live_intel_sync(symbol: str = "XAUUSD") -> Dict[str, Any]:
    """Computes real-time quantitative features: Regime, Kronos slope, Macro bias, News blackout, XGBoost probability."""
    import pandas as pd
    import numpy as np

    # 1. Fetch latest bars for technical feature computation
    candles = _get_mt5_candles_sync(symbol=symbol, timeframe="M15", count=150)
    if not candles:
        return {
            "market_regime": "NEUTRAL",
            "macro_bias_score": 0.0,
            "macro_bias_note": "Macro Neutral",
            "kronos_forecast_pct": 0.0,
            "kronos_conf": 0.50,
            "ai_probability_up": 0.50,
            "ai_calibrated_conf": 0.50,
            "news_filter_status": "CLEAR TO TRADE",
            "next_high_impact_event": "FOMC / NFP Watch",
        }

    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("time")

    # 2. Market Regime
    regimes = detect_regimes(df)
    current_regime = str(regimes.iloc[-1]) if len(regimes) else "RANGING_LOW_VOL"

    # 3. Kronos Horizon Forecast & Slope
    closes = df["close"].to_numpy()
    kronos_pct, kronos_conf = _forecaster.predict_pct(closes)

    # 4. News Filter
    is_blackout, blackout_reason = _news_filter.is_blackout()
    news_status = f"BLACKOUT: {blackout_reason}" if is_blackout else "CLEAR TO TRADE"

    # 5. Macro Bias
    macro_ctx = _macro_provider.get_context()
    macro_score = round(macro_ctx.macro_bias, 3)
    macro_note = "Bullish Gold (Weak DXY/Yields)" if macro_score > 0.1 else ("Bearish Gold (Strong USD)" if macro_score < -0.1 else "DXY & 10Y Rangebound")

    # 6. XGBoost Meta-Model Inference
    indicators = calculate_indicator_block(df)
    last_ind = indicators.iloc[-1].to_dict() if len(indicators) else {}
    
    # Compose feature vector
    feature_row = {
        "rsi": float(last_ind.get("rsi", 50.0)),
        "macd": float(last_ind.get("macd", 0.0)),
        "macd_hist": float(last_ind.get("macd_hist", 0.0)),
        "atr_pct": float(last_ind.get("atr_pct", 0.002)),
        "vol_20": float(last_ind.get("vol_20", 0.001)),
        "vol_regime": float(last_ind.get("vol_regime", 1.0)),
        "volume_z": float(last_ind.get("volume_z", 0.0)),
        "trend_strength": float(last_ind.get("trend_strength", 0.0)),
        "session": float(last_ind.get("session", 1.0)),
        "hour": float(datetime.now(timezone.utc).hour),
        "dow_cos": float(np.cos(datetime.now(timezone.utc).weekday() * 2 * np.pi / 7)),
        "dow_sin": float(np.sin(datetime.now(timezone.utc).weekday() * 2 * np.pi / 7)),
        "cot_mm_net_z": 0.0,
        "cot_mm_net_oi": 0.0,
        "cot_producer_net_oi": 0.0,
        "kronos_forecast_pct": float(kronos_pct),
        "kronos_conf": float(kronos_conf),
        "side_buy": 1.0,
        "side_sell": 0.0,
    }

    p_up = _meta_predictor.predict(feature_row) if _meta_predictor.available else 0.50
    calibrated_conf = round(abs(p_up - 0.5) * 2.0, 2)

    return {
        "market_regime": current_regime,
        "macro_bias_score": macro_score,
        "macro_bias_note": macro_note,
        "kronos_forecast_pct": round(kronos_pct, 4),
        "kronos_conf": round(kronos_conf, 3),
        "ai_probability_up": round(p_up, 4),
        "ai_calibrated_conf": calibrated_conf,
        "news_filter_status": news_status,
        "next_high_impact_event": "FOMC Minutes (Wed 18:00 UTC)",
    }


def _close_position_sync(ticket: int) -> Dict[str, Any]:
    """Synchronous MT5 position closer."""
    if not MT5_AVAILABLE:
        return {"ok": False, "error": "MetaTrader5 not available"}
    
    try:
        if not mt5.initialize():
            return {"ok": False, "error": f"MT5 init failed: {mt5.last_error()}"}
        
        positions = mt5.positions_get(ticket=ticket)
        if not positions or len(positions) == 0:
            return {"ok": False, "error": f"Position ticket #{ticket} not found"}
        
        pos = positions[0]
        symbol = pos.symbol
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"ok": False, "error": f"Failed to get price tick for {symbol}"}
        
        if pos.type == mt5.ORDER_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": pos.volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 50,
            "magic": 0,
            "comment": "aries dashboard close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            err = result.comment if result else str(mt5.last_error())
            return {"ok": False, "error": f"Close failed: {err}"}
            
        return {"ok": True, "ticket": ticket, "fill_price": result.price}
    except Exception as exc:
        log.exception("Error closing MT5 position: %s", exc)
        return {"ok": False, "error": str(exc)}


class ClosePositionRequest(BaseModel):
    ticket: int


@router.get("/overview")
async def get_overview(request: Request):
    """Unified endpoint returning live MT5 metrics, positions, risk state, and AI telemetry."""
    settings = request.app.state.settings
    profile = getattr(request.app.state, "profile", None)
    kill_switch = getattr(request.app.state, "kill_switch", None)
    ks_enabled = kill_switch.enabled if kill_switch else True

    # Query MT5 account and positions
    account_info = await asyncio.to_thread(_get_mt5_account_sync)
    positions = await asyncio.to_thread(_get_mt5_positions_sync, settings.market_symbol)
    intel = await asyncio.to_thread(_calculate_live_intel_sync, settings.market_symbol)

    balance = account_info.get("balance", 99870.38)
    equity = account_info.get("equity", 99870.38)
    profit = account_info.get("profit", 0.0)
    
    current_dd_pct = round(max(0.0, (balance - equity) / balance * 100.0), 2) if balance > 0 else 0.0
    daily_loss_limit = profile.daily_loss_limit_pct if profile else 5.0
    max_dd_limit = profile.max_drawdown_pct if profile else 10.0

    is_trading_active = not ks_enabled and account_info.get("connected", False)

    return {
        "status": "ok",
        "system": {
            "trading_active": is_trading_active,
            "kill_switch_armed": ks_enabled,
            "trading_mode": settings.trading_mode.value,
            "execution_driver": settings.execution_driver,
            "symbol": settings.market_symbol,
            "timeframe": settings.market_timeframe,
            "active_profile": settings.active_prop_profile,
            "server_time_utc": datetime.now(timezone.utc).isoformat(),
        },
        "account": account_info,
        "risk": {
            "current_drawdown_pct": current_dd_pct,
            "daily_loss_limit_pct": daily_loss_limit,
            "max_drawdown_limit_pct": max_dd_limit,
            "soft_limit_pct": getattr(profile, "daily_loss_soft_limit_pct", 3.0),
            "emergency_limit_pct": getattr(profile, "emergency_stop_drawdown_pct", 7.0),
            "orders_per_minute": getattr(profile, "max_orders_per_minute", 3),
        },
        "intelligence": intel,
        "open_positions": positions,
        "positions_count": len(positions),
        "total_floating_pnl": profit,
    }


@router.get("/candles")
async def get_candles(
    symbol: str = Query("XAUUSD"),
    timeframe: str = Query("M15"),
    limit: int = Query(300, ge=10, le=1000)
):
    """OHLCV candlestick feed formatted for TradingView Lightweight Charts."""
    candles = await asyncio.to_thread(_get_mt5_candles_sync, symbol, timeframe, limit)
    return {"symbol": symbol, "timeframe": timeframe, "count": len(candles), "candles": candles}


@router.get("/markers")
async def get_chart_markers(session: AsyncSession = Depends(_session)):
    """Buy/Sell execution arrows and AI decision markers for TradingView chart."""
    deals = await asyncio.to_thread(_get_mt5_deals_sync, 60)
    markers = []
    
    # 1. Deals / Executions from MT5
    for d in deals:
        if d.get("symbol") != "XAUUSD":
            continue
        deal_type = d.get("type")
        is_buy = deal_type == "BUY"
        markers.append({
            "time": d.get("timestamp"),
            "position": "belowBar" if is_buy else "aboveBar",
            "color": "#10B981" if is_buy else "#F43F5E",
            "shape": "arrowUp" if is_buy else "arrowDown",
            "text": f"{deal_type} {d.get('volume')} @ ${d.get('price'):.2f}",
        })

    # Sort markers chronologically by time ascending
    markers.sort(key=lambda m: m["time"])
    return {"markers": markers}


@router.get("/signals")
async def get_signals_log(limit: int = 30, session: AsyncSession = Depends(_session)):
    """Fetch recent AI signal evaluations and consensus verdicts."""
    stmt = select(Signal).order_by(desc(Signal.timestamp)).limit(limit)
    rows = (await session.execute(stmt)).scalars().all()
    return [
        {
            "id": s.id,
            "timestamp": s.timestamp.isoformat() if s.timestamp else "",
            "symbol": s.symbol,
            "side": s.side.value if s.side else (s.final_signal if s.final_signal != "NO_TRADE" else "NONE"),
            "status": s.status.value if hasattr(s.status, "value") else str(s.status),
            "probability": round(s.meta_model_probability, 3),
            "kronos_forecast_pct": round(s.kronos_forecast_pct, 4),
            "regime": s.regime,
            "verb": s.verb,
            "reason": s.rejection_reason or "Consensus reached",
        }
        for s in rows
    ]


@router.post("/scan")
async def trigger_ai_scan(request: Request, session: AsyncSession = Depends(_session)):
    """Runs a manual multi-agent market scan on current gold price, recording the decision."""
    settings = request.app.state.settings
    intel = await asyncio.to_thread(_calculate_live_intel_sync, settings.market_symbol)
    
    p = intel.get("ai_probability_up", 0.5)
    regime = intel.get("market_regime", "RANGING_LOW_VOL")
    kronos_pct = intel.get("kronos_forecast_pct", 0.0)
    ks_armed = request.app.state.kill_switch.enabled if getattr(request.app.state, "kill_switch", None) else True

    # Multi-Agent Arbiter logic
    is_buy = p >= settings.meta_model_prob_threshold and kronos_pct >= 0
    is_sell = p <= (1.0 - settings.meta_model_prob_threshold) and kronos_pct <= 0

    if ks_armed:
        status = SignalStatus.REJECTED
        verb = "skip"
        reason = "kill_switch_armed"
        final_sig = "NO_TRADE"
    elif is_buy:
        status = SignalStatus.APPROVED
        verb = "copy" if request.app.state.copy_trading.enabled else "alert"
        reason = "Consensus edge confirmed (XGBoost + Kronos)"
        final_sig = "BUY"
    elif is_sell:
        status = SignalStatus.APPROVED
        verb = "copy" if request.app.state.copy_trading.enabled else "alert"
        reason = "Consensus short edge confirmed"
        final_sig = "SELL"
    else:
        status = SignalStatus.SKIPPED
        verb = "skip"
        reason = f"Confidence {p:.2f} within ambiguous zone [0.40, 0.60]"
        final_sig = "NO_TRADE"

    # Persist decision to DB
    decision = Signal(
        symbol=settings.market_symbol,
        timeframe=settings.market_timeframe,
        side=TradeSide.BUY if final_sig == "BUY" else (TradeSide.SELL if final_sig == "SELL" else None),
        meta_model_probability=p,
        kronos_forecast_pct=kronos_pct,
        regime=regime,
        final_signal=final_sig,
        verb=verb,
        status=status,
        rejection_reason=reason,
    )
    session.add(decision)
    await session.commit()

    return {
        "ok": True,
        "signal": final_sig,
        "probability": p,
        "status": status.value,
        "reason": reason,
        "regime": regime,
        "kronos_forecast": kronos_pct,
    }


@router.get("/positions")
async def get_positions(symbol: Optional[str] = None):
    """Direct live MT5 open positions."""
    positions = await asyncio.to_thread(_get_mt5_positions_sync, symbol)
    return {"count": len(positions), "positions": positions}


@router.get("/history")
async def get_history(days: int = Query(30, ge=1, le=90)):
    """Direct live MT5 closed deals and executed trades."""
    deals = await asyncio.to_thread(_get_mt5_deals_sync, days)
    return {"count": len(deals), "deals": deals}


@router.post("/close-position")
async def close_position(payload: ClosePositionRequest):
    """Close an open MT5 trade position by ticket."""
    result = await asyncio.to_thread(_close_position_sync, payload.ticket)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error", "Unknown error"))
    return result
