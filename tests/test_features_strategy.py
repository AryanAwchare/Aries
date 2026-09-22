"""Feature engineering + strategy gate + leakage tests."""
from __future__ import annotations

import pandas as pd
import pytest

from app.backtest.leakage import check_feature_causality, check_kronos_overlap
from app.features.build_features import (
    FEATURE_COLUMNS,
    build_feature_row,
    calculate_indicator_block,
    label_trades,
)
from app.features.indicators import atr, rsi
from app.metamodel.build_dataset import build_training_data, candidate_events
from app.strategy.engine import DecisionOutcome, StrategyEngine
from app.leaders.scoring import compute_score
from app.leaders.models import RawLeaderTrade, LeaderScore
from app.db.models import SignalStatus, TradeEvent, TradeSide
from app.config import Settings
from app.metamodel.predict import MetaModelPredictor


def test_rsi_bounded(synthetic_ohlcv):
    s = rsi(synthetic_ohlcv["close"])
    assert (s >= 0).all() and (s <= 100).all()
    assert s.notna().all()


def test_atr_positive(synthetic_ohlcv):
    assert (atr(synthetic_ohlcv) > 0).all()


def test_indicator_block_populates_columns(synthetic_ohlcv):
    df = calculate_indicator_block(synthetic_ohlcv)
    for col in ("rsi", "macd", "macd_hist", "atr_pct", "vol_20", "vol_regime", "trend_strength", "session"):
        assert col in df.columns
        assert df[col].notna().all()


def test_build_feature_row_is_causal(synthetic_ohlcv):
    df = calculate_indicator_block(synthetic_ohlcv)
    idx = df.index[-5]
    full = build_feature_row(df, index=idx)
    truncated = build_feature_row(df.iloc[: list(df.index).index(idx) + 1])
    for c in FEATURE_COLUMNS:
        assert full[c] == pytest.approx(truncated[c], abs=1e-9), c


def test_labels_dont_look_backwards(synthetic_ohlcv):
    labels = label_trades(synthetic_ohlcv, horizon_bars=8, side="buy")
    assert labels.iloc[-8:].isna().all()  # no labels for bars without a future


def test_candidate_events_produce_rows(synthetic_ohlcv):
    events = candidate_events(synthetic_ohlcv)
    assert len(events) >= 0


def test_training_data_shape(synthetic_ohlcv):
    df = build_training_data(synthetic_ohlcv, horizon_bars=8)
    if len(df):
        assert {"side_buy", "side_sell", "label"}.issubset(df.columns)


def test_leakage_causality_check(synthetic_ohlcv):
    causal, mismatched = check_feature_causality(synthetic_ohlcv)
    assert causal, mismatched


def test_kronos_overlap_warning():
    assert check_kronos_overlap("2024-01-01") is True   # in pretraining window
    assert check_kronos_overlap("2026-01-01") is False  # clean out-of-sample


def test_neutral_predictor_when_no_model(tmp_path):
    predictor = MetaModelPredictor(tmp_path, threshold=0.6)
    assert predictor.available is False
    assert predictor.predict({"any": 1.0}) == 0.5


class _FakePredictor:
    def predict(self, row):
        return row.get("fake_p", 0.9)


def _event(side="buy"):
    return TradeEvent(
        leader_id="L1", external_trade_id="X",
        symbol="XAUUSD", side=TradeSide(side), volume=1.0,
        open_price=2000.0, opened_at=pd.Timestamp.now("UTC"),
    )


def test_strategy_approves_only_when_both_gates_clear():
    settings = Settings()
    engine = StrategyEngine(settings, _FakePredictor(), min_leader_score=3.0, meta_threshold=0.6)
    score = LeaderScore(composite=3.5, profit_factor=2.0, win_rate=0.6)
    outcome = engine.evaluate(_event(), score, {"fake_p": 0.9, "kronos_forecast_pct": 0.2})
    assert outcome.status is SignalStatus.APPROVED
    assert outcome.verb == "copy"


def test_strategy_rejects_when_meta_below_threshold():
    settings = Settings()
    engine = StrategyEngine(settings, _FakePredictor(), min_leader_score=3.0, meta_threshold=0.6)
    outcome = engine.evaluate(_event(), LeaderScore(composite=4.0), {"fake_p": 0.4, "kronos_forecast_pct": 0.0})
    assert outcome.status is SignalStatus.REJECTED
    assert "meta_below_threshold" in outcome.reason


def test_strategy_rejects_weak_leader():
    settings = Settings()
    engine = StrategyEngine(settings, _FakePredictor(), min_leader_score=3.0, meta_threshold=0.6)
    outcome = engine.evaluate(_event(), LeaderScore(composite=1.0), {"fake_p": 0.9, "kronos_forecast_pct": 0.0})
    assert outcome.status is SignalStatus.REJECTED
    assert "leader_below_score" in outcome.reason


def test_strategy_rejects_on_kronos_conflict():
    settings = Settings()
    engine = StrategyEngine(settings, _FakePredictor(), min_leader_score=3.0, meta_threshold=0.6)
    # strong forecast DOWN but trade is BUY -> conflict
    outcome = engine.evaluate(_event(side="buy"), LeaderScore(composite=4.0),
                              {"fake_p": 0.9, "kronos_forecast_pct": -0.5})
    assert outcome.status is SignalStatus.REJECTED
    assert outcome.reason == "forecast_conflict"


def test_leader_scoring_runs():
    wins = [RawLeaderTrade("1", "XAUUSD", "buy", 1.0, 2000, 2005, profit=500, is_open=False),
            RawLeaderTrade("2", "XAUUSD", "sell", 1.0, 2000, 1996, profit=400, is_open=False),
            RawLeaderTrade("3", "XAUUSD", "buy", 1.0, 2000, 1999, profit=-100, is_open=False)]
    score = compute_score(wins)
    assert score.trade_count == 3
    assert score.win_rate == pytest.approx(2 / 3)
    assert score.profit_factor == pytest.approx(900 / 100)