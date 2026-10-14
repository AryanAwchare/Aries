# Aries — Profitable Trading Implementation Plan

**Status**: Active | **Created**: 2026-10-14 | **Target**: AUC ≥ 0.65, Profit Factor ≥ 1.65, Max DD ≤ 6%

---

## Executive Summary

Current state: **M15 AUC 0.512 (random), M1 AUC 0.543 (loses money after costs)**. Neither model is tradeable.

This plan addresses three pillars in priority order:
1. **Alpha Improvement** — Make the algo viable (AUC ≥ 0.62)
2. **Leader Discovery + Copy Trading** — Add human alpha stream
3. **Hybrid Execution** — Combine both with prop-firm risk discipline

---

## Phase 1: Alpha Improvement (Weeks 1–4)

### 1.1 Regime-Aware Modeling (Week 1)
- [ ] Add HMM/volatility regime detector (3 states: trend/range/volatile) in `app/features/regime.py`
- [ ] Train separate XGBoost/LightGBM models per regime
- [ ] Implement regime-switching inference in `signal_pipeline.py`
- [ ] **Verification**: Walk-forward AUC ≥ 0.58 on M15 with regime models vs 0.512 baseline

### 1.2 Microstructure Features (Week 1–2)
- [ ] Add tick-data ingestion (MT5 `CopyTicksRange` or Polygon.io) → `app/data/ticks.py`
- [ ] Implement: bid-ask spread, tick rule, VPIN, order flow imbalance, toxicity index
- [ ] Add to feature matrix in `calculate_indicator_block`
- [ ] **Verification**: Feature importance shows microstructure features in top 10

### 1.3 Better Meta-Model + Tuning (Week 2)
- [ ] Swap XGBoost → LightGBM + CatBoost ensemble in `app/metamodel/xgboost_model.py`
- [ ] Add Optuna hyperparameter optimization (50 trials, purged CV)
- [ ] Implement purged/embargoed walk-forward CV (`app/backtest/purged_cv.py`)
- [ ] **Verification**: M15 AUC ≥ 0.62 on purged test set

### 1.4 Alternative Data Integration (Week 3)
- [ ] COT non-commercial net position → feature (already ingested, not used)
- [ ] Add funding rate (perpetuals basis) from Binance/Bybit API
- [ ] Add news sentiment using **Gemini API** for gold-related news (XAUUSD, gold, precious metals, Fed, rates)
- [ ] Add DXY/US10Y correlation features
- [ ] **Verification**: Ablation shows +0.02 AUC per alternative data source

### 1.5 Label Engineering (Week 3)
- [ ] Replace fixed horizon/threshold → triple-barrier labeling (mlfinlab)
- [ ] Dynamic volatility-scaled barriers (ATR-multiples)
- [ ] Meta-labeling: primary model predicts direction, secondary predicts profit probability
- [ ] **Verification**: Label distribution balanced, leakage check passes

### 1.6 Cost-Aware Training (Week 4)
- [ ] Incorporate transaction costs into loss function (weight samples by net PnL)
- [ ] Train with realistic spread/slippage/commission per timeframe
- [ ] Add execution delay simulation (1 bar M15, 1 bar M1)
- [ ] **Verification**: Backtest Sharpe ≥ 1.5, Profit Factor ≥ 1.5 on M15

---

## Phase 2: Leader Discovery + Copy Trading (Weeks 3–5)

### 2.1 Myfxbook Discovery Engine (Week 3)
- [ ] Build `app/leaders/discovery.py`: scrape Myfxbook "Top Systems" for XAUUSD
- [ ] Filters: age ≥ 365d, trades ≥ 200, max DD ≤ 20%, profit factor ≥ 1.5, real account only
- [ ] Auto-populate watch tier (up to 50) → persist to `data/roster.json`
- [ ] **Verification**: Discovery returns ≥ 10 qualified gold leaders

### 2.2 Live-Tier Polling + Toxic Flow (Week 4)
- [ ] Schedule `poll_leader_trades` every 30s for live tier (max 5)
- [ ] Implement martingale/grid detection: lot increase while floating loss > 40 pips → instant evict
- [ ] Add leader health scoring: win rate, avg R-multiple, consecutive losses, recovery factor
- [ ] **Verification**: Live tier updates within 60s of leader trade; toxic detection fires on synthetic grid

### 2.3 Hybrid Gate Integration (Week 4)
- [ ] Wire leader signals into `StrategyEngine` (already coded in `hybrid_gate.py`)
- [ ] Decision logic:
  ```
  IF leader_score ≥ 4.0 AND meta_prob ≥ 0.65 AND directions_align → COPY_TRADE (0.5x size)
  ELIF meta_prob ≥ 0.70 AND no_leader_conflict → ALGO_TRADE (1.0x size)
  ELSE → NO_TRADE
  ```
- [ ] Add copy-trade sizing: `min(leader_lots * equity_ratio, risk_engine_max_lots)`
- [ ] **Verification**: Hybrid backtest shows improved Sharpe vs algo-only

### 2.4 Leader Dashboard + Controls (Week 5)
- [ ] Extend `/api/leaders/roster` with live-tier status, last trade, health score
- [ ] Add `POST /api/leaders/live-tier` to manually promote/demote
- [ ] Frontend: leader cards with copy-trade toggle, risk metrics, trade history
- [ ] **Verification**: Can add/remove leaders via UI; trades appear in signal log

---

## Phase 3: Hybrid Execution + Prop Discipline (Weeks 5–7)

### 3.1 Prop Profile Hardening (Week 5)
- [ ] Create `app/risk/prop_profiles/ftmo.yaml`, `fundednext.yaml`, `5ers.yaml`
- [ ] Enforce: daily loss ≤ 4%, max trailing DD ≤ 6%, consistency rule, max positions
- [ ] Add pre-trade simulation: "would this trade breach any limit?"
- [ ] **Verification**: Backtest with prop profile shows zero limit breaches

### 3.2 Paper → Prop Eval → Live Pipeline (Week 6)
- [ ] Implement mode transition guards in `config.py`:
  - `paper` → `prop_eval`: requires MetaApi token, zero DD breaches 30 days
  - `prop_eval` → `live`: requires passed eval, live MetaApi token, signed risk acknowledgment
- [ ] Add position reconciliation on startup (query broker → sync DB)
- [ ] **Verification**: Full mode transition test passes without manual steps

### 3.3 Kill Switch + Monitoring (Week 6)
- [ ] Fix `dashboard.py` kill switch inversion (rejects when enabled)
- [ ] Add real-time equity curve WebSocket (`/ws/equity`)
- [ ] Add Telegram alerts: kill switch triggered, DD > 50% limit, leader evicted
- [ ] **Verification**: Kill switch test blocks orders in < 100ms

### 3.4 Performance Attribution (Week 7)
- [ ] Decompose PnL: algo vs copy, long vs short, regime, time-of-day
- [ ] Weekly automated report → Telegram + dashboard
- [ ] Add model drift detection: rolling AUC window, feature distribution shift
- [ ] **Verification**: Attribution report generated automatically after paper week

---

## Phase 4: Validation & Go/No-Go (Week 8)

### 4.1 30-Day Paper Trading (Week 5–8, parallel)
- [ ] Run `paper` mode 24/7 on M15 hybrid (algo + leaders)
- [ ] Daily checkpoint: AUC rolling, DD, trade count, leader health
- [ ] **Go criteria**: 
  - Rolling 30-day AUC ≥ 0.60
  - Profit Factor ≥ 1.5
  - Max DD ≤ 4%
  - Zero prop limit breaches
  - ≥ 50 trades (statistical significance)

### 4.2 Prop Evaluation Account (Week 8+)
- [ ] If paper passes: deploy to `prop_eval` with FTMO/FundedNext $50k
- [ ] Same criteria over 30 calendar days
- [ ] **Go to live**: Passed eval + live capital allocated

---

## Resource Requirements

| Resource | Status | Action |
|----------|--------|--------|
| MT5 terminal + EA compiled | ❌ | You: F7 compile `mt5/ea_bridge.mq5` |
| MetaApi token | ❌ | You: Create account, add token to `.env` |
| Tick data source | ❌ | Decision: MT5 `CopyTicksRange` (free) vs Polygon.io ($99/mo) |
| Myfxbook credentials | ❌ | You: Add `MYFXBOOK_EMAIL/PASSWORD` to `.env` (roster seeded with 5 leaders) |
| Telegram bot token | ❌ | You: Create bot, add `TELEGRAM_BOT_TOKEN/CHAT_ID` |
| **Gemini API key** | ✅ Available | You: Add `GEMINI_API_KEY` to `.env` for sentiment/regime features |

---

## Seeded Leader Roster (Watch Tier)

| ID | Name | Trader | Style | Verified | Risk Flags |
|----|------|--------|-------|----------|------------|
| 10515575 | Gold Trade Pro MT5 Live | strueli | Breakout/Swing (9.7h hold) | ✅ TR + VP | None |
| 12106293 | Happy Gold - OxS | HappyForex | Momentum Scalping (43s hold) | ✅ TR + VP | Broker slippage sensitive |
| 10740968 | Golden Buffalo high return | atlantisvn | Aggressive Intraday | ✅ TR only | **50% DD, grid suspected** |
| 11340280 | SuperGoldBot | SuperGoldBot | High Win Rate (90%) | ✅ TR only | **Tail risk: avg loss 7.6x avg win** |
| 12076857 | SONIC AI | SonicExperts | Conservative Intraday (27m hold) | ✅ TR + VP | Low return, low DD |

**Next**: Run `scan-leaders scan --apply` after adding Myfxbook credentials to promote top performers to live tier.

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| M15 AUC never reaches 0.62 | Medium | High | Fallback: deploy leader-only copy trading with strict risk |
| Myfxbook API breaks / rate limits | Medium | Medium | Cache roster 12h; add fallback to manual ID entry |
| Leader blows up (martingale) | Medium | High | Toxic flow detection + max 0.5x copy size + daily DD cap |
| Prop firm rule change | Low | High | Profile YAML versioning; automated compliance tests |
| Model drift in production | High | High | Rolling AUC monitor; auto-retrain trigger at AUC < 0.55 |

---

## Success Metrics (from PRD)

| KPI | Target | Current | Measurement |
|-----|--------|---------|-------------|
| Model AUC (OOS) | ≥ 0.65 | 0.512 | Walk-forward test set |
| Profit Factor | ≥ 1.65 | 0.31 (M1) | Cost-inclusive backtest |
| Max Drawdown | ≤ 6% | 4.26% (M1) | Paper/prop eval |
| Daily Loss Breaches | 0 | N/A | Prop profile enforcement |
| Slippage (95th pctl) | ≤ 1.2 pips | ~8 pips (M1) | Execution analysis |

---

## Immediate Next Actions (This Week)

1. **You**: Compile MT5 EA (`mt5/ea_bridge.mq5` → F7 in MetaEditor)
2. **You**: Add `MYFXBOOK_EMAIL/PASSWORD` to `.env`; run `scan-leaders scan --apply` to promote to live tier
3. **You**: Add `GEMINI_API_KEY` to `.env` for news sentiment/regime features
4. **Me**: Start Phase 1.1 (regime detector) + 1.2 (microstructure features)
5. **Both**: Review leader risk flags; demote 10740968 (grid) and 11340280 (tail risk) if scan confirms

---

## File Map for Implementation

```
app/
├── features/
│   ├── regime.py              # NEW: HMM regime detector
│   ├── timeframe.py           # EXISTING: profiles
│   ├── indicators.py          # ADD: microstructure features
│   └── gemini_features.py     # NEW: Gemini API sentiment/regime
├── data/
│   ├── ticks.py               # NEW: tick ingestion
│   └── ohlcv_provider.py      # EXISTING
├── leaders/
│   ├── discovery.py           # NEW: Myfxbook scraper
│   ├── roster.py              # EXISTING: watch/live tiers
│   ├── poll.py                # EXISTING: poll_leader_trades
│   └── toxic_flow.py          # EXISTING: martingale detection
├── metamodel/
│   ├── xgboost_model.py       # REPLACE: LightGBM/CatBoost ensemble
│   └── train.py               # ADD: Optuna + purged CV
├── backtest/
│   ├── purged_cv.py           # NEW: purged/embargoed CV
│   └── runner.py              # EXISTING: cost model
├── strategy/
│   └── hybrid_gate.py         # EXISTING: decision logic
├── risk/
│   ├── engine.py              # EXISTING: risk pipeline
│   └── prop_profiles/         # NEW: firm YAMLs
├── pipelines/
│   └── signal_pipeline.py     # EXISTING: regime-aware inference
└── api/
    └── routes_leaders.py      # EXTEND: discovery endpoints
```

---

## Decision Points (Need Your Input)

1. **Tick data source**: MT5 `CopyTicksRange` (free, local) vs Polygon.io ($99/mo, cleaner)?
2. **Leader IDs**: Use the 4 IDs from research, or wait for discovery engine to find more?
3. **Prop firm target**: FTMO (most popular), FundedNext, or 5%ers? (Affects profile YAML)
4. **Paper trading duration**: 30 days minimum, or extend to 60 for more confidence?
5. **Capital allocation**: Start with $10k paper → $50k prop eval → $100k+ live?

---

*Plan version 1.0 — review and confirm before Phase 1 start.*