# Gold (XAUUSD) Hybrid Copy-Trading + AI Strategy Platform

Hybrid system — verified human leaders (**Myfxbook**) provide trade ideas, but a
**Kronos + XGBoost meta-model** must independently confirm a setup before it is
alerted or copied. Execution is prop-firm oriented and mode-aware; paper trading
is the mandatory first live-adjacent step.

```
MARKET DATA (OHLCV, COT)                    LEADER SIGNALS (Myfxbook)
        │                                            │
 FEATURE LAYER (indicators, vol, regime)            │
        │                                            ▼
        └──────────────►      FORECAST LAYER (Kronos) + leader event
                                      │
                                  META-MODEL (XGBoost)
                              → calibrated P(profitable)
                                      │
                               STRATEGY ENGINE
                    (leader signal AND meta-model must agree)
                                      │
                                RISK ENGINE
            (prop profile, sizing, drawdown, order-rate, kill switch)
                                      │
                            EXECUTION ENGINE (MetaApi)
              ┌────────┼──────────┬──────────┬───────────┐
           BACKTEST  REPLAY     PAPER     PROP EVAL    LIVE
```

**Both Kronos and leader signals are inputs — neither decides alone.** The
leader branch can be switched off entirely per profile (`copy_trading_allowed:
false`) without touching the ML side.

---

## Project layout

```
app/
  config.py            validated settings; trading_mode can never silently be live
  services.py          kill switch, copy-trading toggle, db singletons
  db/                  async SQLAlchemy 2.0 models + audit trail
  data/                OHLCV providers (CSV/MetaApi/synthetic) + CFTC COT
  features/            indicators + point-in-time feature vectors
  leaders/             Myfxbook client, daily scoring, persistence
  forecast/            Kronos wrapper (forecast % + confidence)
  metamodel/           XGBoost vs logistic baseline, calibration, prediction
  strategy/            hybrid gate (leader AND meta-model must agree)
  risk/                risk engine + prop_profiles/*.yaml + sizing + rate limiter
  execution/           mode-aware executor, MetaApi client, paper broker, kill switch
  alerts/              Telegram notifier
  scheduler/           APScheduler jobs (leader poll, daily scoring, weekly COT)
  backtest/            runner, walk-forward, replay, leakage checks
  pipelines/           live signal pipeline (leader event → decision → order)
  api/                 FastAPI routes (health, leaders, signals, risk, modes)
  cli.py               train / backtest / walk-forward / replay / poll
scripts/setup.ps1      one-shot Windows setup (venv + deps + db)
tests/                 risk, features/strategy, metamodel, backtest
```

## Setup

Python **3.10–3.13** recommended (`setup.ps1` targets 3.12 — torch, xgboost and
pandas don't have stable Python 3.14 Windows wheels yet).

```powershell
# one-shot (creates .venv, installs deps, copies .env, creates DB)
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1

# or manually:
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env      # then fill in credentials
.\.venv\Scripts\python.exe -m app.cli setup-db
```

## Try it (no live credentials needed)

```powershell
# Deterministic synthetic data exercises the whole stack end-to-end:
.\.venv\Scripts\python.exe -m app.cli backtest --synthetic
.\.venv\Scripts\python.exe -m app.cli walk-forward --synthetic
.\.venv\Scripts\python.exe -m app.cli replay --synthetic

# Tests
.\.venv\Scripts\python.exe -m pytest tests -q

# API + dashboard endpoints
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
#  http://127.0.0.1:8000/health , /api/risk/status , /api/signals ...
```

## Modes

`TRADING_MODE` in `.env`: `backtest | replay | paper | prop_eval | live`.
Default is **paper**. `live` additionally refuses to start while an
internal/demo profile is active and while `METAAPI_TOKEN` is unset.

| mode       | execution client        | risk profile                       |
|------------|-------------------------|------------------------------------|
| backtest   | none (vector loop)      | full risk engine applied per trade |
| replay     | PaperBroker (live code path) | full risk engine + audit       |
| paper      | PaperBroker (demo data) | default_demo (or chosen)           |
| prop_eval  | MetaApiClient (prop account) | firm YAML profile enforced   |
| live       | MetaApiClient (live account) | firm YAML profile enforced   |

## Risk / prop compatibility gate

Every candidate order passes, in order: kill switch → prop-compatibility
(source branch vs `copy_trading_allowed` / `third_party_signals_allowed`) →
trading hours → drawdown (firm hard limit, then bot emergency stop, then
operating ceiling) → daily loss (firm, then soft internal) → profit target →
order-rate limiter → open-position cap → consistency rule → position sizing.
Internal limits always sit strictly below the firm's (e.g. firm 10% drawdown →
bot emergency stop 6%, operating ceiling 4%).

Create `app/risk/prop_profiles/<firm>.yaml` per researched firm and set
`ACTIVE_PROP_PROFILE=<firm>`.

## Copy-trading toggle (Phase 7)

- **OFF** is the default; one click, immediate, no confirmation.
- **ON** requires two explicit confirmations; if the active profile has
  `copy_trading_allowed: false` it is refused outright (API echoes this —
  `POST /api/risk/copy-trading/on` returns 409).

## Open items (from the build plan)

1. **Prop-firm research** → turn 2–3 candidate firms into `prop_profiles/*.yaml`
   and cross-check against `tests/test_risk.py`.
2. **Broker for the paper demo MT5 account** (Pepperstone / IC Markets / …).
3. **Meta-model threshold** = result of Phase 4 backtesting, not a guess.

## Leakage discipline

`app/backtest/leakage.py` automates two checks run before any backtest result
is trusted: (a) feature causality (recomputing a truncated frame must equal the
full-frame snapshot) and (b) Kronos-pretraining overlap (test windows before the
Kronos training cutoff are flagged). Walk-forward split per the plan:
train 2020–2023 → validate 2024 → test 2025–2026.

## Skills

Project-scoped skills for building/maintaining this platform live in
`.opencode/skills/` (copied from `C:\Users\dell\.agents\skills`). They are git-
ignored by design — they're personal tooling, not source code.