# Autonomous Agents Specification (AGENTS.md)
## Aries — Multi-Agent Trading Desk & Systems Orchestration

---

## 1. Multi-Agent System (MAS) Philosophy

Aries models an institutional quantitative trading firm where distinct autonomous agents perform specialized tasks with strict division of responsibility and clear separation of powers.

**Core Rules of Separation:**
1. **The Alpha Agent cannot execute:** The agents generating market forecasts or tracking leaders have zero authority to place or size orders.
2. **The Arbiter cannot ignore the Risk Sentinel:** Even if the Consensus Arbiter detects a 99% probability setup, the Risk Sentinel possesses absolute unilateral veto power.
3. **Execution is Idempotent & Monitored:** The Execution Agent only acts upon cryptographically verified or schema-validated orders and must reconcile every fill.

```
                          ┌───────────────────────────┐
                          │   ORCHESTRATOR / ARBITER  │
                          │   (Consensus Arbiter)     │
                          └─────────────┬─────────────┘
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
  ┌───────────────────────┐  ┌───────────────────────┐  ┌───────────────────────┐
  │  Alpha ML Agent       │  │  Leader Scout Agent   │  │  Macro Sentiment      │
  │  (XGBoost + Kronos)   │  │  (Myfxbook / Flow)    │  │  (COT & Session Bias) │
  └───────────────────────┘  └───────────────────────┘  └───────────────────────┘
                                        │
                                        ▼
                             ┌───────────────────────┐
                             │  Risk Sentinel Agent  │
                             │  (Hard Veto & Sizing) │
                             └──────────┬────────────┘
                                        │
                                        ▼
                             ┌───────────────────────┐
                             │ Execution Agent       │
                             │ (PaperBroker / MT5)   │
                             └───────────────────────┘
```

---

## 2. Agent Catalog & Behavioral Contracts

### 2.1 Alpha ML Agent (`agent_alpha_ml`)
- **Job Title:** Quantitative ML Research & Inference Specialist
- **Mission:** Extract statistical directional edge from stationary technical features and time-series foundation forecasts.
- **Inputs:**
  - Standardized feature vector $X_t \in \mathbb{R}^{d}$ derived from OHLCV bar close (timeframe-configurable: M1, M5, M15, M30, H1; defaults to M15).
  - Multi-step horizon forecast from Kronos foundation model.
- **Output:**
  - Directional probability $P(\text{Up}) \in [0.0, 1.0]$.
  - Calibrated confidence metric and feature attribution vector.
- **Timeframe Profiles:** `app/features/timeframe.py` provides `TimeframeProfile` per timeframe — horizon bars, label threshold, min-move %, ATR stop, lookback hours, Kronos horizon. Feature windows (RSI 14, MACD 12/26/9, ATR 14, etc.) are **bar-based and identical across timeframes** to isolate the timeframe variable.
- **Failure Mode / Fallback:**
  - If Kronos model inference times out ($> 500\text{ms}$), fall back to stationary exponential moving average momentum slope.
  - If XGBoost confidence is within ambiguous zone $[0.45, 0.55]$, emit `DIRECTION_NEUTRAL` with zero confidence.

### 2.2 Leader Scout Agent (`agent_leader_scout`)
- **Job Title:** Social Intelligence & Toxic Flow Surveillance Officer
- **Mission:** Ingest, rank, and sanitize human trade signals from Myfxbook and external copy accounts.
- **Operational Rules:**
  - **Two-Tier Roster:** Polls up to 50 candidates in the Watch tier every 12 hours; continuously monitors top 3–5 candidates in the Live tier.
  - **Toxic Flow Detection:** Continuously calculates each leader's floating drawdown vs position sizing. If lot size increases while floating loss exceeds 40 pips (Martingale indicator), the leader is immediately evicted and any active mirror trade is marked for defensive exit.
- **Output:**
  - Signal payload containing leader ID, symbol, action (`BUY` / `SELL`), open price, and leader composite score.

### 2.3 Consensus Arbiter Agent (`agent_arbiter`)
- **Job Title:** Chief Investment Officer & Consensus Engine
- **Mission:** Synthesize machine intelligence and human alpha into a unified order intent.
- **Decision Logic:**
  - **Case 1 (Strong Consensus):** Alpha ML and Leader Scout agree in direction with high scores ($P_{\text{ML}} \ge 0.60$ and $\text{Score}_{\text{Leader}} \ge 3.0$) $\rightarrow$ Generate `ORDER_INTENT` with high conviction.
  - **Case 2 (Autonomous ML Only):** If `copy_trading_allowed=false`, evaluate pure Alpha ML setup. If $P_{\text{ML}} \ge 0.65$ $\rightarrow$ Generate `ORDER_INTENT`.
  - **Case 3 (Conflict):** Leader goes Long while Alpha ML predicts Short $\rightarrow$ **Veto trade completely**. Capital preservation overrides action.

### 2.4 Risk Sentinel Agent (`agent_risk_sentinel`)
- **Job Title:** Chief Risk Officer (CRO) & Circuit Breaker Governor
- **Mission:** Enforce prop-firm survival rules and protect capital balance with zero tolerance.
- **Authority:** Absolute unilateral veto over all orders.
- **Checks Enforced Per Order:**
  1. **Kill Switch:** Is the emergency stop active? If yes, immediately reject and log.
  2. **Daily Drawdown Limit:** Is current day's cumulative loss $\ge 75\%$ of prop ceiling? If yes, block new trades. At $100\%$, trigger hard liquidation.
  3. **Volatility-Adjusted Position Sizing:** Calculate lot size using half-Kelly criteria bounded by strict ATR stop distance:
     $$\text{Lots} = \text{clamp}\left(\frac{\text{Equity} \times 0.01}{1.5 \times \text{ATR}_{14} \times \text{Tick Value}}, 0.01, \text{Max Lots}\right)$$
  4. **Rate Limiter:** Maximum 3 orders per 60-second window.

### 2.5 Execution & Reconciliation Agent (`agent_execution`)
- **Job Title:** Senior Execution Trader & Bridge Operator
- **Mission:** Deliver low-latency, deterministic execution through PaperBroker or MetaTrader 5 EA bridge.
- **Operational Rules:**
  - In `paper` mode: Routes to local `PaperBroker` with simulated bid-ask spread and slippage.
  - In `prop_eval` / `live` mode: Dispatches order to MT5 EA via HTTP/WebSocket and tracks ticket acknowledgement.
  - **Reconciliation Protocol:** If ticket ID is not returned within $2500\text{ms}$, freezes order dispatch and probes the broker terminal to check if the order was filled, partial-filled, or rejected.

---

## 3. Inter-Agent Communication Protocols

Agents exchange messages over an asynchronous event bus using standardized JSON schemas.

### 3.1 Signal Event Schema (`event.signal`)
```json
{
  "event_id": "c7a8b9e1-3d2f-4a5b-9c8d-1e2f3a4b5c6d",
  "timestamp": "2026-10-12T10:30:00.000Z",
  "source_agent": "agent_alpha_ml",
  "symbol": "XAUUSD",
  "timeframe": "M15",
  "data": {
    "direction": "BUY",
    "probability_raw": 0.742,
    "probability_calibrated": 0.685,
    "kronos_trend_slope": 0.0014,
    "features_hash": "a1b2c3d4e5"
  }
}
```
**Note:** `timeframe` is configurable per profile (M1, M5, M15, M30, H1). M15 is production; M1 model exists at `wf_main_m1` but is not promoted (AUC 0.543 < 0.65 target).

### 3.2 Order Intent Schema (`event.order_intent`)
```json
{
  "intent_id": "f4e3d2c1-b0a9-8765-4321-fedcba098765",
  "timestamp": "2026-10-12T10:30:00.045Z",
  "source_agent": "agent_arbiter",
  "symbol": "XAUUSD",
  "action": "BUY",
  "basis": "HYBRID_CONSENSUS",
  "parameters": {
    "entry_type": "MARKET",
    "target_stop_pips": 45.0,
    "target_tp_pips": 90.0,
    "confidence_score": 0.685
  }
}
```

### 3.3 Risk Verification Schema (`event.risk_verdict`)
```json
{
  "intent_id": "f4e3d2c1-b0a9-8765-4321-fedcba098765",
  "timestamp": "2026-10-12T10:30:00.070Z",
  "verdict": "APPROVED",
  "allocated_lots": 0.42,
  "stop_loss_price": 2081.50,
  "take_profit_price": 2095.00,
  "current_daily_drawdown_pct": 0.85,
  "veto_reason": null
}
```

---

## 4. Subagent Prompt & Persona Engineering

When running automated agent tasks or maintenance scripts, subagents must adhere to the following core prompt contract:

```text
YOU ARE AN AGENT OF THE ARIES QUANTITATIVE TRADING DESK.
YOUR HIGHEST DIRECTIVE IS CAPITAL PRESERVATION AND PROP-FIRM RISK COMPLIANCE.

RULES:
1. NEVER modify risk limits in code or configurations without explicit audit log generation.
2. NEVER disable the Kill Switch or bypass ATR Stop Loss requirements.
3. IN ALL ML PIPELINES: Strictly enforce point-in-time causality. Test datasets must never be seen during feature engineering, normalization, or model training.
4. IN ALL LEADER TRACKING: Immediately flag any trading system that doubles down on losing positions as TOXIC_MARTINGALE.
```
