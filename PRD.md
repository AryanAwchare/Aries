# Product Requirements Document (PRD)
## Aries — Hybrid Autonomous AI & Copy-Trading Quant Platform for Gold (XAUUSD)

---

## 1. Executive Summary & Vision

**Aries** is an institutional-grade, multi-strategy quantitative trading platform engineered specifically for **Gold (XAUUSD)** and high-liquidity financial instruments. 

Retail algorithmic trading consistently fails due to three systemic vulnerabilities:
1. **Single-Strategy Fragility:** Standalone technical indicators or standalone ML models overfit historical data and decay rapidly when macroeconomic regimes shift.
2. **Social Copy-Trading Deception:** Vetted leaders on platforms like Myfxbook often mask risk using hidden Martingale, grid averaging, or holding losing trades indefinitely to maintain an artificially high win rate until account liquidation.
3. **Execution & Risk Mismatch:** Prop-firm rules (e.g. strict daily loss limits, maximum trailing drawdown, ban on third-party EA copy latency) are violated due to crude, uncalibrated position sizing.

**Aries solves this via a Dual-Gated Consensus Architecture:**
- **Human Alpha (Leader Tracking):** Real-time ingestion, continuous auditing, and scoring of verified human traders via Myfxbook.
- **Machine Alpha (AI Forecast & Meta-Model):** A time-series foundation forecasting layer (**Kronos**) combined with a probability-calibrated gradient boosting meta-model (**XGBoost**).
- **The Hybrid Gate:** Trade ideas from leaders are **never** blindly executed. An order is dispatched if and only if both the leader signal and the AI meta-model independently validate the setup with statistical confidence.
- **Institutional Risk Governor:** Prop-firm compliant risk filters, half-Kelly dynamic lot sizing, volatility-adjusted ATR stops, and an uncompromisable hardware/software kill switch.

---

## 2. Target Market & Personas

### 2.1 The Prop Firm Evaluator / Funded Trader
- **Need:** Passing Phase 1/Phase 2 evaluations and keeping funded accounts alive without breaching trailing drawdown limits (e.g. FTMO, FundedNext, 5%ers).
- **Pain Point:** Emotional manual interventions, revenge trading, and breaching daily 4–5% loss caps during high-volatility news events.
- **Aries Feature:** Hard prop-profile YAML constraints (`app/risk/prop_profiles/*.yaml`) that hard-reject any trade violating daily loss, consistency, or exposure caps.

### 2.2 The Systematic Quant Trader
- **Need:** High-probability trade setups grounded in walk-forward cross-validation, probability calibration (Brier score / ECE), and zero-leakage feature pipelines.
- **Pain Point:** Lack of transparent meta-labeling; black-box systems that can't be audited.
- **Aries Feature:** Transparent feature store (stationary log returns, RSI slopes, relative ATR, COT positioning) with out-of-sample calibration curves.

### 2.3 The Hands-Off Capital Allocator
- **Need:** Automated capital growth with strict downside risk ceilings.
- **Pain Point:** Blown accounts from copy-trading "top ranked" traders who hide 80% unrealized floating drawdowns.
- **Aries Feature:** Two-tier leader discovery, toxic flow filters, and automated demotion of erratic leaders.

---

## 3. Product Architecture & Operating Modes

Aries enforces an explicit operating mode state machine (`TRADING_MODE`):

| Operating Mode | Execution Client | Risk Profile | Purpose |
| :--- | :--- | :--- | :--- |
| `backtest` | Vectorized / Event loop | Full Risk Engine | Historical walk-forward simulation, strategy optimization |
| `replay` | `PaperBroker` (Live code path) | Full Risk Engine + Audit | Replaying recorded tick/M15 feeds through exact production code |
| `paper` | `PaperBroker` | `default_demo` | Real-time simulated execution with simulated spread & slippage |
| `prop_eval` | `MetaApiClient` / MT5 EA | Firm-specific YAML | Live evaluation account with strict firm-enforced drawdown rules |
| `live` | `MetaApiClient` / MT5 EA | Live capital profile | Live funded capital deployment with mandatory API keys |

> [!IMPORTANT]
> The platform fails fast on startup if `TRADING_MODE=live` while using a demo profile or if live credentials (`METAAPI_TOKEN` or `EA_API_KEY`) are missing.

---

## 4. Detailed Feature Specifications

### 4.1 Data Ingestion & Engineering Engine
- **Asset Scope:** Primary focus on `XAUUSD` (Spot Gold / USD) at `M15` (15-minute) resolution, expandable to EURUSD and major indices.
- **CFTC Commitments of Traders (COT):** Ingest weekly non-commercial net positioning reports to establish macro trend bias.
- **Point-in-Time Integrity:** Strict prohibition of future data leakage. Feature calculation on bar $t$ only uses data up to timestamp $t$ (enforced by `app/backtest/leakage.py`).
- **Feature Set:**
  - Stationary log returns over 1, 4, 16, and 48 bars.
  - Volatility-normalized ATR (`ATR_14 / Close`).
  - RSI (14) momentum and 3-bar derivative slope.
  - Bollinger Band %B and bandwidth.
  - Cyclic session timestamps (Sine/Cosine encodings of UTC hour and day-of-week).

### 4.2 Leader Discovery, Scoring & Toxic Flow Filter
- **Two-Tier Architecture:**
  - **Watch Tier (Up to 50 leaders):** Monitored every 12 hours via bulk API (`get-watched-accounts`). Ranked on risk-adjusted metrics.
  - **Live Tier (Top 3–5 leaders):** Polled every 5–15 seconds for active orders (`get-open-trades`).
- **Eligibility Criteria:**
  - Account age $\ge 180$ days.
  - Total closed trades $\ge 100$.
  - Historical peak drawdown $\le 25\%$.
  - Active within the last 30 days.
  - Minimum 3 observation cycles before promotion.
- **Toxic Flow & Martingale Filter:**
  - If a leader increases lot size while a position is in floating loss $> 40$ pips, the leader is immediately flagged as `TOXIC_MARTINGALE` and evicted from the Live Tier.

### 4.3 AI Predictive Engine (Kronos + XGBoost Meta-Model)
- **Time-Series Foundation Model (Kronos):** Computes multi-step forward horizon price paths and trend probabilities. If uninstalled or unavailable, gracefully falls back to a neutral forecast without crashing.
- **XGBoost Meta-Classifier:** Ingests the technical feature vector, Kronos forecast vector, and leader signal event to output:
  $$P(\text{Profitable} \mid \text{Market Features}, \text{Leader Event})$$
- **Calibration Layer:** Isotonic Regression / Platt Scaling applied to map model logits into reliable empirical probabilities.
- **Decision Gate:** An order is validated only if calibrated $P \ge \tau$ (default $\tau = 0.60$).

### 4.4 Risk Governor & Circuit Breakers
- **Multi-Level Drawdown Defense:**
  1. *Operating Ceiling:* Warning threshold at $60\%$ of allowable daily loss.
  2. *Bot Soft Emergency Stop:* Trading halted for the session at $75\%$ of allowable daily loss.
  3. *Firm Hard Limit:* Hard kill switch liquidation at $100\%$ limit (e.g. 4.0% daily loss).
- **Position Sizing:** Fractional Kelly sizing modulated by ATR volatility:
  $$\text{Lot Size} = \min\left(\frac{\text{Account Equity} \times \text{Risk \%}}{\text{Stop Loss Distance (pips)} \times \text{Pip Value}}, \text{Max Allowed Lots}\right)$$
- **Stop Loss & Take Profit Discipline:**
  - Stop Loss: $1.5 \times \text{ATR}_{14}$ (hard ceiling: 60 pips on Gold).
  - Take Profit: Dynamic $2.0 \times \text{ATR}_{14}$ or trailing breakeven trigger at $+1.0 \times \text{ATR}$.
- **Rate Limiting:** Maximum 3 order attempts per 60-second window to prevent runaway loops.

### 4.5 Execution Layer (PaperBroker & MetaApi / MT5)
- **PaperBroker:** Local deterministic fill simulator accounting for bid-ask spread expansion during London/NY overlaps and simulated slippage (0.5 to 2.0 pips).
- **MT5 EA Bridge / MetaApi:** Asynchronous order submission with idempotent ticket reconciliation. If an order fill response is not acknowledged within 5 seconds, an automatic reconciliation query is dispatched before re-trying.

---

## 5. Non-Functional Requirements

### 5.1 Performance & Latency
- Signal evaluation pipeline latency (Feature calculation $\rightarrow$ XGBoost inference $\rightarrow$ Risk check) must complete in $\le 100\text{ms}$.
- Leader poll execution round-trip $\le 2000\text{ms}$.

### 5.2 Fault Tolerance & Recovery
- All open orders and audit events are committed to persistent storage (`app.db` or PostgreSQL) before dispatch.
- In the event of process termination, active orders on the broker carry hard broker-side Stop Losses to prevent unmonitored exposure.

### 5.3 Security & Secrets Management
- Zero secret leakage in logs or API responses.
- API authentication on FastAPI endpoints via Bearer tokens or API keys (`EA_API_KEY`).

---

## 6. Success Metrics & Key Performance Indicators (KPIs)

1. **Model Reliability:** Out-of-sample Test AUC $\ge 0.65$; Expected Calibration Error (ECE) $\le 0.08$.
2. **Strategy Expectancy:** Backtested & Paper-traded Profit Factor $\ge 1.65$; Sharpe Ratio $\ge 1.80$.
3. **Capital Preservation:** Maximum cumulative drawdown $\le 6.0\%$; Zero prop-firm daily breach events.
4. **Execution Fidelity:** Slippage between intended price and filled price $\le 1.2$ pips on $95\%$ of orders.
