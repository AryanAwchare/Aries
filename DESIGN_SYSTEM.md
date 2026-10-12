# Design System Specification (DESIGN_SYSTEM.md)
## Aries Terminal — Institutional Dark Fintech UI

---

## 1. Design Philosophy & Ethos

Aries Terminal is an institutional-grade, data-dense interface engineered for active quantitative traders and portfolio supervisors. Every pixel is purposeful, every animation is functional, and visual noise is eliminated.

### Core Tenets
1. **Zero Layout Jitter:** Numbers, order prices, P&L, timestamps, and confidence percentages must never cause layout shift. All financial numerals are monospaced with tabular figures.
2. **Sub-second Legibility:** Critical status indicators (Risk State, Kill Switch, Order Fill, Directional Bias) must be identifiable in $< 50\text{ms}$ peripheral vision.
3. **Ergonomic Contrast:** Built upon an OLED-friendly obsidian and slate base (`#070A13` to `#0E1424`), with carefully calibrated semantic accents that prevent visual fatigue during long market sessions.
4. **Information Density over Whitespace:** Prioritizes dense financial data (order books, execution logs, feature vectors, leader scores) in clear bento grids without wasteful padding.

---

## 2. Color Palette & Design Tokens

### 2.1 Surfaces & Canvas (Dark Mode Primary)
```css
:root {
  /* Surfaces */
  --bg-canvas: #070A13;         /* Deepest background */
  --bg-surface-subtle: #0B0F19; /* Secondary panels / top nav */
  --bg-surface: #0E1424;        /* Primary card / widget surface */
  --bg-surface-elevated: #151F36;/* Hover states, dropdowns, modals */
  --border-subtle: #1E293B;     /* Default borders */
  --border-focus: #38BDF8;      /* Active / focused control border */
  --border-glass: rgba(255, 255, 255, 0.08); /* Frosted card stroke */

  /* Text & Content */
  --text-primary: #F8FAFC;      /* Primary labels, active numbers */
  --text-secondary: #94A3B8;    /* Field descriptions, headers */
  --text-tertiary: #64748B;     /* Muted metadata, timestamps */
  --text-disabled: #334155;     /* Inactive controls */
}
```

### 2.2 Semantic Financial Palette
```css
:root {
  /* Bullish / Long / Profit */
  --trade-long: #10B981;        /* Emerald 500 */
  --trade-long-hover: #059669;  /* Emerald 600 */
  --trade-long-glow: rgba(16, 185, 129, 0.25);
  --trade-long-surface: rgba(16, 185, 129, 0.08);

  /* Bearish / Short / Loss */
  --trade-short: #F43F5E;       /* Rose 500 */
  --trade-short-hover: #E11D48; /* Rose 600 */
  --trade-short-glow: rgba(244, 63, 94, 0.25);
  --trade-short-surface: rgba(244, 63, 94, 0.08);

  /* Caution / Risk Limit / Warning */
  --status-warning: #F59E0B;    /* Amber 500 */
  --status-warning-glow: rgba(245, 158, 11, 0.25);
  --status-warning-surface: rgba(245, 158, 11, 0.1);

  /* AI & Intelligence Accents */
  --accent-ai: #06B6D4;         /* Cyan 500 (ML XGBoost, Kronos forecast) */
  --accent-leader: #6366F1;     /* Indigo 500 (Myfxbook human alpha) */
  --accent-consensus: #A855F7;  /* Purple 500 (Blended consensus trigger) */
}
```

---

## 3. Typography Hierarchy

### 3.1 Font Stacks
- **UI Font:** `Inter`, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif.
- **Financial & Monospace Font:** `JetBrains Mono`, "SF Mono", Menlo, Monaco, Consolas, monospace.
  - Mandatory CSS rule: `font-variant-numeric: tabular-nums lining-nums; font-feature-settings: "tnum" 1, "zero" 1;`

### 3.2 Type Scale
| Level | Font Family | Size / Line-Height | Weight | Usage |
| :--- | :--- | :--- | :--- | :--- |
| **Hero Metric** | `JetBrains Mono` | 32px / 36px | 700 | Total Equity, Today's Net P&L |
| **Metric Large** | `JetBrains Mono` | 24px / 28px | 600 | Live Price ($2,084.50), Drawdown % |
| **Section Header** | `Inter` | 16px / 22px | 600 | Panel Title, Widget Header |
| **Field Label** | `Inter` | 13px / 18px | 500 | Form label, Table Column Title |
| **Data Row** | `JetBrains Mono` | 13px / 18px | 500 | Order Lot, Stop Loss, Ticket ID |
| **Timestamp / Micro** | `JetBrains Mono` | 11px / 14px | 400 | Server UTC time, Log timestamp |

---

## 4. Component Library Specifications

### 4.1 Master App Navigation Bar
- **Height:** 48px fixed.
- **Left Cluster:**
  - Platform Logo: Minimal geometric Aries constellation icon in `--accent-ai`.
  - Mode Badge: Dynamic pill (`PAPER` in Emerald, `PROP_EVAL` in Amber, `LIVE` in Rose).
  - Symbol & Timeframe: `XAUUSD · M15` dropdown selector.
- **Center Cluster:**
  - Rolling Daily Drawdown Gauge: Horizontal micro-progress bar (`Current: 1.4% / Max: 4.0%`).
  - System Heartbeat: Pulsing green dot ($\le 100\text{ms}$ ping) or amber warning dot.
- **Right Cluster:**
  - Master Kill Switch: Prominent tactile button with safety latch.
  - Active Profile: `default_demo` or `FTMO_100k`.
  - Account Equity: `$104,250.80 (+4.25%)`.

### 4.2 The Emergency Kill Switch (`<KillSwitch />`)
- **Default Armed State:** Dark charcoal container with rose border and text: `[!] KILL SWITCH (ARMED)`.
- **Engagement Interaction:**
  - Desktop: Hold for 500ms or double-click with confirmation modal: *"Liquidate all positions and freeze trading?"*
- **Triggered State:**
  - Persistent flashing red border with label `CIRCUIT BREAKER ACTIVE`.
  - Color transitions to `--trade-short-glow`.
  - Disables all order submission controls.

### 4.3 Signal Consensus Meter (`<SignalConsensusGauge />`)
Displays the alignment between Machine Alpha (XGBoost + Kronos) and Human Alpha (Myfxbook):
```text
┌─────────────────────────────────────────────────────────────┐
│ SIGNAL CONSENSUS ENGINE                                     │
├─────────────────────────────────────────────────────────────┤
│  AI Meta-Model: [ BUY (72% calibrated) ]   ████████░░░░     │
│  Kronos Forecast: [ +14.2 pips / 24 bars ] ██████░░░░░░     │
│  Leader Alpha:  [ 4 of 5 Long (Score 4.2)] ████████░░░░     │
├─────────────────────────────────────────────────────────────┤
│  GATE STATUS: ✅ CONSENSUS CONFIRMED → BUY 0.35 LOTS        │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 Real-time Execution Blotter (`<OrderBlotter />`)
- Tabular data grid with fixed headers and virtualized scrolling for $\ge 1,000$ executions.
- Columns: `Time (UTC)`, `Ticket`, `Source`, `Direction`, `Lots`, `Open Price`, `Current`, `SL / TP`, `Net P&L`, `Action`.
- Right-click context menu: *Close Position, Trailing Breakeven, View Log Audit Trail*.

### 4.5 Leader Roster & Health Card (`<LeaderRosterCard />`)
- Visual card for each monitored Myfxbook leader:
  - Header: Leader Name, Verification Badge, Strategy Style (Scalper, Swing).
  - Metrics Grid: Win Rate %, Profit Factor, Max Drawdown %, Open Trades.
  - **Toxic Flow Indicator:** Green shield icon if healthy; flashing hazard icon if Martingale / Grid behavior is detected.

---

## 5. Micro-Interactions & Animation Guidelines

1. **Price Flash Animations:**
   - When a tick is higher than previous close: 300ms background transition to `--trade-long-surface`.
   - When a tick is lower: 300ms background transition to `--trade-short-surface`.
2. **Modal & Drawer Easing:**
   - Entry: `transform: scale(0.98) -> scale(1.0)`, `opacity: 0 -> 1` over 120ms with cubic-bezier(0.16, 1, 0.3, 1).
   - Exit: `opacity: 1 -> 0` over 80ms.
3. **Audio Feedback (Toggleable):**
   - Soft high-pitch glass tap on order fill.
   - Low-frequency alert tone on risk rejection or kill switch trigger.

---

## 6. Accessibility & Responsiveness
- High color contrast ratio: $\ge 4.5:1$ between text and background on all interactive components.
- Keyboard navigation: Full shortcut support (`Space` to pause log stream, `Esc` to dismiss modals, `Ctrl+K` for global command palette).
- Breakpoints:
  - Ultra-Wide ($>1920\text{px}$): 4-column multi-monitor layout.
  - Standard Desktop ($1280\text{px} - 1919\text{px}$): 3-column default layout.
  - Laptop / Compact ($1024\text{px} - 1279\text{px}$): 2-column layout with tabbed bottom dock.
