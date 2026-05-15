# Portfolio Page — Design Spec

**Date:** 2026-05-14  
**Status:** Approved

## Overview

A new Streamlit page (`pages/0_Portfolio.py`) that shows the user's Alpaca account holdings with P&L, a performance equity curve, and per-position close buttons. Appears immediately below the main page in the sidebar via the `0_` filename prefix.

---

## Architecture

### New file
- `pages/0_Portfolio.py` — self-contained page, Bloomberg terminal aesthetic

### Modified file
- `core/broker.py` — add `get_portfolio_history(period, timeframe) → dict` public method

### No changes to
- `app.py`, `core/design_system.py`, any other page, or test files

---

## Sections

### 1. Account Summary Cards

Four `metric_card` components from `core.design_system` in a single row:

| Card | Alpaca field |
|------|-------------|
| Equity | `account.equity` |
| Buying Power | `account.buying_power` |
| Cash | `account.cash` |
| Day P&L | `account.equity - account.last_equity` ($ and %) |

Day P&L card border color: green if positive, red if negative.

### 2. Performance Chart

A Plotly line chart of portfolio equity over time, rendered between the account cards and the holdings table.

**Period selector:** `1D | 1W | 1M | 3M | 1Y` — implemented as `st.radio` horizontal buttons. Default: `1M`.

**Data source:** `AlpacaBroker.get_portfolio_history(period, timeframe)` → wraps `TradingClient.get_portfolio_history()`. Returns `{"timestamps": [...], "equity": [...]}`.

Timeframe mapping:

| Period | Alpaca period | Alpaca timeframe |
|--------|--------------|-----------------|
| 1D | 1D | 5Min |
| 1W | 1W | 1H |
| 1M | 1M | 1D |
| 3M | 3M | 1D |
| 1Y | 1A | 1D |

**Chart style:**
- `#00d4ff` cyan line, width 2
- Filled area below line: `rgba(0, 212, 255, 0.12)`
- `get_plotly_layout()` dark theme base
- Headline above chart: current equity + change from period start (green/red colored)
- X-axis: timestamps, Y-axis: USD equity

### 3. Holdings Table

Columns (left to right):

| Column | Alpaca field | Format |
|--------|-------------|--------|
| Symbol | `symbol` | bold, cyan |
| Side | `side` | LONG / SHORT badge |
| Qty | `qty` | integer |
| Avg Entry | `avg_entry_price` | `$X.XX` |
| Price | `current_price` | `$X.XX` |
| Day % | `change_today` × 100 | `+X.XX%` green/red |
| Market Value | `market_value` | `$X,XXX.XX` |
| Cost Basis | `cost_basis` | `$X,XXX.XX` |
| P&L $ | `unrealized_pl` | `+$X.XX` green/red |
| P&L % | `unrealized_plpc` × 100 | `+X.XX%` green/red |
| Day P&L $ | `unrealized_intraday_pl` | `+$X.XX` green/red |
| Day P&L % | `unrealized_intraday_plpc` × 100 | `+X.XX%` green/red |
| Action | — | Close button |

**Close button behavior:**
- Calls `AlpacaBroker().submit_order(symbol=sym, qty=float(qty), side="sell")`
- On success: replace button with green "Closed ✓" and `st.rerun()`
- On error: show `st.error(...)` inline
- Closed symbols tracked in `st.session_state.closed`

**Empty state:** If no positions, show `st.info("No open positions — account is flat.")`.

**Broker error state:** If `AlpacaBroker()` raises, show `st.warning(...)` and skip table/chart.

### 4. Refresh

A `🔄 Refresh` button in the top-right clears `@st.cache_data` for positions and history, then calls `st.rerun()`. Positions and history fetches are cached with `ttl=30` seconds.

---

## Design Language

Bloomberg terminal aesthetic — consistent with `app.py` and Dashboard 1:

| Token | Value |
|-------|-------|
| Background | `#0e1117` |
| Accent | `#00d4ff` (cyan) |
| Font | monospace / Inter |
| Positive P&L | `#10b981` (emerald) |
| Negative P&L | `#ef4444` (red) |

Uses `core.design_system.metric_card`, `section_header`, `get_plotly_layout`, `REGIME_COLORS` (for side badges if needed).

---

## Error Handling

- Broker unavailable → warning banner, no crash
- Portfolio history unavailable → skip chart, show info message
- Individual position field missing → default to `0.0` / `"—"`

---

## Non-Negotiables

- Paper trading default: `LIVE_TRADING=false`; sell orders route through all 5 safety circuit breakers
- No CSS leaking to other pages (scoped `<style>` block)
- No redefinition of `REGIME_COLORS`
