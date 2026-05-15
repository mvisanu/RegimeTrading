# Portfolio Page Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `pages/0_Portfolio.py` — a Bloomberg-terminal-style page showing Alpaca holdings with all P&L columns, an equity-curve chart, and a per-position Close button.

**Architecture:** Expose `get_portfolio_history()` on `AlpacaBroker`, then build a self-contained Streamlit page that calls the three broker read methods (`get_account`, `get_positions`, `get_portfolio_history`) via `@st.cache_data(ttl=30)` wrappers. All sell orders route through the existing safety circuit breakers unchanged.

**Tech Stack:** Python 3.11+, Streamlit, Plotly, alpaca-py, core.broker.AlpacaBroker, core.design_system

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `core/broker.py` | Add `get_portfolio_history(period, timeframe) → dict` public method |
| Create | `tests/test_broker_portfolio.py` | Tests for the new broker method |
| Create | `pages/0_Portfolio.py` | Full portfolio page (account cards, chart, holdings table) |

---

## Task 1: Add `get_portfolio_history()` to `AlpacaBroker`

**Files:**
- Modify: `core/broker.py` (after the `get_clock` method, around line 171)
- Create: `tests/test_broker_portfolio.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_broker_portfolio.py`:

```python
"""Tests for AlpacaBroker.get_portfolio_history()."""
import pytest
from unittest.mock import MagicMock, patch


def _make_broker():
    """Return an AlpacaBroker with a fully mocked Alpaca client."""
    from core.broker import AlpacaBroker
    with patch("core.broker.TradingClient"):
        broker = AlpacaBroker.__new__(AlpacaBroker)
        broker._live = False
    broker._client = MagicMock()
    return broker


def test_get_portfolio_history_returns_timestamps_and_equity():
    broker = _make_broker()
    fake_history = MagicMock()
    fake_history.timestamp = [1700000000, 1700086400, 1700172800]
    fake_history.equity = [10000.0, 10200.0, 10150.0]
    broker._client.get_portfolio_history.return_value = fake_history

    result = broker.get_portfolio_history("1M", "1D")

    assert result["timestamps"] == [1700000000, 1700086400, 1700172800]
    assert result["equity"] == [10000.0, 10200.0, 10150.0]


def test_get_portfolio_history_filters_none_equity():
    broker = _make_broker()
    fake_history = MagicMock()
    fake_history.timestamp = [1700000000, 1700086400]
    fake_history.equity = [10000.0, None]
    broker._client.get_portfolio_history.return_value = fake_history

    result = broker.get_portfolio_history("1W", "1H")

    assert result["equity"] == [10000.0, None]  # None preserved, caller filters


def test_get_portfolio_history_passes_period_and_timeframe():
    broker = _make_broker()
    fake_history = MagicMock()
    fake_history.timestamp = []
    fake_history.equity = []
    broker._client.get_portfolio_history.return_value = fake_history

    broker.get_portfolio_history("3M", "1D")

    broker._client.get_portfolio_history.assert_called_once_with(period="3M", timeframe="1D")


def test_get_portfolio_history_raises_runtime_error_on_api_failure():
    broker = _make_broker()
    broker._client.get_portfolio_history.side_effect = Exception("network error")

    with pytest.raises(RuntimeError, match="get_portfolio_history failed"):
        broker.get_portfolio_history("1M", "1D")
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_broker_portfolio.py -v
```

Expected: 4 FAILs — `AttributeError: 'AlpacaBroker' object has no attribute 'get_portfolio_history'`

- [ ] **Step 3: Add `get_portfolio_history` to `core/broker.py`**

Insert after the `get_clock` method (after line 171, before the `# Order submission` comment block):

```python
def get_portfolio_history(self, period: str = "1M", timeframe: str = "1D") -> dict:
    """Return portfolio equity history for charting.

    Args:
        period: Alpaca period string — "1D", "1W", "1M", "3M", "1A".
        timeframe: Alpaca timeframe string — "5Min", "1H", "1D".

    Returns:
        Dict with keys ``timestamps`` (list[int] of Unix epoch seconds) and
        ``equity`` (list[float|None]).

    Raises:
        RuntimeError: On API errors.
    """
    try:
        history = self._client.get_portfolio_history(period=period, timeframe=timeframe)
        return {
            "timestamps": list(history.timestamp),
            "equity": [float(v) if v is not None else None for v in history.equity],
        }
    except Exception as exc:
        raise RuntimeError(f"get_portfolio_history failed: {exc}") from exc
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_broker_portfolio.py -v
```

Expected: 4 PASSes

- [ ] **Step 5: Run full gate tests to confirm nothing broke**

```
pytest tests/test_no_lookahead.py tests/test_safety.py -v
```

Expected: all PASS

- [ ] **Step 6: Commit**

```
git add core/broker.py tests/test_broker_portfolio.py
git commit -m "feat: add get_portfolio_history() to AlpacaBroker"
```

---

## Task 2: Create `pages/0_Portfolio.py` — account cards and page skeleton

**Files:**
- Create: `pages/0_Portfolio.py`

- [ ] **Step 1: Create the page file with imports, config, helpers, and account cards**

Create `pages/0_Portfolio.py`:

```python
"""Portfolio — live Alpaca holdings with P&L and equity curve."""
from __future__ import annotations

import datetime

import plotly.graph_objects as go
import streamlit as st

from core.broker import AlpacaBroker
from core.design_system import ACCENT_CYAN, get_plotly_layout, metric_card, section_header

st.set_page_config(page_title="Portfolio", page_icon="💼", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── session state ─────────────────────────────────────────────────────────────
if "pf_closed" not in st.session_state:
    st.session_state.pf_closed: set[str] = set()

# ── cached data fetchers ──────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def _fetch_account() -> tuple[dict, str]:
    try:
        return AlpacaBroker().get_account(), ""
    except Exception as exc:
        return {}, str(exc)


@st.cache_data(ttl=30)
def _fetch_positions() -> tuple[list[dict], str]:
    try:
        return AlpacaBroker().get_positions(), ""
    except Exception as exc:
        return [], str(exc)


@st.cache_data(ttl=30)
def _fetch_history(period: str, timeframe: str) -> tuple[dict, str]:
    try:
        return AlpacaBroker().get_portfolio_history(period, timeframe), ""
    except Exception as exc:
        return {}, str(exc)


# ── formatting helpers ────────────────────────────────────────────────────────
def _pnl_color(value: float) -> str:
    return "#10b981" if value >= 0 else "#ef4444"


def _fmt_dollar(value: float, sign: bool = False) -> str:
    prefix = ("+" if value >= 0 else "") if sign else ""
    return f"{prefix}${value:,.2f}"


def _fmt_pct(value: float, sign: bool = False) -> str:
    prefix = ("+" if value >= 0 else "") if sign else ""
    return f"{prefix}{value:.2f}%"


def _safe_float(d: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(d.get(key) or default)
    except (TypeError, ValueError):
        return default


# ── page header ───────────────────────────────────────────────────────────────
col_title, col_refresh = st.columns([6, 1])
with col_title:
    st.title("Portfolio")
with col_refresh:
    st.markdown("<div style='margin-top:14px'>", unsafe_allow_html=True)
    if st.button("🔄 Refresh", use_container_width=True):
        _fetch_account.clear()
        _fetch_positions.clear()
        _fetch_history.clear()
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

st.divider()

# ── account cards ─────────────────────────────────────────────────────────────
account, acct_err = _fetch_account()

if acct_err:
    st.warning(f"Broker unavailable: {acct_err}")
else:
    equity = _safe_float(account, "equity")
    last_equity = _safe_float(account, "last_equity", equity)
    buying_power = _safe_float(account, "buying_power")
    cash = _safe_float(account, "cash")
    day_pnl = equity - last_equity
    day_pnl_pct = (day_pnl / last_equity * 100) if last_equity else 0.0
    day_color = _pnl_color(day_pnl)
    day_label = f"{_fmt_dollar(day_pnl, sign=True)}  ({_fmt_pct(day_pnl_pct, sign=True)})"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Equity", f"${equity:,.2f}"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("Buying Power", f"${buying_power:,.2f}"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card("Cash", f"${cash:,.2f}"), unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card("Day P&L", day_label, color=day_color), unsafe_allow_html=True)
```

- [ ] **Step 2: Verify the page loads with account cards**

```
streamlit run app.py
```

Navigate to "0 Portfolio" in the sidebar. Expected: page loads, shows 4 account cards (or a broker warning if credentials aren't set). No crash.

- [ ] **Step 3: Commit**

```
git add pages/0_Portfolio.py
git commit -m "feat: portfolio page skeleton with account cards"
```

---

## Task 3: Add performance chart to `pages/0_Portfolio.py`

**Files:**
- Modify: `pages/0_Portfolio.py` (append after the account cards block)

- [ ] **Step 1: Append the chart section to `pages/0_Portfolio.py`**

Add this block after the account cards section (after the `with c4:` block):

```python

st.divider()

# ── performance chart ─────────────────────────────────────────────────────────
section_header("Portfolio Equity")

_PERIOD_MAP: dict[str, tuple[str, str]] = {
    "1D": ("1D", "5Min"),
    "1W": ("1W", "1H"),
    "1M": ("1M", "1D"),
    "3M": ("3M", "1D"),
    "1Y": ("1A", "1D"),
}

selected_period: str = st.radio(
    "Period",
    list(_PERIOD_MAP.keys()),
    index=2,
    horizontal=True,
    label_visibility="collapsed",
)

period_str, timeframe_str = _PERIOD_MAP[selected_period]
hist, hist_err = _fetch_history(period_str, timeframe_str)

if hist_err:
    st.info(f"Portfolio history unavailable: {hist_err}")
elif hist and hist.get("timestamps") and hist.get("equity"):
    ts = [datetime.datetime.fromtimestamp(t) for t in hist["timestamps"]]
    eq = hist["equity"]

    valid_eq = [e for e in eq if e is not None]
    if valid_eq:
        start_eq, end_eq = valid_eq[0], valid_eq[-1]
        change = end_eq - start_eq
        change_pct = (change / start_eq * 100) if start_eq else 0.0
        hl_color = _pnl_color(change)
        st.markdown(
            f"<span style='font-size:1.3rem;font-weight:700;color:#f8fafc'>"
            f"${end_eq:,.2f}</span>"
            f"<span style='font-size:1rem;color:{hl_color};margin-left:10px'>"
            f"{_fmt_dollar(change, sign=True)} ({_fmt_pct(change_pct, sign=True)})</span>",
            unsafe_allow_html=True,
        )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=ts,
        y=eq,
        mode="lines",
        line=dict(color=ACCENT_CYAN, width=2),
        fill="tozeroy",
        fillcolor="rgba(0, 212, 255, 0.12)",
        name="Equity",
        hovertemplate="$%{y:,.2f}<extra></extra>",
    ))
    layout = get_plotly_layout()
    layout.update({"height": 280, "showlegend": False,
                   "margin": {"l": 48, "r": 24, "t": 20, "b": 40}})
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)
```

- [ ] **Step 2: Verify chart renders**

```
streamlit run app.py
```

Navigate to Portfolio. Select each period button (1D, 1W, 1M, 3M, 1Y). Expected: cyan equity curve with filled area below, headline equity + change. If broker unavailable, shows info message — no crash.

- [ ] **Step 3: Commit**

```
git add pages/0_Portfolio.py
git commit -m "feat: add equity curve chart to portfolio page"
```

---

## Task 4: Add holdings table with Close buttons to `pages/0_Portfolio.py`

**Files:**
- Modify: `pages/0_Portfolio.py` (append after the chart block)

- [ ] **Step 1: Append the holdings table to `pages/0_Portfolio.py`**

Add this block after the chart section (after the `st.plotly_chart` block):

```python

st.divider()

# ── holdings table ────────────────────────────────────────────────────────────
section_header("Holdings")

positions, pos_err = _fetch_positions()

if pos_err:
    st.warning(f"Could not load positions: {pos_err}")
elif not positions:
    st.info("No open positions — account is flat.")
else:
    _COL_W = [0.7, 0.5, 0.5, 0.8, 0.8, 0.65, 0.9, 0.9, 0.75, 0.65, 0.75, 0.65, 0.7]
    _HEADERS = [
        "Symbol", "Side", "Qty", "Avg Entry", "Price", "Day %",
        "Mkt Val", "Cost", "P&L $", "P&L %", "Day P&L $", "Day P&L %", "",
    ]

    header_cols = st.columns(_COL_W)
    for col, label in zip(header_cols, _HEADERS):
        col.markdown(
            f"<span style='color:#94a3b8;font-size:0.72rem;font-weight:600;"
            f"text-transform:uppercase;letter-spacing:0.05em'>{label}</span>",
            unsafe_allow_html=True,
        )
    st.markdown("<hr style='margin:2px 0 6px;border-color:#1e2535'>", unsafe_allow_html=True)

    for pos in positions:
        sym = pos.get("symbol", "?")
        side_raw = str(pos.get("side", "long"))
        side = side_raw.upper()
        qty = _safe_float(pos, "qty")
        avg_entry = _safe_float(pos, "avg_entry_price")
        current = _safe_float(pos, "current_price")
        change_today = _safe_float(pos, "change_today") * 100
        market_val = _safe_float(pos, "market_value")
        cost = _safe_float(pos, "cost_basis")
        upl = _safe_float(pos, "unrealized_pl")
        uplpc = _safe_float(pos, "unrealized_plpc") * 100
        day_pl = _safe_float(pos, "unrealized_intraday_pl")
        day_plpc = _safe_float(pos, "unrealized_intraday_plpc") * 100

        side_color = "#10b981" if side == "LONG" else "#94a3b8"
        row = st.columns(_COL_W)

        row[0].markdown(
            f"<b style='color:{ACCENT_CYAN};font-family:monospace'>{sym}</b>",
            unsafe_allow_html=True,
        )
        row[1].markdown(
            f"<span style='color:{side_color}'>{side}</span>",
            unsafe_allow_html=True,
        )
        row[2].write(f"{int(qty)}")
        row[3].write(f"${avg_entry:.2f}")
        row[4].write(f"${current:.2f}")
        row[5].markdown(
            f"<span style='color:{_pnl_color(change_today)}'>"
            f"{_fmt_pct(change_today, sign=True)}</span>",
            unsafe_allow_html=True,
        )
        row[6].write(f"${market_val:,.2f}")
        row[7].write(f"${cost:,.2f}")
        row[8].markdown(
            f"<span style='color:{_pnl_color(upl)}'>{_fmt_dollar(upl, sign=True)}</span>",
            unsafe_allow_html=True,
        )
        row[9].markdown(
            f"<span style='color:{_pnl_color(uplpc)}'>{_fmt_pct(uplpc, sign=True)}</span>",
            unsafe_allow_html=True,
        )
        row[10].markdown(
            f"<span style='color:{_pnl_color(day_pl)}'>{_fmt_dollar(day_pl, sign=True)}</span>",
            unsafe_allow_html=True,
        )
        row[11].markdown(
            f"<span style='color:{_pnl_color(day_plpc)}'>{_fmt_pct(day_plpc, sign=True)}</span>",
            unsafe_allow_html=True,
        )

        with row[12]:
            if sym in st.session_state.pf_closed:
                st.success("Closed ✓")
            else:
                if st.button("Close", key=f"pf_close_{sym}"):
                    try:
                        AlpacaBroker().submit_order(
                            symbol=sym, qty=qty, side="sell"
                        )
                        st.session_state.pf_closed.add(sym)
                        st.rerun()
                    except (RuntimeError, ValueError) as exc:
                        st.error(f"Rejected: {exc}")

    st.markdown("<hr style='margin:8px 0 4px;border-color:#1e2535'>", unsafe_allow_html=True)
    st.caption(
        f"All close orders route through 5 safety circuit breakers. "
        f"Paper trading active (LIVE_TRADING={__import__('os').getenv('LIVE_TRADING', 'false')})."
    )
```

- [ ] **Step 2: Verify holdings table**

```
streamlit run app.py
```

Navigate to Portfolio. Expected:
- If positions exist: table with all 12 data columns + Close button. P&L columns show green (positive) or red (negative) colored text. Close button places a sell order and replaces itself with "Closed ✓".
- If no positions: "No open positions — account is flat." info message.

- [ ] **Step 3: Run full test suite**

```
pytest
```

Expected: all 349+ tests pass (new 4 broker tests + existing 349).

- [ ] **Step 4: Final commit**

```
git add pages/0_Portfolio.py
git commit -m "feat: add holdings table with close buttons to portfolio page"
```

---

## Spec Coverage Check

| Spec requirement | Task |
|-----------------|------|
| `pages/0_Portfolio.py` sidebar position via `0_` prefix | Task 2 |
| Bloomberg terminal aesthetic (`#0e1117`, cyan, monospace) | Task 2 |
| Equity, Buying Power, Cash, Day P&L cards | Task 2 |
| Refresh button clears cache + reruns | Task 2 |
| Broker unavailable → warning banner, no crash | Task 2 |
| Performance chart with period selector 1D/1W/1M/3M/1Y | Task 3 |
| Cyan line + filled area, dark theme via `get_plotly_layout()` | Task 3 |
| Equity headline showing change for period | Task 3 |
| `get_portfolio_history()` on broker | Task 1 |
| All 12 data columns (symbol through day P&L%) | Task 4 |
| P&L green/red coloring | Task 4 |
| Close button → `submit_order(..., side="sell")` | Task 4 |
| Safety circuit breakers fire on close | Task 4 (inherits from broker) |
| Empty state info message | Task 4 |
| No CSS leaking to other pages (scoped style block) | Task 2 |
| No redefinition of `REGIME_COLORS` | Task 2–4 |
