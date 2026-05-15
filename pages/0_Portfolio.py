"""Portfolio — live Alpaca holdings with P&L and equity curve."""
from __future__ import annotations

import datetime
import os

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
    if value < 0:
        return f"-${abs(value):,.2f}"
    prefix = "+" if sign else ""
    return f"{prefix}${value:,.2f}"


def _fmt_pct(value: float, sign: bool = False) -> str:
    prefix = ("+" if value >= 0 else "") if sign else ""
    return f"{prefix}{value:.2f}%"


def _safe_float(d: dict, key: str, default: float = 0.0) -> float:
    val = d.get(key)
    if val is None:
        return default
    try:
        return float(val)
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
    if not valid_eq:
        st.info("No equity data available for this period.")
    else:
        start_eq, end_eq = valid_eq[0], valid_eq[-1]
        change = end_eq - start_eq
        change_pct = (change / start_eq * 100) if start_eq != 0 else 0.0
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
        side_raw = pos.get("side", "long")
        side = (side_raw.value if hasattr(side_raw, "value") else str(side_raw)).upper()
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
        qty_display = f"{qty:.4f}".rstrip("0").rstrip(".") if qty != int(qty) else str(int(qty))
        row[2].write(qty_display)
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
                        # live_confirmed omitted — paper trading only; live requires explicit flag
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
        f"Paper trading active (LIVE_TRADING={os.getenv('LIVE_TRADING', 'false')})."
    )
