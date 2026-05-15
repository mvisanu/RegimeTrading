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
