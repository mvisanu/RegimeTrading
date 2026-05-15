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
