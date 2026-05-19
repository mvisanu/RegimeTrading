"""RegimeTrading — Streamlit multi-page hub with Swing Scanner."""
from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from core.broker import AlpacaBroker
from core.design_system import REGIME_COLORS
from swing.scanner import scan, ScanResult

st.set_page_config(
    page_title="Regime Trading",
    page_icon="📈",
    layout="wide",
)

if "scan_results" not in st.session_state:
    st.session_state.scan_results: list[ScanResult] = []
if "ordered" not in st.session_state:
    st.session_state.ordered: set[str] = set()
if "account_equity" not in st.session_state:
    st.session_state.account_equity: float = 0.0
if "broker_error" not in st.session_state:
    st.session_state.broker_error: str = ""

st.title("Regime Trading")
st.subheader("Adaptive algorithmic trading driven by market-regime detection")
st.caption(
    "Use the sidebar to navigate between dashboards. "
    "Run the Swing Scanner below to find top buy candidates from your watchlist."
)

st.divider()

st.header("Swing Scanner")

WATCHLIST_PATH = Path(__file__).parent / "swing" / "watchlist.json"


def _fetch_equity() -> tuple[float, str]:
    try:
        broker = AlpacaBroker()
        account = broker.get_account()
        return float(account.get("equity", 10_000.0)), ""
    except Exception as exc:
        return 10_000.0, str(exc)


col_btn, col_equity = st.columns([2, 3])
with col_btn:
    run_scan = st.button("🔍 Run Scan", type="primary", use_container_width=True)
with col_equity:
    if st.session_state.account_equity:
        st.metric("Account Equity", f"${st.session_state.account_equity:,.2f}")
    elif st.session_state.broker_error:
        st.warning(f"Broker unavailable — using $10,000 placeholder. {st.session_state.broker_error}")

if run_scan:
    equity, err = _fetch_equity()
    st.session_state.account_equity = equity
    st.session_state.broker_error = err
    st.session_state.ordered = set()

    with st.spinner("Running regime detection across watchlist symbols…"):
        try:
            results = scan(
                watchlist_path=WATCHLIST_PATH,
                account_equity=equity,
                top_n=10,
            )
            st.session_state.scan_results = results
        except Exception as exc:
            st.error(f"Scanner failed: {exc}")
            st.session_state.scan_results = []

if st.session_state.scan_results:
    st.subheader("Top 10 Buy Candidates")

    results: list[ScanResult] = st.session_state.scan_results

    header_cols = st.columns([0.4, 1.2, 1.4, 0.8, 0.8, 1.0, 0.9, 0.9, 1.2])
    for col, label in zip(header_cols, ["#", "Symbol", "Regime", "Score", "Shares", "Est. Cost", "Stop", "TP1", "Action"]):
        col.markdown(f"**{label}**")

    st.markdown("<hr style='margin:4px 0 8px'>", unsafe_allow_html=True)

    for i, r in enumerate(results, start=1):
        regime_color = REGIME_COLORS.get(r.regime, "#94a3b8")
        cols = st.columns([0.4, 1.2, 1.4, 0.8, 0.8, 1.0, 0.9, 0.9, 1.2])
        cols[0].write(f"**{i}**")
        cols[1].markdown(f"<span style='color:{regime_color};font-weight:700'>{r.symbol}</span>", unsafe_allow_html=True)
        cols[2].markdown(f"<span style='color:{regime_color}'>{r.regime}</span>", unsafe_allow_html=True)
        cols[3].write(f"{r.final_score:.3f}")
        cols[4].write(str(r.shares))
        cols[5].write(f"${r.estimated_cost:,.2f}")
        cols[6].write(f"${r.stop:.2f}")
        cols[7].write(f"${r.tp1:.2f}" if r.tp1 else "—")

        with cols[8]:
            if r.symbol in st.session_state.ordered:
                st.success("Ordered ✓", icon="✅")
            else:
                if st.button(f"Buy {r.symbol}", key=f"buy_{r.symbol}_{i}"):
                    try:
                        broker = AlpacaBroker()
                        broker.submit_order(symbol=r.symbol, qty=float(r.shares), side="buy")
                        st.session_state.ordered.add(r.symbol)
                        st.success(f"Order placed: {r.shares} × {r.symbol} @ ~${r.entry:.2f}")
                        st.rerun()
                    except (RuntimeError, ValueError) as exc:
                        st.error(f"Order rejected: {exc}")

    st.markdown("<hr style='margin:8px 0 4px'>", unsafe_allow_html=True)
    st.caption(
        f"Positions sized at 1% account risk per trade. "
        f"All orders route through 5 safety circuit breakers. "
        f"Paper trading active (LIVE_TRADING={os.getenv('LIVE_TRADING', 'false')})."
    )

elif not run_scan:
    st.info("Click **Run Scan** to score your watchlist symbols against the current market regime.")

st.divider()
st.markdown(
    """
    **Dashboards** — use the sidebar to navigate:
    | # | Dashboard | Purpose |
    |---|-----------|---------|
    | 1 | Regime Detection | Live HMM regime timeline |
    | 2 | Monte Carlo | Future return probability fan |
    | 3 | Sensitivity | SMA parameter sweep |
    | 4 | Portfolio Risk | Stress tests & drawdown |
    | 5 | Multi-Asset Backtest | HMM strategy vs buy-and-hold |
    | 6 | Sentiment | VADER news sentiment |
    | 7 | Correlation Breaks | Z-score break alerts |
    """
)
