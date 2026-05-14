"""
pages/5_Multi_Asset_Backtest.py
================================
Dashboard 5 — Multi-Asset Regime Backtester.

Design language: Asset-colored. A single ``--accent`` CSS variable shifts to
the selected asset's color (SPY cyan, BTC-USD orange, GLD gold, TLT purple),
re-tinting card borders, glow shadows, and chart traces across the whole page.

Background:  #0c0c14   Card bg: #141420   Fonts: Outfit + Fira Code
"""

from __future__ import annotations

import html as _html
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dateutil.relativedelta import relativedelta

from core import verify as _verify
from core.backtest import BacktestResult, walk_forward_backtest
from core.data import load_ohlcv
from core.design_system import REGIME_COLORS, get_plotly_layout
from core.hmm_utils import RegimeResult, fit_and_filter

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Multi-Asset Backtester",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Asset color constants
# ---------------------------------------------------------------------------

ASSET_COLORS: dict[str, str] = {
    "SPY":     "#00d4ff",  # cyan
    "BTC-USD": "#f7921a",  # bitcoin orange
    "GLD":     "#ffd700",  # gold
    "TLT":     "#a78bfa",  # soft purple
}
EXTRA_COLORS: list[str] = ["#f87171", "#34d399", "#fb923c", "#818cf8", "#38bdf8"]

DEFAULT_TICKERS: list[str] = ["SPY", "BTC-USD", "GLD", "TLT"]

# ---------------------------------------------------------------------------
# Stress-test periods
# ---------------------------------------------------------------------------

STRESS_PERIODS: dict[str, tuple[str, str]] = {
    "2008 Crisis": ("2008-09-01", "2009-03-31"),
    "2020 COVID":  ("2020-02-01", "2020-04-30"),
    "2022 Bear":   ("2022-01-01", "2022-10-31"),
}

# ---------------------------------------------------------------------------
# Google Fonts injection
# ---------------------------------------------------------------------------

_FONT_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Fira+Code:wght@400;500&display=swap');
"""

# ---------------------------------------------------------------------------
# Helper: resolve color for an arbitrary ticker
# ---------------------------------------------------------------------------


def _ticker_color(ticker: str, index: int = 0) -> str:
    """Return the canonical accent color for *ticker*, falling back to EXTRA_COLORS."""
    return ASSET_COLORS.get(ticker.upper(), EXTRA_COLORS[index % len(EXTRA_COLORS)])


def _hex_alpha_to_rgba(hex6: str, alpha_hex: str) -> str:
    """Convert a 6-digit hex color + 2-digit hex alpha to an rgba() string.

    Plotly does not accept 8-digit hex colors (#rrggbbAA).  This helper
    produces the equivalent ``rgba(r,g,b,a)`` string that Plotly accepts.

    Parameters
    ----------
    hex6:
        6-digit hex color, with or without leading ``#`` (e.g. ``"#f87171"``).
    alpha_hex:
        2-digit hexadecimal alpha value (e.g. ``"0d"`` → 13/255 ≈ 0.051).

    Returns
    -------
    str
        Valid Plotly color string, e.g. ``"rgba(248,113,113,0.051)"``.
    """
    h = hex6.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    a = round(int(alpha_hex, 16) / 255, 4)
    return f"rgba({r},{g},{b},{a})"


# ---------------------------------------------------------------------------
# Cached data-loading and backtest helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def _load_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV data with a 1-hour cache."""
    return load_ohlcv(ticker, start, end)


@st.cache_data(ttl=3600)
def _run_backtest(ticker: str, start: str, end: str) -> BacktestResult:
    """Fit HMM and run walk-forward backtest; result cached for 1 hour."""
    df = _load_data(ticker, start, end)
    return walk_forward_backtest(df, train_years=1, test_months=6)


@st.cache_data(ttl=3600)
def _fit_regime(ticker: str, start: str, end: str) -> RegimeResult:
    """Fit HMM for regime timeline visualization; cached for 1 hour."""
    df = load_ohlcv(ticker, start, end)
    return fit_and_filter(df)


# ---------------------------------------------------------------------------
# Stress-test drawdown helper
# ---------------------------------------------------------------------------


def _compute_stress_drawdown(equity_curve: pd.Series, start: str, end: str) -> float | None:
    """Return max drawdown during [start, end] period.

    Parameters
    ----------
    equity_curve:
        Strategy equity curve indexed by date.
    start:
        ISO date string for the start of the stress window (inclusive).
    end:
        ISO date string for the end of the stress window (inclusive).

    Returns
    -------
    float | None
        Maximum drawdown in the period as a negative fraction (e.g. -0.35).
        Returns None if fewer than 2 bars fall within the range.
    """
    segment = equity_curve.loc[start:end]
    if len(segment) < 2:
        return None
    peak = segment.cummax()
    dd = (segment / peak - 1).min()
    return float(dd)


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------


def _build_equity_chart(
    ticker: str,
    result: BacktestResult,
    accent: str,
) -> go.Figure:
    """Hero equity curve chart for a single asset.

    Strategy line: accent color, 2.5px.
    Buy-and-hold: slate dashed.
    Grid lines at 5% opacity of accent.

    Parameters
    ----------
    ticker:
        Asset symbol (used in trace names).
    result:
        BacktestResult for the asset.
    accent:
        Hex color string for the strategy line.

    Returns
    -------
    go.Figure
        Fully configured Plotly figure.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=result.equity_curve.index,
        y=result.equity_curve.values,
        name=f"{_html.escape(ticker)} Strategy",
        line=dict(color=accent, width=2.5),
        hovertemplate="%{y:,.0f}<extra>Strategy</extra>",
    ))

    fig.add_trace(go.Scatter(
        x=result.benchmark_equity.index,
        y=result.benchmark_equity.values,
        name="Buy & Hold",
        line=dict(color="rgba(148,163,184,0.5)", width=1.5, dash="dash"),
        hovertemplate="%{y:,.0f}<extra>Buy & Hold</extra>",
    ))

    layout = get_plotly_layout(theme="dark")

    # Override backgrounds and grid to asset accent
    layout["paper_bgcolor"] = "#0c0c14"
    layout["plot_bgcolor"] = "#0c0c14"
    layout["xaxis"]["gridcolor"] = _hex_alpha_to_rgba(accent, "0d")  # ~5% opacity
    layout["yaxis"]["gridcolor"] = _hex_alpha_to_rgba(accent, "0d")
    layout["title"] = dict(
        text=f"<b>{_html.escape(ticker)}</b> — Equity Curve",
        font=dict(family="Outfit, sans-serif", size=16, color="#f8fafc"),
        x=0.0,
        xanchor="left",
    )
    layout["margin"] = {"l": 60, "r": 24, "t": 50, "b": 40}

    fig.update_layout(**layout)
    return fig


def build_regime_timeline_chart(
    tickers: list[str],
    regime_results: dict[str, RegimeResult],
    ticker_colors: dict[str, str],
) -> go.Figure:
    """One horizontal bar per asset, colored by stable_labels segments.

    Consecutive bars with the same regime label are collapsed into a single
    wide segment for rendering performance.

    Parameters
    ----------
    tickers:
        Ordered list of ticker symbols to render (top to bottom).
    regime_results:
        Mapping from ticker to RegimeResult from fit_and_filter.
    ticker_colors:
        Mapping from ticker to accent hex color (for y-axis labels).

    Returns
    -------
    go.Figure
        Multi-row Plotly figure with shared x-axis.
    """
    n = len(tickers)
    if n == 0:
        return go.Figure()

    fig = make_subplots(
        rows=n,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=[1.0] * n,
    )

    for row_idx, ticker in enumerate(tickers, start=1):
        rr = regime_results.get(ticker)
        if rr is None:
            continue

        labels = rr.stable_labels
        # We need dates — use equity_curve dates if available, else skip
        # Regime labels come from the full fit; they may have fewer rows
        # than the original data due to the 20-bar rolling window dropna.
        # We'll fabricate an index aligned to the label length.
        # In practice the regime timeline is illustrative (full-period fit).
        n_bars = len(labels)
        if n_bars == 0:
            continue

        # Build segments: (regime, start_idx, end_idx)
        segments: list[tuple[str, int, int]] = []
        current_regime = labels[0]
        seg_start = 0
        for i in range(1, n_bars):
            if labels[i] != current_regime:
                segments.append((current_regime, seg_start, i - 1))
                current_regime = labels[i]
                seg_start = i
        segments.append((current_regime, seg_start, n_bars - 1))

        for regime, s, e in segments:
            color = REGIME_COLORS.get(regime, REGIME_COLORS["Uncertain"])
            fig.add_trace(
                go.Bar(
                    x=[e - s + 1],
                    y=[_html.escape(ticker)],
                    orientation="h",
                    marker_color=color,
                    showlegend=(row_idx == 1 and s == segments[0][1]),
                    name=regime,
                    hovertemplate=(
                        f"<b>{_html.escape(ticker)}</b><br>"
                        f"Regime: {_html.escape(regime)}<br>"
                        f"Bars: {e - s + 1}<extra></extra>"
                    ),
                ),
                row=row_idx,
                col=1,
            )

    base_layout = get_plotly_layout(theme="dark")
    base_layout["paper_bgcolor"] = "#0c0c14"
    base_layout["plot_bgcolor"] = "#0c0c14"
    base_layout["barmode"] = "stack"
    base_layout["title"] = dict(
        text="<b>Regime Timeline</b> — all assets",
        font=dict(family="Outfit, sans-serif", size=15, color="#f8fafc"),
        x=0.0,
        xanchor="left",
    )
    base_layout["margin"] = {"l": 80, "r": 24, "t": 50, "b": 40}
    base_layout["showlegend"] = True
    base_layout["height"] = max(200, n * 60 + 80)
    fig.update_layout(**base_layout)

    return fig


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def _render_comparison_table(
    tickers: list[str],
    results: dict[str, BacktestResult],
    ticker_colors: dict[str, str],
) -> None:
    """Render a styled HTML comparison table into the Streamlit page.

    Columns: Asset | Total Return | Sharpe | Max Drawdown | Sharpe vs B&H | Best Period.
    Sharpe improvement is green + ↑ if positive, red + ↓ if negative.
    Best-ranked asset row gets a subtle glow border in its accent color.
    """
    if not results:
        st.info("No results to display.")
        return

    # Find best asset by Sharpe improvement
    sharpe_improvements: dict[str, float] = {}
    for t, r in results.items():
        sharpe_improvements[t] = r.sharpe - r.benchmark_sharpe

    best_ticker = max(sharpe_improvements, key=lambda t: sharpe_improvements[t])

    # Find best stress period per asset
    def _best_period(ticker: str, result: BacktestResult) -> str:
        best_name = "—"
        best_val = -999.0
        for pname, (ps, pe) in STRESS_PERIODS.items():
            strat_dd = _compute_stress_drawdown(result.equity_curve, ps, pe)
            bh_dd = _compute_stress_drawdown(result.benchmark_equity, ps, pe)
            if strat_dd is None or bh_dd is None:
                continue
            # Better if strategy drawdown is less negative than benchmark
            improvement = strat_dd - bh_dd  # positive = less drawdown = better
            if improvement > best_val:
                best_val = improvement
                best_name = pname
        return best_name

    rows_html = ""
    for ticker in tickers:
        result = results.get(ticker)
        if result is None:
            continue

        color = ticker_colors.get(ticker, "#ffffff")
        safe_ticker = _html.escape(ticker)

        total_ret_pct = result.total_return * 100
        sharpe_val = result.sharpe
        mdd_pct = result.max_drawdown * 100
        si = sharpe_improvements[ticker]
        si_color = "#22c55e" if si >= 0 else "#ef4444"
        si_arrow = "↑" if si >= 0 else "↓"
        best_p = _best_period(ticker, result)

        is_best = ticker == best_ticker
        row_style = (
            f"box-shadow: 0 0 12px {color}30; background: {color}0a;"
            if is_best else ""
        )

        rows_html += f"""
        <tr style="border-bottom: 1px solid rgba(255,255,255,0.05); {row_style}">
          <td style="padding:10px 16px; color:{color}; font-family:'Outfit',sans-serif;
                     font-weight:700; font-size:0.95rem;">
            {safe_ticker}{'&nbsp;<span style="font-size:0.7rem;opacity:0.7">BEST</span>' if is_best else ''}
          </td>
          <td style="padding:10px 16px; color:#e2e8f0; font-family:'Fira Code',monospace;
                     text-align:right;">
            {total_ret_pct:+.1f}%
          </td>
          <td style="padding:10px 16px; color:#e2e8f0; font-family:'Fira Code',monospace;
                     text-align:right;">
            {sharpe_val:.2f}
          </td>
          <td style="padding:10px 16px; color:#e2e8f0; font-family:'Fira Code',monospace;
                     text-align:right;">
            {mdd_pct:.1f}%
          </td>
          <td style="padding:10px 16px; color:{si_color}; font-family:'Fira Code',monospace;
                     text-align:right; font-weight:600;">
            {si_arrow} {abs(si):.2f}
          </td>
          <td style="padding:10px 16px; color:#94a3b8; font-family:'Outfit',sans-serif;
                     font-size:0.88rem;">
            {_html.escape(best_p)}
          </td>
        </tr>
        """

    table_html = f"""
    <div style="overflow-x:auto; margin:1rem 0;">
      <table style="width:100%; border-collapse:collapse;
                    background:#141420; border-radius:12px; overflow:hidden;">
        <thead>
          <tr style="background:rgba(255,255,255,0.03);">
            <th style="padding:12px 16px; text-align:left; color:#64748b;
                       font-family:'Outfit',sans-serif; font-size:0.78rem;
                       text-transform:uppercase; letter-spacing:0.08em;
                       font-weight:600;">Asset</th>
            <th style="padding:12px 16px; text-align:right; color:#64748b;
                       font-family:'Outfit',sans-serif; font-size:0.78rem;
                       text-transform:uppercase; letter-spacing:0.08em;
                       font-weight:600;">Total Return</th>
            <th style="padding:12px 16px; text-align:right; color:#64748b;
                       font-family:'Outfit',sans-serif; font-size:0.78rem;
                       text-transform:uppercase; letter-spacing:0.08em;
                       font-weight:600;">Sharpe</th>
            <th style="padding:12px 16px; text-align:right; color:#64748b;
                       font-family:'Outfit',sans-serif; font-size:0.78rem;
                       text-transform:uppercase; letter-spacing:0.08em;
                       font-weight:600;">Max DD</th>
            <th style="padding:12px 16px; text-align:right; color:#64748b;
                       font-family:'Outfit',sans-serif; font-size:0.78rem;
                       text-transform:uppercase; letter-spacing:0.08em;
                       font-weight:600;">Sharpe vs B&amp;H</th>
            <th style="padding:12px 16px; text-align:left; color:#64748b;
                       font-family:'Outfit',sans-serif; font-size:0.78rem;
                       text-transform:uppercase; letter-spacing:0.08em;
                       font-weight:600;">Best Period</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    """
    st.markdown(table_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Stress-test section
# ---------------------------------------------------------------------------


def _render_stress_section(
    tickers: list[str],
    results: dict[str, BacktestResult],
    ticker_colors: dict[str, str],
) -> None:
    """Render the three-column stress-test section.

    Each crisis period gets a column with mini horizontal bars comparing
    strategy drawdown (asset color) vs buy-and-hold drawdown (gray).
    """
    cols = st.columns(3)
    for col, (period_name, (ps, pe)) in zip(cols, STRESS_PERIODS.items()):
        with col:
            safe_period = _html.escape(period_name)
            st.markdown(
                f'<div style="color:#94a3b8; font-family:Outfit,sans-serif; '
                f'font-size:0.78rem; text-transform:uppercase; letter-spacing:0.08em; '
                f'font-weight:600; margin-bottom:8px;">{safe_period}</div>',
                unsafe_allow_html=True,
            )
            for ticker in tickers:
                result = results.get(ticker)
                if result is None:
                    continue
                color = ticker_colors.get(ticker, "#ffffff")
                safe_ticker = _html.escape(ticker)

                strat_dd = _compute_stress_drawdown(result.equity_curve, ps, pe)
                bh_dd = _compute_stress_drawdown(result.benchmark_equity, ps, pe)

                no_data = strat_dd is None or bh_dd is None

                # Convert drawdowns to display percentages (already negative or zero)
                strat_pct = abs(strat_dd) * 100 if strat_dd is not None else 0.0
                bh_pct = abs(bh_dd) * 100 if bh_dd is not None else 0.0

                # Bar widths: scale to max 100% of container (relative to max 50% dd)
                max_scale = max(strat_pct, bh_pct, 1.0)
                strat_w = min(100, strat_pct / max_scale * 100)
                bh_w = min(100, bh_pct / max_scale * 100)
                na_text = '<span style="color:#475569;font-size:0.75rem;">no data in range</span>' if no_data else ""

                st.markdown(
                    f"""
                    <div style="margin-bottom:10px;">
                      <div style="font-family:'Fira Code',monospace; font-size:0.82rem;
                                  color:{color}; margin-bottom:3px;">{safe_ticker}</div>
                      {na_text if no_data else f'''
                      <div style="display:flex; align-items:center; gap:6px; margin-bottom:2px;">
                        <div style="width:{strat_w:.1f}%; background:{color}; height:8px;
                                    border-radius:2px; min-width:2px;"></div>
                        <span style="font-family:'Fira Code',monospace; font-size:0.72rem;
                                     color:{color};">-{strat_pct:.1f}%</span>
                      </div>
                      <div style="display:flex; align-items:center; gap:6px;">
                        <div style="width:{bh_w:.1f}%; background:rgba(148,163,184,0.4);
                                    height:8px; border-radius:2px; min-width:2px;"></div>
                        <span style="font-family:'Fira Code',monospace; font-size:0.72rem;
                                     color:rgba(148,163,184,0.7);">-{bh_pct:.1f}% B&amp;H</span>
                      </div>
                      '''}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ---------------------------------------------------------------------------
# Global CSS injection
# ---------------------------------------------------------------------------


def _inject_global_css(accent: str) -> None:
    """Inject page-level CSS with dynamic accent color variable."""
    st.markdown(
        f"""
        <style>
        {_FONT_CSS}

        /* Page background */
        .stApp, .stApp > header {{
            background-color: #0c0c14 !important;
        }}
        section[data-testid="stSidebar"] {{
            background-color: #0c0c14 !important;
            border-right: 1px solid rgba(255,255,255,0.05);
        }}

        /* CSS variable */
        :root {{
            --accent: {accent};
        }}

        /* Asset cards */
        .asset-card {{
            background: #141420;
            border: 1px solid {accent}20;
            border-radius: 12px;
            padding: 16px 20px;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }}
        .asset-card:hover {{
            border-color: {accent}60;
            box-shadow: 0 0 20px {accent}20;
        }}

        /* Streamlit tab overrides — pill style */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            background: transparent;
            border-bottom: none;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 999px;
            padding: 6px 20px;
            font-family: 'Outfit', sans-serif;
            font-weight: 600;
            font-size: 0.88rem;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            color: #94a3b8;
            transition: all 0.2s ease;
        }}
        .stTabs [aria-selected="true"] {{
            background: {accent}33 !important;
            border-color: {accent} !important;
            color: {accent} !important;
            box-shadow: 0 0 12px {accent}40;
        }}
        .stTabs [data-baseweb="tab-highlight"] {{
            display: none;
        }}
        .stTabs [data-baseweb="tab-border"] {{
            display: none;
        }}

        /* Metric display */
        .regime-metric {{
            font-family: 'Fira Code', monospace;
            font-size: 1.4rem;
            font-weight: 500;
            color: {accent};
        }}

        /* Sidebar inputs */
        .stTextInput input, .stTextArea textarea {{
            background: #141420 !important;
            border: 1px solid rgba(255,255,255,0.1) !important;
            color: #e2e8f0 !important;
            font-family: 'Fira Code', monospace !important;
            border-radius: 6px !important;
        }}
        .stButton > button {{
            background: {accent}22 !important;
            border: 1px solid {accent}66 !important;
            color: {accent} !important;
            font-family: 'Outfit', sans-serif !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            width: 100%;
            transition: all 0.2s ease;
        }}
        .stButton > button:hover {{
            background: {accent}44 !important;
            box-shadow: 0 0 16px {accent}30 !important;
        }}

        /* Global text */
        h1, h2, h3, label, p {{
            font-family: 'Outfit', sans-serif !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Lookahead guard
# ---------------------------------------------------------------------------


def _render_lookahead_badge() -> None:
    """Render verification status badge in sidebar."""
    if _verify.LOOKAHEAD_CHECK_PASSED:
        st.sidebar.markdown(
            '<div style="background:#10b98122; border:1px solid #10b981; '
            'border-radius:6px; padding:6px 12px; color:#10b981; '
            'font-family:Fira Code,monospace; font-size:0.78rem; '
            'text-align:center;">✓ No Lookahead Bias</div>',
            unsafe_allow_html=True,
        )
    else:
        st.error(
            "LOOKAHEAD_CHECK_PASSED is False — forward-filter self-check failed. "
            "Dashboard results are blocked until this is resolved."
        )
        st.stop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for Dashboard 5 — Multi-Asset Regime Backtester."""
    today = date.today().isoformat()
    five_years_ago = (date.today() - relativedelta(years=5)).isoformat()

    # ---- Sidebar --------------------------------------------------------
    st.sidebar.markdown(
        '<h2 style="font-family:Outfit,sans-serif; color:#f8fafc; '
        'font-size:1.1rem; margin-bottom:1rem;">Multi-Asset Backtester</h2>',
        unsafe_allow_html=True,
    )

    _render_lookahead_badge()

    st.sidebar.markdown("---")

    raw_tickers_input = st.sidebar.text_input(
        "Assets (comma-separated)",
        value="SPY, BTC-USD, GLD, TLT",
        help="Enter ticker symbols separated by commas. e.g. SPY, BTC-USD, GLD, TLT",
    )
    run_clicked = st.sidebar.button("Run Backtest", type="primary")

    # Parse ticker list
    tickers: list[str] = [
        t.strip().upper()
        for t in raw_tickers_input.split(",")
        if t.strip()
    ]
    if not tickers:
        tickers = DEFAULT_TICKERS

    # Build ticker color map
    ticker_colors: dict[str, str] = {
        t: _ticker_color(t, i) for i, t in enumerate(tickers)
    }

    # Default selected tab = first ticker
    selected_ticker = tickers[0] if tickers else "SPY"
    accent = ticker_colors.get(selected_ticker, EXTRA_COLORS[0])  # noqa: F841 — kept for future use

    # ---- Page header ---------------------------------------------------
    st.markdown(
        f'<h1 style="font-family:Outfit,sans-serif; font-size:1.6rem; '
        f'font-weight:700; color:#f8fafc; margin-bottom:0.25rem;">'
        f'Multi-Asset Regime Backtester</h1>'
        f'<p style="font-family:Outfit,sans-serif; color:#64748b; '
        f'font-size:0.92rem; margin-top:0;">Walk-forward HMM strategy vs buy-and-hold across assets</p>',
        unsafe_allow_html=True,
    )

    # ---- Guard: only run if button clicked (or on first load show prompt) --
    session_key = "backtest_results_v1"
    if run_clicked or session_key not in st.session_state:
        if not run_clicked and session_key not in st.session_state:
            st.info(
                "Configure assets in the sidebar and click **Run Backtest** to begin."
            )
            return

    if run_clicked:
        st.session_state.pop(session_key, None)

    # ---- Run or retrieve results ----------------------------------------
    if session_key not in st.session_state:
        results: dict[str, BacktestResult] = {}
        errors: dict[str, str] = {}

        with st.spinner(f"Running backtest for {len(tickers)} asset(s)..."):
            for ticker in tickers:
                try:
                    result = _run_backtest(ticker, five_years_ago, today)
                    results[ticker] = result
                except Exception as exc:
                    errors[ticker] = str(exc)

        st.session_state[session_key] = {"results": results, "errors": errors}
    else:
        cached = st.session_state[session_key]
        results = cached["results"]
        errors = cached["errors"]

    # Show any per-ticker errors in sidebar
    if errors:
        for ticker, err_msg in errors.items():
            st.sidebar.error(f"{_html.escape(ticker)}: {err_msg[:120]}")

    if not results:
        st.error("No backtest results available. Check ticker symbols and try again.")
        return

    # ---- Asset tabs -------------------------------------------------------
    tab_labels = [t for t in tickers if t in results]
    if not tab_labels:
        st.warning("No successful results to display.")
        return

    tabs = st.tabs(tab_labels)

    for tab, ticker in zip(tabs, tab_labels):
        result = results[ticker]
        color = ticker_colors.get(ticker, EXTRA_COLORS[0])

        with tab:
            # Re-inject CSS with this tab's accent so pill highlights correctly
            _inject_global_css(color)

            si = result.sharpe - result.benchmark_sharpe
            si_sign = "+" if si >= 0 else ""

            # Tab summary line
            st.markdown(
                f'<div style="margin-bottom:1rem;">'
                f'<span style="font-family:Fira Code,monospace; color:{color}; '
                f'font-size:1.05rem; font-weight:600;">'
                f'{result.total_return*100:+.1f}% total</span>'
                f'&nbsp;&nbsp;'
                f'<span style="font-family:Fira Code,monospace; color:#94a3b8; '
                f'font-size:0.88rem;">Sharpe Δ: {si_sign}{si:.2f} vs B&H</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # ---- Metric cards row ----------------------------------------
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.markdown(
                    f'<div class="asset-card">'
                    f'<div style="color:#64748b;font-family:Outfit,sans-serif;'
                    f'font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;'
                    f'margin-bottom:4px;">Total Return</div>'
                    f'<div class="regime-metric" style="color:{color};">'
                    f'{result.total_return*100:+.2f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with m_col2:
                st.markdown(
                    f'<div class="asset-card">'
                    f'<div style="color:#64748b;font-family:Outfit,sans-serif;'
                    f'font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;'
                    f'margin-bottom:4px;">Sharpe Ratio</div>'
                    f'<div class="regime-metric" style="color:{color};">'
                    f'{result.sharpe:.2f}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with m_col3:
                st.markdown(
                    f'<div class="asset-card">'
                    f'<div style="color:#64748b;font-family:Outfit,sans-serif;'
                    f'font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;'
                    f'margin-bottom:4px;">Max Drawdown</div>'
                    f'<div class="regime-metric" style="color:{color};">'
                    f'{result.max_drawdown*100:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with m_col4:
                st.markdown(
                    f'<div class="asset-card">'
                    f'<div style="color:#64748b;font-family:Outfit,sans-serif;'
                    f'font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;'
                    f'margin-bottom:4px;">Win Rate</div>'
                    f'<div class="regime-metric" style="color:{color};">'
                    f'{result.win_rate*100:.1f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ---- Hero equity curve chart ----------------------------------
            fig = _build_equity_chart(ticker, result, color)
            st.plotly_chart(fig, use_container_width=True)

    # ---- Regime Timeline section -----------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<h3 style="font-family:Outfit,sans-serif; font-size:1.1rem; '
        'font-weight:600; color:#f8fafc; border-left:3px solid var(--accent);'
        'padding-left:10px; margin-bottom:1rem;">Regime Timeline</h3>',
        unsafe_allow_html=True,
    )

    # Fit regimes for all successful tickers (cached separately)
    regime_results: dict[str, RegimeResult] = {}
    for ticker in tab_labels:
        try:
            rr = _fit_regime(ticker, five_years_ago, today)
            regime_results[ticker] = rr
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Regime fit failed for %s: %s", ticker, exc)

    if regime_results:
        timeline_fig = build_regime_timeline_chart(
            list(regime_results.keys()),
            regime_results,
            ticker_colors,
        )
        st.plotly_chart(timeline_fig, use_container_width=True)
    else:
        st.info("Regime timeline unavailable — HMM fit failed for all assets.")

    # ---- Asset ranking table ---------------------------------------------
    st.markdown(
        '<h3 style="font-family:Outfit,sans-serif; font-size:1.1rem; '
        'font-weight:600; color:#f8fafc; border-left:3px solid var(--accent);'
        'padding-left:10px; margin:1.5rem 0 0.5rem 0;">Asset Comparison</h3>',
        unsafe_allow_html=True,
    )

    # Sort tickers by Sharpe improvement descending
    ranked_tickers = sorted(
        tab_labels,
        key=lambda t: results[t].sharpe - results[t].benchmark_sharpe,
        reverse=True,
    )
    _render_comparison_table(ranked_tickers, results, ticker_colors)

    # ---- Stress test section ---------------------------------------------
    st.markdown(
        '<h3 style="font-family:Outfit,sans-serif; font-size:1.1rem; '
        'font-weight:600; color:#f8fafc; border-left:3px solid var(--accent);'
        'padding-left:10px; margin:1.5rem 0 0.5rem 0;">Stress Test Periods</h3>',
        unsafe_allow_html=True,
    )
    _render_stress_section(ranked_tickers, results, ticker_colors)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

main()
