"""
pages/3_Sensitivity.py — Clean Minimal Sensitivity Analysis Dashboard.

Design language: clean minimal Jupyter-inspired — light-touch grid, IBM Plex typefaces,
muted analytical green accent (#22c55e), generous whitespace, zero glow effects.

Displays:
  - Sidebar: ticker, date range, 4 base-parameter sliders, Run Analysis button
  - Overall robustness gauge: large centered score with Robust / Moderate / Fragile label
  - 4-column parameter robustness summary cards
  - 2x2 parameter sweep charts: each shows 4 metric lines vs the swept parameter

Note: Stop-loss and take-profit are approximated as per-bar return caps, not true
exit-at-price logic. Drawdown figures will appear lower than real trading results.
"""

from __future__ import annotations

import datetime
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# --- Must be the FIRST Streamlit call ---
st.set_page_config(
    page_title="Sensitivity Analysis",
    page_icon="📐",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------
from core.data import date_range_default, load_ohlcv
from core.design_system import get_plotly_layout

# ---------------------------------------------------------------------------
# Design tokens (local to this dashboard — clean minimal palette)
# ---------------------------------------------------------------------------
_BG = "#0f1117"
_CARD_BG = "#1a1c25"
_CARD_BORDER = "rgba(255,255,255,0.06)"
_PRIMARY = "#22c55e"
_WARNING = "#f59e0b"
_DANGER = "#ef4444"
_TEXT_PRIMARY = "#e2e8f0"
_TEXT_MUTED = "#64748b"
_GRID_COLOR = "#1e2535"

# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

.stApp {
    background-color: #0f1117;
    color: #e2e8f0;
    font-family: 'IBM Plex Sans', sans-serif;
}

.stSidebar {
    background-color: #13151e !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}

/* Sidebar text */
.stSidebar label,
.stSidebar .stMarkdown p {
    font-family: 'IBM Plex Sans', sans-serif !important;
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
}

/* Inputs */
.stTextInput input,
.stDateInput input,
.stNumberInput input {
    background-color: #1a1c25;
    border: 1px solid rgba(255,255,255,0.09);
    color: #e2e8f0;
    font-family: 'IBM Plex Mono', monospace;
    border-radius: 4px;
}

/* Sliders */
.stSlider [data-baseweb="slider"] {
    color: #22c55e;
}

/* Run button */
.stButton > button {
    background-color: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.35);
    color: #22c55e;
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.88rem;
    font-weight: 500;
    border-radius: 6px;
    padding: 0.45rem 1.2rem;
    letter-spacing: 0.02em;
}

.stButton > button:hover {
    background-color: rgba(34,197,94,0.16);
    border-color: #22c55e;
    color: #ffffff;
}

/* Robustness gauge block */
.robustness-gauge {
    background: #1a1c25;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 2.5rem 2rem;
    text-align: center;
}

.robustness-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 5rem;
    font-weight: 500;
    line-height: 1;
    letter-spacing: -0.02em;
}

.robustness-label {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.6rem;
}

/* Parameter robustness card */
.param-card {
    background: #1a1c25;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 1.1rem 1.2rem;
    height: 100%;
}

.param-card-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
}

.param-card-score {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    line-height: 1.1;
}

.param-card-tag {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

/* Section heading */
.sens-section-heading {
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 0.7rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
    margin-bottom: 0.5rem;
    padding-left: 0.75rem;
    border-left: 2px solid rgba(34,197,94,0.4);
}

/* Status indicator */
.sens-status {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    font-weight: 500;
    padding: 2px 10px;
    border-radius: 4px;
    letter-spacing: 0.04em;
}

/* Divider */
.sens-divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 1rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# SMA crossover backtest (fast local — does NOT call core/backtest.py)
# ---------------------------------------------------------------------------


def _sma_backtest(
    prices: pd.Series,
    fast: int,
    slow: int,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> dict[str, float]:
    """Run a simple SMA crossover strategy and return performance metrics.

    Parameters
    ----------
    prices:           Close price series.
    fast:             Fast MA window in bars.
    slow:             Slow MA window in bars.
    stop_loss_pct:    Per-trade stop-loss as a positive percentage (e.g. 2.0 = 2%).
    take_profit_pct:  Per-trade take-profit as a positive percentage (e.g. 4.0 = 4%).

    Returns
    -------
    dict with keys: total_return, sharpe, max_drawdown, win_rate
    """
    prices = prices.dropna()
    if len(prices) < slow + 5:
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.5}

    fast_ma = prices.rolling(fast).mean()
    slow_ma_s = prices.rolling(slow).mean()

    # Long when fast MA > slow MA; shift by 1 to avoid look-ahead bias
    signal = (fast_ma > slow_ma_s).astype(float).shift(1).fillna(0.0)

    daily_ret = prices.pct_change().fillna(0.0)

    # Apply simplified stop-loss / take-profit on a per-bar basis:
    # When in a long position, cap single-bar return within [-stop_loss, take_profit].
    stop = stop_loss_pct / 100.0
    tp = take_profit_pct / 100.0
    strategy_ret = signal * daily_ret.clip(lower=-stop, upper=tp)

    # Metrics
    equity = (1.0 + strategy_ret).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)

    ann_ret = float(strategy_ret.mean() * 252)
    ann_vol = float(strategy_ret.std() * (252 ** 0.5))
    sharpe = ann_ret / ann_vol if ann_vol > 1e-9 else 0.0

    rolling_max = equity.cummax()
    drawdowns = equity / rolling_max - 1.0
    max_dd = float(drawdowns.min())  # negative number

    in_trade = strategy_ret[strategy_ret != 0.0]
    win_rate = float((in_trade > 0).mean()) if len(in_trade) > 0 else 0.5

    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
    }


# ---------------------------------------------------------------------------
# Parameter sweep (cached for performance)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def _sweep_fast_ma(
    prices_tuple: tuple[float, ...],
    slow: int,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> dict[str, list[Any]]:
    """Sweep fast_ma from 5 to 30 (step 1) holding other params fixed."""
    prices = pd.Series(prices_tuple)
    fast_range = (5, 30)
    fast_step = 1
    rows: list[dict[str, float]] = []
    param_values: list[int] = []
    for v in range(fast_range[0], fast_range[1] + 1, fast_step):
        if v >= slow:
            continue  # skip: fast MA must be < slow MA
        param_values.append(v)
        rows.append(_sma_backtest(prices, fast=v, slow=slow,
                                  stop_loss_pct=stop_loss_pct,
                                  take_profit_pct=take_profit_pct))
    return {"values": param_values, "metrics": rows}


@st.cache_data(ttl=3600)
def _sweep_slow_ma(
    prices_tuple: tuple[float, ...],
    fast: int,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> dict[str, list[Any]]:
    """Sweep slow_ma from 20 to 100 (step 5) holding other params fixed."""
    prices = pd.Series(prices_tuple)
    slow_range = (20, 100)
    slow_step = 5
    rows: list[dict[str, float]] = []
    param_values: list[int] = []
    for v in range(slow_range[0], slow_range[1] + 1, slow_step):
        if v <= fast:
            continue  # skip: slow MA must be > fast MA
        param_values.append(v)
        rows.append(_sma_backtest(prices, fast=fast, slow=v,
                                  stop_loss_pct=stop_loss_pct,
                                  take_profit_pct=take_profit_pct))
    return {"values": param_values, "metrics": rows}


@st.cache_data(ttl=3600)
def _sweep_stop_loss(
    prices_tuple: tuple[float, ...],
    fast: int,
    slow: int,
    take_profit_pct: float,
) -> dict[str, list[Any]]:
    """Sweep stop_loss_pct from 0.5 to 5.0 (step 0.5) holding other params fixed."""
    prices = pd.Series(prices_tuple)
    param_values = [round(v * 0.5, 1) for v in range(1, 11)]  # 0.5 → 5.0
    rows: list[dict[str, float]] = []
    for v in param_values:
        rows.append(_sma_backtest(prices, fast=fast, slow=slow,
                                  stop_loss_pct=v,
                                  take_profit_pct=take_profit_pct))
    return {"values": param_values, "metrics": rows}


@st.cache_data(ttl=3600)
def _sweep_take_profit(
    prices_tuple: tuple[float, ...],
    fast: int,
    slow: int,
    stop_loss_pct: float,
) -> dict[str, list[Any]]:
    """Sweep take_profit_pct from 1.0 to 10.0 (step 0.5) holding other params fixed."""
    prices = pd.Series(prices_tuple)
    param_values = [round(v * 0.5, 1) for v in range(2, 21)]  # 1.0 → 10.0
    rows: list[dict[str, float]] = []
    for v in param_values:
        rows.append(_sma_backtest(prices, fast=fast, slow=slow,
                                  stop_loss_pct=stop_loss_pct,
                                  take_profit_pct=v))
    return {"values": param_values, "metrics": rows}


@st.cache_data(ttl=3600)
def cached_load_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Cache-wrapped yfinance loader — one network call per (ticker, start, end)."""
    return load_ohlcv(ticker, start, end)


# ---------------------------------------------------------------------------
# Robustness scoring
# ---------------------------------------------------------------------------


def _robustness_score_for_sweep(metrics: list[dict[str, float]]) -> float:
    """Compute a 0–100 robustness score from a list of metric dicts.

    Score per metric = max(0, 100 - CV * 100)  where CV = std/|mean|.
    Overall = mean of the 4 metric scores.
    """
    metric_keys = ["total_return", "sharpe", "max_drawdown", "win_rate"]
    scores: list[float] = []
    for key in metric_keys:
        vals = np.array([m[key] for m in metrics], dtype=float)
        mean_abs = abs(float(np.mean(vals)))
        if mean_abs < 1e-9:
            # Degenerate — CV undefined; treat as fully uncertain (score 50)
            scores.append(50.0)
        else:
            cv = float(np.std(vals)) / mean_abs
            scores.append(max(0.0, 100.0 - cv * 100.0))
    return float(np.mean(scores))


def _score_label_color(score: float) -> tuple[str, str]:
    """Return (label, CSS color) for a robustness score."""
    if score > 70:
        return "Robust", _PRIMARY
    if score >= 40:
        return "Moderate", _WARNING
    return "Fragile", _DANGER


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> tuple[str, datetime.date, datetime.date, int, int, float, float, bool]:
    """Render sidebar inputs.

    Returns
    -------
    (ticker, start_date, end_date, fast_ma, slow_ma, stop_loss_pct, take_profit_pct,
     run_clicked)
    """
    st.sidebar.markdown(
        '<p style="color:#22c55e;font-family:\'IBM Plex Mono\',monospace;'
        'font-size:0.68rem;letter-spacing:0.12em;font-weight:500;'
        'margin-bottom:0.8rem;">SENSITIVITY / INPUTS</p>',
        unsafe_allow_html=True,
    )

    ticker = (
        st.sidebar.text_input("TICKER", value="SPY", max_chars=10, key="sa_ticker")
        .upper()
        .strip()
    )

    default_start_str, default_end_str = date_range_default(years=5)
    default_start = datetime.date.fromisoformat(default_start_str)
    default_end = datetime.date.fromisoformat(default_end_str)

    start_date = st.sidebar.date_input("START DATE", value=default_start, key="sa_start")
    end_date = st.sidebar.date_input("END DATE", value=default_end, key="sa_end")

    st.sidebar.markdown(
        '<div class="sens-divider" style="margin:0.8rem 0;"></div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<p style="color:#64748b;font-family:\'IBM Plex Mono\',monospace;'
        'font-size:0.65rem;letter-spacing:0.1em;margin-bottom:0.4rem;">'
        'BASE PARAMETER VALUES</p>',
        unsafe_allow_html=True,
    )

    fast_ma = int(
        st.sidebar.slider("Fast MA (bars)", min_value=5, max_value=30,
                          value=10, step=1, key="sa_fast_ma")
    )
    slow_ma = int(
        st.sidebar.slider("Slow MA (bars)", min_value=20, max_value=100,
                          value=50, step=5, key="sa_slow_ma")
    )
    stop_loss_pct = float(
        st.sidebar.slider("Stop Loss (%)", min_value=0.5, max_value=5.0,
                          value=2.0, step=0.5, key="sa_stop_loss")
    )
    take_profit_pct = float(
        st.sidebar.slider("Take Profit (%)", min_value=1.0, max_value=10.0,
                          value=4.0, step=0.5, key="sa_take_profit")
    )
    st.sidebar.caption(
        "⚠ Stop-loss / take-profit approximated as per-bar return caps. "
        "Drawdown figures are optimistic."
    )

    st.sidebar.markdown(
        '<div class="sens-divider" style="margin:0.8rem 0;"></div>',
        unsafe_allow_html=True,
    )

    run_clicked = st.sidebar.button("▶ Run Analysis", use_container_width=True)

    # Status indicator
    status_key = st.session_state.get("sa_status", "ready")
    if status_key == "running":
        status_html = (
            '<span class="sens-status" style="background:rgba(245,158,11,0.1);'
            'color:#f59e0b;border:1px solid rgba(245,158,11,0.4);">● Running...</span>'
        )
    elif status_key == "done":
        status_html = (
            '<span class="sens-status" style="background:rgba(34,197,94,0.1);'
            'color:#22c55e;border:1px solid rgba(34,197,94,0.4);">✓ Complete</span>'
        )
    else:
        status_html = (
            '<span class="sens-status" style="background:rgba(100,116,139,0.1);'
            'color:#64748b;border:1px solid rgba(100,116,139,0.3);">○ Ready</span>'
        )

    st.sidebar.markdown(
        f'<div style="margin-top:0.5rem;">{status_html}</div>',
        unsafe_allow_html=True,
    )

    return (
        ticker,
        start_date,
        end_date,
        fast_ma,
        slow_ma,
        stop_loss_pct,
        take_profit_pct,
        run_clicked,
    )


# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------


def render_welcome() -> None:
    """Show instructions when no analysis has been run yet."""
    st.markdown(
        """
<div style="
    border: 1px solid rgba(255,255,255,0.06);
    border-left: 3px solid #22c55e;
    background: #1a1c25;
    padding: 2rem 2.5rem;
    border-radius: 8px;
    margin-top: 2rem;
    max-width: 720px;
">
  <p style="
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.8rem;
  ">SENSITIVITY ANALYSIS / READY</p>
  <h2 style="
    font-family: 'IBM Plex Sans', sans-serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #e2e8f0;
    margin: 0 0 1rem 0;
  ">Parameter Robustness Testing</h2>
  <p style="font-family:'IBM Plex Sans',sans-serif;font-size:0.85rem;color:#94a3b8;line-height:1.7;margin-bottom:0.8rem;">
    Configure the ticker, date range, and base parameter values in the sidebar, then press
    <span style="color:#22c55e;font-weight:600;">▶ Run Analysis</span>
    to sweep each parameter independently and score strategy robustness.
  </p>
  <p style="font-family:'IBM Plex Mono',sans-serif;font-size:0.78rem;color:#64748b;line-height:1.6;margin:0;">
    Strategy: SMA crossover on daily close prices.&nbsp;&nbsp;
    Parameters swept: fast_ma · slow_ma · stop_loss · take_profit
  </p>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


def render_header(ticker: str) -> None:
    """Render the page title and subtitle."""
    st.markdown(
        f'<h1 style="font-family:\'IBM Plex Sans\',sans-serif;font-size:1.6rem;'
        f'font-weight:600;color:#e2e8f0;margin-bottom:0.2rem;">'
        f'Sensitivity Analysis</h1>'
        f'<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.78rem;'
        f'color:#64748b;margin-top:0;margin-bottom:1.4rem;">'
        f'Parameter robustness testing for SMA crossover strategy'
        f'&nbsp;·&nbsp;<span style="color:#94a3b8;">{ticker}</span>'
        f'</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Overall robustness gauge
# ---------------------------------------------------------------------------


def render_overall_gauge(overall_score: float) -> None:
    """Render the large centred robustness score block."""
    label, color = _score_label_color(overall_score)

    st.markdown(
        f'<div class="robustness-gauge">'
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;'
        f'color:#64748b;text-transform:uppercase;letter-spacing:0.1em;'
        f'margin-bottom:1rem;">Overall Robustness Score</div>'
        f'<div class="robustness-score" style="color:{color};">'
        f'{overall_score:.0f}</div>'
        f'<div class="robustness-label" style="color:{color};">{label}</div>'
        f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.7rem;'
        f'color:#64748b;margin-top:1rem;">'
        f'<span style="color:#22c55e;">▌</span>&nbsp;&gt; 70 Robust&nbsp;&nbsp;'
        f'<span style="color:#f59e0b;">▌</span>&nbsp;40–70 Moderate&nbsp;&nbsp;'
        f'<span style="color:#ef4444;">▌</span>&nbsp;&lt; 40 Fragile'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Parameter robustness summary cards (1 per parameter)
# ---------------------------------------------------------------------------


def render_param_cards(param_scores: dict[str, float]) -> None:
    """Render 4 small cards — one per swept parameter — with individual scores."""
    param_labels = {
        "fast_ma": "Fast MA",
        "slow_ma": "Slow MA",
        "stop_loss": "Stop Loss",
        "take_profit": "Take Profit",
    }
    cols = st.columns(4)
    for col, (key, score) in zip(cols, param_scores.items()):
        label, color = _score_label_color(score)
        with col:
            st.markdown(
                f'<div class="param-card">'
                f'<div class="param-card-label">{param_labels.get(key, key)}</div>'
                f'<div class="param-card-score" style="color:{color};">{score:.0f}</div>'
                f'<div class="param-card-tag" style="color:{color};">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Sweep chart helpers
# ---------------------------------------------------------------------------

_METRIC_DISPLAY: dict[str, tuple[str, str]] = {
    # key -> (display name, CSS color)
    "total_return": ("Total Return", "#22c55e"),
    "sharpe":       ("Sharpe Ratio", "#3b82f6"),
    "max_drawdown": ("Max Drawdown", "#ef4444"),
    "win_rate":     ("Win Rate",     "#f59e0b"),
}


def _build_sweep_chart(
    param_label: str,
    param_values: list[float],
    metrics: list[dict[str, float]],
) -> go.Figure:
    """Build a Plotly figure with 4 normalised metric lines for one parameter sweep.

    Each metric is normalised to its own [0, 1] range across the sweep so all
    four lines share the same Y axis for easy comparison.
    """
    fig = go.Figure()

    for metric_key, (metric_name, line_color) in _METRIC_DISPLAY.items():
        raw_vals = np.array([m[metric_key] for m in metrics], dtype=float)

        # Normalise to [0, 1]; if flat, keep at 0.5
        v_min = raw_vals.min()
        v_max = raw_vals.max()
        span = v_max - v_min
        if span < 1e-9:
            norm_vals = np.full_like(raw_vals, 0.5)
        else:
            norm_vals = (raw_vals - v_min) / span

        hover_raw = [f"{v:.4f}" for v in raw_vals]

        fig.add_trace(
            go.Scatter(
                x=param_values,
                y=norm_vals.tolist(),
                mode="lines",
                name=metric_name,
                line={"color": line_color, "width": 1.5},
                customdata=hover_raw,
                hovertemplate=(
                    f"<b>{param_label}</b> %{{x}}<br>"
                    f"{metric_name}: %{{customdata}}<extra></extra>"
                ),
            )
        )

    # Start from the shared dark-theme base, then apply Dashboard-3-specific overrides.
    _base_layout = get_plotly_layout(theme="dark")
    fig.update_layout(**_base_layout)
    fig.update_layout(
        paper_bgcolor=_CARD_BG,
        plot_bgcolor=_CARD_BG,
        height=240,
        margin={"l": 44, "r": 16, "t": 36, "b": 36},
        font={
            "color": _TEXT_PRIMARY,
            "family": "'IBM Plex Mono', monospace",
            "size": 11,
        },
        title={
            "text": param_label,
            "font": {
                "family": "'IBM Plex Sans', sans-serif",
                "size": 13,
                "color": _TEXT_PRIMARY,
            },
            "x": 0.0,
            "xanchor": "left",
            "pad": {"l": 6},
        },
        xaxis={
            "showgrid": True,
            "gridcolor": _GRID_COLOR,
            "gridwidth": 1,
            "zeroline": False,
            "tickfont": {"color": _TEXT_MUTED, "size": 10},
            "titlefont": {"color": _TEXT_MUTED, "size": 10},
        },
        yaxis={
            "showgrid": True,
            "gridcolor": _GRID_COLOR,
            "gridwidth": 1,
            "zeroline": False,
            "range": [-0.05, 1.05],
            "tickformat": ".1f",
            "title": "Normalised",
            "tickfont": {"color": _TEXT_MUTED, "size": 10},
            "titlefont": {"color": _TEXT_MUTED, "size": 10},
        },
        legend={
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "font": {"color": _TEXT_MUTED, "size": 10},
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1.0,
        },
        hoverlabel={
            "bgcolor": _CARD_BG,
            "bordercolor": _GRID_COLOR,
            "font": {"color": _TEXT_PRIMARY, "size": 11},
        },
    )

    return fig


# ---------------------------------------------------------------------------
# Sweep charts section (2×2 grid)
# ---------------------------------------------------------------------------


def render_sweep_charts(sweep_results: dict[str, dict]) -> None:
    """Render all four parameter sweep charts in a 2×2 grid."""
    st.markdown(
        '<p class="sens-section-heading">Parameter Sweep Charts — '
        'normalised metrics vs each swept parameter</p>',
        unsafe_allow_html=True,
    )

    param_meta = [
        ("fast_ma",     "Fast MA (bars)"),
        ("slow_ma",     "Slow MA (bars)"),
        ("stop_loss",   "Stop Loss (%)"),
        ("take_profit", "Take Profit (%)"),
    ]

    row1_left, row1_right = st.columns(2)
    row2_left, row2_right = st.columns(2)
    columns_grid = [row1_left, row1_right, row2_left, row2_right]

    for col, (key, param_label) in zip(columns_grid, param_meta):
        sweep = sweep_results[key]
        fig = _build_sweep_chart(
            param_label=param_label,
            param_values=sweep["values"],
            metrics=sweep["metrics"],
        )
        with col:
            st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Base-value backtest summary strip
# ---------------------------------------------------------------------------


def render_base_metrics(base_result: dict[str, float]) -> None:
    """Render a compact strip showing the strategy metrics at base parameter values."""
    st.markdown(
        '<p class="sens-section-heading">Base Parameter Performance</p>',
        unsafe_allow_html=True,
    )

    items = [
        ("Total Return", f"{base_result['total_return'] * 100:.1f}%"),
        ("Sharpe Ratio", f"{base_result['sharpe']:.2f}"),
        ("Max Drawdown", f"{base_result['max_drawdown'] * 100:.1f}%"),
        ("Win Rate",     f"{base_result['win_rate'] * 100:.1f}%"),
    ]

    cols = st.columns(4)
    colors = [_PRIMARY, "#3b82f6", _DANGER, _WARNING]
    for col, (label, val), color in zip(cols, items, colors):
        with col:
            st.markdown(
                f'<div style="background:#1a1c25;border:1px solid rgba(255,255,255,0.06);'
                f'border-left:3px solid {color};border-radius:6px;padding:0.9rem 1rem;">'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.65rem;'
                f'color:#64748b;text-transform:uppercase;letter-spacing:0.08em;'
                f'margin-bottom:0.35rem;">{label}</div>'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:1.4rem;'
                f'font-weight:500;color:{color};line-height:1.1;">{val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the Sensitivity Analysis dashboard."""
    # Sidebar
    (
        ticker,
        start_date,
        end_date,
        fast_ma,
        slow_ma,
        stop_loss_pct,
        take_profit_pct,
        run_clicked,
    ) = render_sidebar()

    render_header(ticker)

    # ---- Run analysis when button clicked ----
    if run_clicked:
        if start_date >= end_date:
            st.error("End date must be after start date.")
            st.stop()

        st.session_state["sa_status"] = "running"
        # Clear previous results
        for key in ("sa_result", "sa_ticker_display"):
            st.session_state.pop(key, None)

        with st.spinner("Fetching data and running parameter sweeps…"):
            try:
                raw_df = cached_load_ohlcv(
                    ticker,
                    start_date.isoformat(),
                    end_date.isoformat(),
                )
                # Normalise column names to lowercase
                df = raw_df.rename(columns=str.lower)
                prices = df["close"]

                # Convert to tuple for cache-friendly hashing
                prices_tuple = tuple(prices.tolist())

                # Base-case result
                base_result = _sma_backtest(
                    prices, fast=fast_ma, slow=slow_ma,
                    stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
                )

                # Sweeps
                sweep_fast = _sweep_fast_ma(
                    prices_tuple, slow=slow_ma,
                    stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
                )
                sweep_slow = _sweep_slow_ma(
                    prices_tuple, fast=fast_ma,
                    stop_loss_pct=stop_loss_pct, take_profit_pct=take_profit_pct,
                )
                sweep_stop = _sweep_stop_loss(
                    prices_tuple, fast=fast_ma, slow=slow_ma,
                    take_profit_pct=take_profit_pct,
                )
                sweep_tp = _sweep_take_profit(
                    prices_tuple, fast=fast_ma, slow=slow_ma,
                    stop_loss_pct=stop_loss_pct,
                )

                sweep_results = {
                    "fast_ma":     sweep_fast,
                    "slow_ma":     sweep_slow,
                    "stop_loss":   sweep_stop,
                    "take_profit": sweep_tp,
                }

                # Robustness scores
                param_scores: dict[str, float] = {
                    k: _robustness_score_for_sweep(v["metrics"])
                    for k, v in sweep_results.items()
                }
                overall_score = float(np.mean(list(param_scores.values())))

                st.session_state["sa_result"] = {
                    "base_result":   base_result,
                    "sweep_results": sweep_results,
                    "param_scores":  param_scores,
                    "overall_score": overall_score,
                }
                st.session_state["sa_ticker_display"] = ticker
                st.session_state["sa_status"] = "done"

            except ValueError as exc:
                st.session_state["sa_status"] = "ready"
                st.error(f"Data load failed: {exc}")
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.session_state["sa_status"] = "ready"
                st.error(f"Analysis failed: {exc}")
                st.stop()

        st.rerun()

    # ---- Render results if available ----
    if "sa_result" not in st.session_state:
        render_welcome()
        return

    result = st.session_state["sa_result"]
    overall_score: float = result["overall_score"]
    param_scores: dict[str, float] = result["param_scores"]
    sweep_results: dict = result["sweep_results"]
    base_result: dict[str, float] = result["base_result"]

    # Overall gauge — centred in a narrow column for visual weight
    _, gauge_col, _ = st.columns([1, 2, 1])
    with gauge_col:
        render_overall_gauge(overall_score)

    st.markdown('<div style="height:1.2rem;"></div>', unsafe_allow_html=True)

    # Per-parameter score cards
    st.markdown(
        '<p class="sens-section-heading">Individual Parameter Scores</p>',
        unsafe_allow_html=True,
    )
    render_param_cards(param_scores)

    st.markdown('<div style="height:1.4rem;"></div>', unsafe_allow_html=True)

    # Base parameter metrics strip
    render_base_metrics(base_result)

    st.markdown('<div style="height:1.4rem;"></div>', unsafe_allow_html=True)

    # 2×2 sweep charts
    render_sweep_charts(sweep_results)

    # Footer metadata
    st.markdown(
        f'<p style="font-family:\'IBM Plex Mono\',monospace;font-size:0.68rem;'
        f'color:#64748b;margin-top:1.2rem;">'
        f'Strategy: SMA Crossover · '
        f'fast={fast_ma} · slow={slow_ma} · '
        f'stop={stop_loss_pct}% · tp={take_profit_pct}%'
        f'</p>',
        unsafe_allow_html=True,
    )


main()
