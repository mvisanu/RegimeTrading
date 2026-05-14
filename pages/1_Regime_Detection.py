"""
pages/1_Regime_Detection.py — Bloomberg Terminal Regime Detection Dashboard.

Design language: Bloomberg terminal — dark background, cyan accent, monospace
numbers, angular cards, no rounded corners.

Displays:
  - Hero price chart with colored regime bands (the visual centrepiece)
  - Top-bar summary metrics (ticker, current regime badge, confidence, stability)
  - Regime statistics grid (mean return, mean vol, % time per regime)
  - Confidence timeline area chart
"""

from __future__ import annotations

import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Must be the FIRST Streamlit call ---
st.set_page_config(page_title="Regime Detection", page_icon="📊", layout="wide")

# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------
from core.design_system import (
    ACCENT_CYAN,
    REGIME_COLORS,
    get_plotly_layout,
    metric_card,
    regime_badge,
    section_header,
)
from core.data import date_range_default, load_ohlcv
from core.hmm_utils import fit_and_filter
import core.verify as _verify

# ---------------------------------------------------------------------------
# Bloomberg terminal CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
/* Bloomberg terminal override */
.stApp { background-color: #0e1117; }
.stSidebar { background-color: #161922; border-right: 1px solid #2d3748; }
.stMetric { background: #1e2130; border: 1px solid #2d3748; padding: 12px; border-radius: 2px; }
.stMetric label { color: #64748b !important; font-family: 'Courier New', monospace; font-size: 11px; }
.stMetric [data-testid="metric-container"] { font-family: 'Courier New', monospace; }
div[data-testid="stHorizontalBlock"] { gap: 1px; }
/* Monospace all text elements */
.stMarkdown, .stText, h1, h2, h3, p, span {
    font-family: 'Courier New', 'IBM Plex Mono', monospace !important;
}
/* Tighten sidebar padding */
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
/* Angular buttons — Bloomberg style */
.stButton > button {
    background-color: #0e1117;
    border: 1px solid #00d4ff;
    color: #00d4ff;
    font-family: 'Courier New', monospace;
    font-size: 0.85rem;
    font-weight: 600;
    border-radius: 2px;
    letter-spacing: 0.05em;
    padding: 0.45rem 1.2rem;
}
.stButton > button:hover {
    background-color: #00d4ff18;
    border-color: #00d4ff;
    color: #ffffff;
}
/* Ticker input field */
.stTextInput input {
    background-color: #1e2130;
    border: 1px solid #2d3748;
    color: #e2e8f0;
    font-family: 'Courier New', monospace;
    border-radius: 2px;
}
/* Date pickers */
.stDateInput input {
    background-color: #1e2130;
    border: 1px solid #2d3748;
    color: #e2e8f0;
    font-family: 'Courier New', monospace;
    border-radius: 2px;
}
/* Number input */
.stNumberInput input {
    background-color: #1e2130;
    border: 1px solid #2d3748;
    color: #e2e8f0;
    font-family: 'Courier New', monospace;
    border-radius: 2px;
}
/* Selectbox / radio */
.stRadio label, .stSelectbox label {
    font-family: 'Courier New', monospace;
    color: #94a3b8;
    font-size: 0.8rem;
}
/* Section divider */
.bloomberg-divider {
    height: 1px;
    background: linear-gradient(90deg, #00d4ff44 0%, #2d3748 100%);
    margin: 1rem 0;
}
/* Ticker display */
.bb-ticker {
    font-family: 'Courier New', monospace;
    font-size: 2.2rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: 0.04em;
    line-height: 1.1;
}
/* Metric label above value */
.bb-metric-label {
    font-family: 'Courier New', monospace;
    font-size: 0.65rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 2px;
}
/* Metric value */
.bb-metric-value {
    font-family: 'Courier New', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #00d4ff;
    line-height: 1.2;
}
/* Status pill */
.bb-status {
    display: inline-block;
    font-family: 'Courier New', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 2px 10px;
    border-radius: 2px;
    letter-spacing: 0.04em;
}
/* Lookahead badge */
.bb-lookahead-pass {
    background: #10b981;
    color: #fff;
    font-family: 'Courier New', monospace;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 12px;
    border-radius: 2px;
    letter-spacing: 0.04em;
    display: inline-block;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def cached_load_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Cache-wrapped yfinance loader — one network request per (ticker, start, end) combo."""
    return load_ohlcv(ticker, start, end)


# ---------------------------------------------------------------------------
# Helper: build contiguous regime segments
# ---------------------------------------------------------------------------


def build_regime_segments(
    dates: list, labels: list[str]
) -> list[tuple]:
    """Return a list of (start_date, end_date, regime_label) for contiguous runs."""
    segments: list[tuple] = []
    if not labels:
        return segments
    current_label = labels[0]
    current_start = dates[0]
    for i in range(1, len(labels)):
        if labels[i] != current_label:
            segments.append((current_start, dates[i - 1], current_label))
            current_label = labels[i]
            current_start = dates[i]
    segments.append((current_start, dates[-1], current_label))
    return segments


# ---------------------------------------------------------------------------
# Helper: compute per-regime statistics
# ---------------------------------------------------------------------------


def compute_regime_stats(
    df: pd.DataFrame, stable_labels: list[str]
) -> dict[str, dict]:
    """
    Compute annualised mean return, annualised mean volatility, and time-share
    for each regime present in stable_labels.

    Returns a dict keyed by regime name with values:
        {"mean_ret": float, "mean_vol": float, "pct_time": float, "count": int}
    """
    close = df["close"]
    log_returns = np.log(close / close.shift(1)).fillna(0.0).to_numpy()

    T = len(stable_labels)
    regime_data: dict[str, list[float]] = {}

    for i, label in enumerate(stable_labels):
        if label not in regime_data:
            regime_data[label] = []
        regime_data[label].append(log_returns[i])

    stats: dict[str, dict] = {}
    for regime, rets in regime_data.items():
        arr = np.array(rets)
        mean_ret_annual = float(np.mean(arr) * 252)         # annualised
        mean_vol_annual = float(np.std(arr) * np.sqrt(252))  # annualised
        pct_time = len(arr) / T * 100.0
        stats[regime] = {
            "mean_ret": mean_ret_annual,
            "mean_vol": mean_vol_annual,
            "pct_time": pct_time,
            "count": len(arr),
        }

    return stats


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> tuple[str, datetime.date, datetime.date, int | None, bool]:
    """
    Render sidebar controls.

    Returns (ticker, start_date, end_date, n_regimes_override, run_clicked).
    n_regimes_override is None when Auto (BIC) is selected.
    """
    st.sidebar.markdown(
        '<p style="color:#00d4ff;font-family:\'Courier New\',monospace;'
        'font-size:0.7rem;letter-spacing:0.12em;font-weight:700;'
        'margin-bottom:0.8rem;">REGIME DETECTION / INPUTS</p>',
        unsafe_allow_html=True,
    )

    ticker = st.sidebar.text_input(
        "TICKER",
        value="SPY",
        max_chars=10,
        key="ticker_input",
    ).upper().strip()

    default_start_str, default_end_str = date_range_default(years=3)
    default_start = datetime.date.fromisoformat(default_start_str)
    default_end = datetime.date.fromisoformat(default_end_str)

    start_date = st.sidebar.date_input("START DATE", value=default_start, key="start_date")
    end_date = st.sidebar.date_input("END DATE", value=default_end, key="end_date")

    st.sidebar.markdown(
        '<p style="color:#64748b;font-family:\'Courier New\',monospace;'
        'font-size:0.68rem;margin-top:0.6rem;margin-bottom:0.2rem;">'
        'N REGIMES</p>',
        unsafe_allow_html=True,
    )

    auto_bic = st.sidebar.checkbox("Auto (BIC)", value=True, key="auto_bic")
    n_override: int | None = None
    if not auto_bic:
        n_override = int(
            st.sidebar.number_input(
                "Number of regimes",
                min_value=3,
                max_value=6,
                value=4,
                step=1,
                key="n_regimes",
            )
        )

    st.sidebar.markdown(
        '<div style="height:1px;background:#2d3748;margin:1rem 0;"></div>',
        unsafe_allow_html=True,
    )

    run_clicked = st.sidebar.button("▶ Run Analysis", use_container_width=True)

    # Status indicator
    status_key = st.session_state.get("run_status", "ready")
    if status_key == "running":
        status_html = (
            '<span class="bb-status" style="background:#f59e0b22;color:#f59e0b;'
            'border:1px solid #f59e0b;">● RUNNING...</span>'
        )
    elif status_key == "done":
        status_html = (
            '<span class="bb-status" style="background:#10b98122;color:#10b981;'
            'border:1px solid #10b981;">✅ ANALYSIS COMPLETE</span>'
        )
    else:
        status_html = (
            '<span class="bb-status" style="background:#2d374866;color:#64748b;'
            'border:1px solid #2d3748;">○ READY</span>'
        )

    st.sidebar.markdown(
        f'<div style="margin-top:0.6rem;">{status_html}</div>',
        unsafe_allow_html=True,
    )

    return ticker, start_date, end_date, n_override, run_clicked


# ---------------------------------------------------------------------------
# Top bar metrics
# ---------------------------------------------------------------------------


def render_top_bar(
    ticker: str,
    result,
    df: pd.DataFrame,
) -> None:
    """Render the full-width top metrics bar."""
    # Current (last) stable regime and confidence
    current_regime = result.stable_labels[-1]
    current_confidence = float(result.confidence[-1])

    # Stability: check last 20 bars
    last_20 = result.stable_labels[-20:] if len(result.stable_labels) >= 20 else result.stable_labels
    transitions = sum(1 for i in range(1, len(last_20)) if last_20[i] != last_20[i - 1])
    is_stable = transitions <= 2

    badge_html = regime_badge(current_regime, current_confidence, glow=True)

    col_ticker, col_badge, col_conf, col_stab, col_regimes, col_lookahead = st.columns(
        [1.2, 2, 1.2, 1.2, 1.2, 2]
    )

    with col_ticker:
        st.markdown(
            f'<p class="bb-metric-label">TICKER</p>'
            f'<p class="bb-ticker">{ticker}</p>',
            unsafe_allow_html=True,
        )

    with col_badge:
        st.markdown(
            f'<p class="bb-metric-label">CURRENT REGIME</p>'
            f'<div style="margin-top:6px;">{badge_html}</div>',
            unsafe_allow_html=True,
        )

    with col_conf:
        st.markdown(
            f'<p class="bb-metric-label">CONFIDENCE</p>'
            f'<p class="bb-metric-value">{current_confidence * 100:.1f}%</p>',
            unsafe_allow_html=True,
        )

    with col_stab:
        stab_color = "#10b981" if is_stable else "#64748b"
        stab_text = "Stable" if is_stable else "Uncertain"
        st.markdown(
            f'<p class="bb-metric-label">STABILITY</p>'
            f'<p style="font-family:\'Courier New\',monospace;font-size:1.1rem;'
            f'font-weight:700;color:{stab_color};">{stab_text}</p>',
            unsafe_allow_html=True,
        )

    with col_regimes:
        st.markdown(
            f'<p class="bb-metric-label">REGIMES</p>'
            f'<p class="bb-metric-value">{result.n_regimes}</p>',
            unsafe_allow_html=True,
        )

    with col_lookahead:
        st.markdown(
            '<p class="bb-metric-label">INTEGRITY</p>'
            '<div style="margin-top:6px;">'
            '<span class="bb-lookahead-pass">✅ LOOK-AHEAD CHECK PASSED</span>'
            "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Hero chart
# ---------------------------------------------------------------------------


def render_hero_chart(
    df: pd.DataFrame,
    result,
    chart_type: str,
) -> None:
    """Render the full-width hero price chart with regime bands."""
    dates = list(df.index)
    close = df["close"].to_numpy()

    # Align stable_labels to the df index (hmm_utils drops NaN rows for features)
    # The posteriors / stable_labels array has length <= len(df) due to dropna in
    # _engineer_features (rolling(20) eats the first 20 rows).
    n_labels = len(result.stable_labels)
    n_df = len(df)
    offset = n_df - n_labels  # number of leading bars with no label

    layout = get_plotly_layout()
    layout["height"] = 600
    layout["margin"] = {"l": 60, "r": 24, "t": 48, "b": 40}
    layout["xaxis"]["rangeslider"] = {"visible": False}
    layout["xaxis"]["showgrid"] = True
    layout["yaxis"]["title"] = "PRICE (USD)"
    layout["yaxis"]["tickprefix"] = "$"
    # Override font to monospace for Bloomberg feel
    layout["font"]["family"] = "'Courier New', 'IBM Plex Mono', monospace"

    fig = go.Figure()

    if chart_type == "Candlestick":
        fig.add_trace(
            go.Candlestick(
                x=dates,
                open=df["open"].to_numpy(),
                high=df["high"].to_numpy(),
                low=df["low"].to_numpy(),
                close=close,
                name=f"{st.session_state.get('ticker_input', 'SPY')}",
                increasing_line_color="#10b981",
                decreasing_line_color="#ef4444",
                increasing_fillcolor="#10b98133",
                decreasing_fillcolor="#ef444433",
            )
        )
    else:
        # Line chart
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=close,
                mode="lines",
                line={"color": "#e2e8f0", "width": 1.5},
                name="Close",
                hovertemplate="<b>%{x}</b><br>$%{y:.2f}<extra></extra>",
            )
        )

    # --- Regime bands (the visual centrepiece) ---
    # Only the bars with labels (from offset onward)
    labeled_dates = dates[offset:]
    segments = build_regime_segments(labeled_dates, result.stable_labels)

    for seg_start, seg_end, label in segments:
        color = REGIME_COLORS.get(label, "#64748b")
        fig.add_vrect(
            x0=seg_start,
            x1=seg_end,
            fillcolor=color,
            opacity=0.13,
            layer="below",
            line_width=0,
        )

    # Regime color legend (synthetic scatter traces for the legend only)
    present_regimes = sorted(set(result.stable_labels))
    for label in present_regimes:
        color = REGIME_COLORS.get(label, "#64748b")
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 10, "color": color, "symbol": "square"},
                name=label,
                showlegend=True,
            )
        )

    layout["legend"]["orientation"] = "h"
    layout["legend"]["yanchor"] = "bottom"
    layout["legend"]["y"] = 1.02
    layout["legend"]["xanchor"] = "right"
    layout["legend"]["x"] = 1.0

    fig.update_layout(**layout)

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Regime statistics grid
# ---------------------------------------------------------------------------


def render_regime_stats(stats: dict[str, dict]) -> None:
    """Render per-regime metric cards in a grid."""
    section_header("Regime Statistics")

    # Sort regimes by ascending volatility (Low → Extreme → Uncertain last)
    order = ["Low Vol", "Medium Vol", "High Vol", "Extreme Vol", "Uncertain"]
    sorted_regimes = [r for r in order if r in stats] + [
        r for r in stats if r not in order
    ]

    cols = st.columns(len(sorted_regimes)) if sorted_regimes else []

    for col, regime in zip(cols, sorted_regimes):
        s = stats[regime]
        color = REGIME_COLORS.get(regime, "#64748b")
        with col:
            ret_sign = "+" if s["mean_ret"] >= 0 else ""
            card_html = metric_card(
                label=regime,
                value=f"{ret_sign}{s['mean_ret'] * 100:.1f}% ret | "
                      f"{s['mean_vol'] * 100:.1f}% vol | "
                      f"{s['pct_time']:.1f}% time",
                color=color,
                border_side="left",
            )
            st.markdown(card_html, unsafe_allow_html=True)

    # Spacer
    st.markdown(
        '<div style="height:0.5rem;"></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Confidence timeline
# ---------------------------------------------------------------------------


def render_confidence_timeline(df: pd.DataFrame, result) -> None:
    """Render a compact area chart of per-bar max-posterior confidence."""
    section_header("Confidence Timeline")

    n_labels = len(result.confidence)
    n_df = len(df)
    offset = n_df - n_labels
    labeled_dates = list(df.index)[offset:]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labeled_dates,
            y=result.confidence,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(0, 212, 255, 0.3)",
            line={"color": "#00d4ff", "width": 1.5},
            name="Confidence",
            hovertemplate="<b>%{x}</b><br>%{y:.1%}<extra></extra>",
        )
    )

    layout = get_plotly_layout()
    layout["height"] = 200
    layout["margin"] = {"l": 60, "r": 24, "t": 24, "b": 40}
    layout["xaxis"]["rangeslider"] = {"visible": False}
    layout["yaxis"]["range"] = [0, 1]
    layout["yaxis"]["tickformat"] = ".0%"
    layout["yaxis"]["title"] = "CONFIDENCE"
    layout["font"]["family"] = "'Courier New', 'IBM Plex Mono', monospace"
    layout["showlegend"] = False

    # Threshold line at 0.6
    fig.add_hline(
        y=0.6,
        line_dash="dot",
        line_color="#64748b",
        line_width=1,
        annotation_text="60%",
        annotation_font_color="#64748b",
        annotation_font_size=10,
    )

    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------


def render_welcome() -> None:
    """Show instructions when no analysis has been run yet."""
    st.markdown(
        """
<div style="
    border: 1px solid #2d3748;
    border-left: 3px solid #00d4ff;
    background: #1e2130;
    padding: 2rem 2.5rem;
    border-radius: 2px;
    margin-top: 2rem;
    max-width: 720px;
">
  <p style="
    font-family: 'Courier New', monospace;
    font-size: 0.68rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.8rem;
  ">REGIME DETECTION / READY</p>
  <h2 style="
    font-family: 'Courier New', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 0 0 1rem 0;
  ">Bloomberg Terminal — Market Regime Analyzer</h2>
  <p style="font-family:'Courier New',monospace;font-size:0.82rem;color:#94a3b8;line-height:1.7;margin-bottom:0.8rem;">
    Configure the ticker and date range in the sidebar, then press
    <span style="color:#00d4ff;font-weight:700;">▶ Run Analysis</span>
    to detect market regimes using a forward-filter Gaussian HMM.
  </p>
  <p style="font-family:'Courier New',monospace;font-size:0.78rem;color:#64748b;line-height:1.6;margin:0;">
    Regimes are ranked by ascending realized volatility:
    <span style="color:#10b981;">● Low Vol</span>
    · <span style="color:#3b82f6;">● Medium Vol</span>
    · <span style="color:#f59e0b;">● High Vol</span>
    · <span style="color:#ef4444;">● Extreme Vol</span>
  </p>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # Sidebar
    ticker, start_date, end_date, n_override, run_clicked = render_sidebar()

    # Look-ahead guard — must run before any result rendering
    if not _verify.LOOKAHEAD_CHECK_PASSED:
        st.error(
            "❌ Look-ahead bias check FAILED — forward filter integrity not verified. "
            "Results are not shown. Check core/verify.py."
        )
        st.stop()

    # ---- Run analysis when button clicked ----
    if run_clicked:
        st.session_state["run_status"] = "running"
        st.session_state.pop("regime_result", None)
        st.session_state.pop("regime_df", None)
        st.session_state.pop("regime_stats", None)
        st.session_state.pop("regime_ticker", None)

        with st.spinner("Fetching data and fitting HMM…"):
            try:
                raw_df = cached_load_ohlcv(
                    ticker,
                    start_date.isoformat(),
                    end_date.isoformat(),
                )
                # Normalise column names to lowercase for hmm_utils
                df = raw_df.rename(columns=str.lower)

                result = fit_and_filter(df, n_override=n_override)
                stats = compute_regime_stats(df, result.stable_labels)

                st.session_state["regime_result"] = result
                st.session_state["regime_df"] = df
                st.session_state["regime_stats"] = stats
                st.session_state["regime_ticker"] = ticker
                st.session_state["run_status"] = "done"

            except ValueError as exc:
                st.session_state["run_status"] = "ready"
                st.error(f"Data load failed: {exc}")
                st.stop()
            except Exception as exc:  # noqa: BLE001
                st.session_state["run_status"] = "ready"
                st.error(f"Analysis failed: {exc}")
                st.stop()

        st.rerun()

    # ---- Render results if available ----
    if "regime_result" not in st.session_state:
        render_welcome()
        return

    result = st.session_state["regime_result"]
    df: pd.DataFrame = st.session_state["regime_df"]
    stats: dict = st.session_state["regime_stats"]
    stored_ticker: str = st.session_state.get("regime_ticker", ticker)

    # Top metrics bar
    st.markdown('<div class="bloomberg-divider"></div>', unsafe_allow_html=True)
    render_top_bar(stored_ticker, result, df)
    st.markdown('<div class="bloomberg-divider"></div>', unsafe_allow_html=True)

    # Chart type toggle (above the hero chart)
    chart_type = st.radio(
        "Chart type",
        options=["Line", "Candlestick"],
        horizontal=True,
        index=0,
        key="chart_type",
        label_visibility="collapsed",
    )

    # Hero price chart
    render_hero_chart(df, result, chart_type)

    # Regime statistics
    render_regime_stats(stats)

    # Confidence timeline
    render_confidence_timeline(df, result)


main()
