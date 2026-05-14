"""
pages/7_Correlation_Breaks.py
=============================
Dashboard 7 — Correlation Break Detector ("Security Operations Center").

Design language: SOC-style. Share Tech Mono for ALL text. Near-black ``#08080c``
background. Normal cards are intentionally bland/muted; Significant and Extreme
alert cards pulse via CSS ``@keyframes`` animations to scream contrast against
the quiet baseline.

Background:    #08080c    Card bg:  #0e0e14
Normal:        #334155    Notable:  #f59e0b
Significant:   #f97316    Extreme:  #ef4444
Font:          Share Tech Mono (all text)

Data: yfinance daily, configurable lookback (default 3 years).
Algorithm: 60-day rolling correlation z-score vs historical baseline.
Alerts persisted to logs/alerts.json using atomic temp-file swap.

NO HMM — this dashboard does not use core.hmm_utils at all.
"""

from __future__ import annotations

import html as _html
import json
import os
import tempfile
from datetime import datetime, timezone
from typing import NamedTuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.design_system import REGIME_COLORS, get_plotly_layout  # noqa: F401 — project convention
from core.data import load_ohlcv, date_range_default

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Correlation Break Detector",
    page_icon="⚡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design tokens — SOC palette
# ---------------------------------------------------------------------------

_BG = "#08080c"
_CARD_BG = "#0e0e14"
_CARD_BORDER = "rgba(255,255,255,0.03)"
_NORMAL_COLOR = "#334155"
_NOTABLE_COLOR = "#f59e0b"
_SIGNIFICANT_COLOR = "#f97316"
_EXTREME_COLOR = "#ef4444"
_MUTED = "#475569"
_TEXT = "#94a3b8"
_FONT = "'Share Tech Mono', monospace"

_GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap"
)

# Severity levels mapped to colors and z-score thresholds
_SEVERITY_ORDER = ["Normal", "Notable", "Significant", "Extreme"]
_SEVERITY_COLOR: dict[str, str] = {
    "Normal":      _NORMAL_COLOR,
    "Notable":     _NOTABLE_COLOR,
    "Significant": _SIGNIFICANT_COLOR,
    "Extreme":     _EXTREME_COLOR,
}

# ---------------------------------------------------------------------------
# Global CSS — SOC style with pulse animations for alert cards
# ---------------------------------------------------------------------------

_CSS = f"""
<style>
@import url('{_GOOGLE_FONTS_URL}');

html, body, [class*="css"] {{
    background-color: {_BG} !important;
    color: {_TEXT};
    font-family: {_FONT};
}}

section[data-testid="stSidebar"] {{
    background-color: {_CARD_BG} !important;
    border-right: 1px solid rgba(255,255,255,0.04);
}}

.block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}

/* --- SOC masthead --- */
.soc-masthead {{
    font-family: {_FONT};
    font-size: 22px;
    letter-spacing: 6px;
    color: {_TEXT};
    text-transform: uppercase;
    margin: 0;
    padding: 0;
}}

.soc-subtitle {{
    font-family: {_FONT};
    font-size: 11px;
    color: {_MUTED};
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
}}

/* --- Pair status cards (top grid) --- */
.pair-card {{
    background: {_CARD_BG};
    border: 1px solid {_CARD_BORDER};
    border-left-width: 3px;
    border-radius: 3px;
    padding: 12px 14px;
    font-family: {_FONT};
    margin-bottom: 8px;
}}

.pair-name {{
    font-size: 13px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: {_TEXT};
    margin: 0 0 6px 0;
}}

.pair-metric {{
    font-size: 12px;
    color: {_MUTED};
    margin: 2px 0;
}}

.pair-metric span.val {{
    color: {_TEXT};
}}

.severity-badge {{
    display: inline-block;
    font-family: {_FONT};
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 2px;
    margin-top: 8px;
    border: 1px solid;
}}

/* --- Historical context table --- */
.ctx-card {{
    background: {_CARD_BG};
    border: 1px solid {_CARD_BORDER};
    border-left-width: 3px;
    border-radius: 3px;
    padding: 16px 18px;
    font-family: {_FONT};
    font-size: 12px;
    margin-top: 12px;
}}

.ctx-header {{
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: {_MUTED};
    margin-bottom: 12px;
}}

.ctx-row {{
    display: grid;
    grid-template-columns: 100px 80px repeat(6, 1fr);
    gap: 4px;
    padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.03);
    font-size: 11px;
    align-items: center;
}}

.ctx-row.header-row {{
    color: {_MUTED};
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding-bottom: 6px;
    margin-bottom: 2px;
}}

.ctx-row.avg-row {{
    color: {_TEXT};
    border-top: 1px solid rgba(255,255,255,0.07);
    margin-top: 4px;
    font-size: 12px;
}}

.ctx-pos {{ color: #22c55e; }}
.ctx-neg {{ color: {_EXTREME_COLOR}; }}
.ctx-neu {{ color: {_MUTED}; }}

.ctx-caveat {{
    font-size: 10px;
    color: {_MUTED};
    margin-top: 12px;
    letter-spacing: 1px;
}}

/* --- Alert log --- */
.log-row {{
    font-family: {_FONT};
    font-size: 11px;
    color: {_MUTED};
    padding: 4px 0;
    border-bottom: 1px solid rgba(255,255,255,0.02);
    display: flex;
    gap: 12px;
}}

.log-row span.lval {{
    color: {_TEXT};
}}

/* --- Pulse animations for Significant / Extreme cards --- */
@keyframes pulse-border {{
    0%   {{ border-left-color: rgba(239,68,68,0.3); }}
    50%  {{ border-left-color: rgba(239,68,68,0.9); }}
    100% {{ border-left-color: rgba(239,68,68,0.3); }}
}}

@keyframes pulse-orange {{
    0%   {{ border-left-color: rgba(249,115,22,0.3); }}
    50%  {{ border-left-color: rgba(249,115,22,0.8); }}
    100% {{ border-left-color: rgba(249,115,22,0.3); }}
}}

.card-extreme     {{ animation: pulse-border 2s ease-in-out infinite; }}
.card-significant {{ animation: pulse-orange 2.5s ease-in-out infinite; }}
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Atomic alert append — implemented locally (never import private helpers)
# ---------------------------------------------------------------------------

_LOGS_DIR = "logs"
_ALERTS_PATH = os.path.join(_LOGS_DIR, "alerts.json")


def _append_alert(record: dict) -> None:
    """Atomically append a single alert record to logs/alerts.json.

    Uses write-to-tempfile + os.replace to guarantee no partial writes
    even under concurrent Streamlit re-renders.

    Args:
        record: Dict to append. Should contain at minimum: timestamp, pair,
                z_score, severity, corr_60d.
    """
    os.makedirs(_LOGS_DIR, exist_ok=True)
    try:
        with open(_ALERTS_PATH, "r") as fh:
            data: list[dict] = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        data = []

    data.append(record)

    fd, tmp_path = tempfile.mkstemp(dir=_LOGS_DIR)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2, default=str)
        os.replace(tmp_path, _ALERTS_PATH)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class PairResult(NamedTuple):
    """Computed correlation metrics for a single asset pair."""

    pair: str
    ticker_a: str
    ticker_b: str
    corr_20d: pd.Series
    corr_60d: pd.Series
    hist_mean: float
    hist_std: float
    z_score: float
    severity: str          # "Normal" | "Notable" | "Significant" | "Extreme"
    current_corr_60d: float
    dates: pd.DatetimeIndex


# ---------------------------------------------------------------------------
# Caching wrappers around core.data.load_ohlcv
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600)
def _cached_load_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Cached yfinance download via core.data.load_ohlcv.

    Args:
        ticker: Yahoo Finance symbol.
        start:  ISO start date string.
        end:    ISO end date string.

    Returns:
        OHLCV DataFrame with Title-case columns, date-indexed.
    """
    return load_ohlcv(ticker, start, end)


@st.cache_data(ttl=3600)
def _compute_pair_result(
    pair: str,
    ticker_a: str,
    ticker_b: str,
    start: str,
    end: str,
) -> PairResult | str:
    """Download both tickers and compute rolling correlations and z-score.

    Rolling computations use only past data — no look-ahead.

    Args:
        pair:     Display label e.g. "SPY/QQQ".
        ticker_a: First ticker symbol.
        ticker_b: Second ticker symbol.
        start:    ISO start date.
        end:      ISO end date.

    Returns:
        PairResult on success, or an error string on failure.
    """
    try:
        df_a = _cached_load_ohlcv(ticker_a, start, end)
        df_b = _cached_load_ohlcv(ticker_b, start, end)
    except ValueError as exc:
        return f"Data error: {exc}"

    # Standardize to lowercase so we always access .close regardless of version
    df_a = df_a.rename(columns=str.lower)
    df_b = df_b.rename(columns=str.lower)

    # Align on common dates
    close_a = df_a["close"].rename(ticker_a)
    close_b = df_b["close"].rename(ticker_b)
    aligned = pd.concat([close_a, close_b], axis=1).dropna()

    if len(aligned) < 62:
        return f"Insufficient data for {pair} (need ≥ 62 rows, got {len(aligned)})"

    # Daily log returns — no look-ahead
    ret_a = np.log(aligned[ticker_a] / aligned[ticker_a].shift(1)).dropna()
    ret_b = np.log(aligned[ticker_b] / aligned[ticker_b].shift(1)).dropna()

    # Re-align after shift
    ret_a, ret_b = ret_a.align(ret_b, join="inner")

    # Rolling correlations — purely backward-looking
    corr_20 = ret_a.rolling(20).corr(ret_b)
    corr_60 = ret_a.rolling(60).corr(ret_b)

    # Historical mean/std of 60d rolling corr (full available history)
    valid_corr_60 = corr_60.dropna()
    if len(valid_corr_60) < 5:
        return f"Not enough rolling data for {pair}"

    hist_mean = float(valid_corr_60.mean())
    hist_std = float(valid_corr_60.std())

    if hist_std < 1e-9:
        return f"Zero std in 60d corr for {pair} — cannot compute z-score"

    current_corr_60d = float(valid_corr_60.iloc[-1])
    z = (current_corr_60d - hist_mean) / hist_std

    # Severity uses negative z (correlation lower than historical = break)
    if z > -1.5:
        severity = "Normal"
    elif z > -2.0:
        severity = "Notable"
    elif z > -2.5:
        severity = "Significant"
    else:
        severity = "Extreme"

    return PairResult(
        pair=pair,
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        corr_20d=corr_20,
        corr_60d=corr_60,
        hist_mean=hist_mean,
        hist_std=hist_std,
        z_score=z,
        severity=severity,
        current_corr_60d=current_corr_60d,
        dates=ret_a.index,
    )


@st.cache_data(ttl=3600)
def _compute_historical_context(
    pair: str,
    ticker_a: str,
    ticker_b: str,
    start: str,
    end: str,
    current_z: float,
) -> pd.DataFrame | str:
    """Find historical dates with similar z-scores and compute forward returns.

    Look-ahead note: we compute forward returns AT each historical date — this
    is valid because those dates are all in the past relative to today. We do
    NOT use future data to label the historical breaks themselves.

    Args:
        pair:      Display label.
        ticker_a:  First ticker.
        ticker_b:  Second ticker.
        start:     ISO start date.
        end:       ISO end date.
        current_z: Current z-score to find similar historical occurrences.

    Returns:
        DataFrame with columns [date, z_score, r5a, r5b, r10a, r10b, r20a, r20b],
        or an error string.
    """
    try:
        df_a = _cached_load_ohlcv(ticker_a, start, end)
        df_b = _cached_load_ohlcv(ticker_b, start, end)
    except ValueError as exc:
        return f"Data error: {exc}"

    df_a = df_a.rename(columns=str.lower)
    df_b = df_b.rename(columns=str.lower)

    close_a = df_a["close"].rename(ticker_a)
    close_b = df_b["close"].rename(ticker_b)
    aligned = pd.concat([close_a, close_b], axis=1).dropna()

    ret_a = np.log(aligned[ticker_a] / aligned[ticker_a].shift(1)).dropna()
    ret_b = np.log(aligned[ticker_b] / aligned[ticker_b].shift(1)).dropna()
    ret_a, ret_b = ret_a.align(ret_b, join="inner")

    corr_60 = ret_a.rolling(60).corr(ret_b)
    valid_corr_60 = corr_60.dropna()

    hist_mean = float(valid_corr_60.mean())
    hist_std = float(valid_corr_60.std())
    if hist_std < 1e-9:
        return "Zero std — cannot compute historical context"

    z_series = (valid_corr_60 - hist_mean) / hist_std

    # Find historical dates within ±0.5 z of current z
    similar_mask = (z_series - current_z).abs() <= 0.5
    similar_dates = z_series[similar_mask].index

    # Prices for forward return computation (from close series directly)
    price_a = aligned[ticker_a]
    price_b = aligned[ticker_b]

    rows = []
    for dt in similar_dates:
        idx = price_a.index.get_loc(dt)
        # Need 20 forward days — skip if not enough future data available
        if idx + 20 >= len(price_a):
            continue
        p0a = price_a.iloc[idx]
        p0b = price_b.iloc[idx]

        def _fwd(series: pd.Series, base: float, offset: int) -> float:
            return float(series.iloc[idx + offset] / base - 1)

        rows.append({
            "date":   dt.date(),
            "z":      round(float(z_series.iloc[z_series.index.get_loc(dt)]), 2),
            "r5a":    _fwd(price_a, p0a, 5),
            "r5b":    _fwd(price_b, p0b, 5),
            "r10a":   _fwd(price_a, p0a, 10),
            "r10b":   _fwd(price_b, p0b, 10),
            "r20a":   _fwd(price_a, p0a, 20),
            "r20b":   _fwd(price_b, p0b, 20),
        })

    if not rows:
        return "No historical dates found with similar z-score"

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------


def _severity_css_class(severity: str) -> str:
    """Return the animation CSS class name for a given severity level.

    Args:
        severity: One of Normal, Notable, Significant, Extreme.

    Returns:
        CSS class string — empty for Normal/Notable, pulse class for alerts.
    """
    if severity == "Extreme":
        return "card-extreme"
    if severity == "Significant":
        return "card-significant"
    return ""


def _pair_card_html(result: PairResult) -> str:
    """Render a compact SOC status card for a single pair.

    Args:
        result: Computed PairResult containing metrics and severity.

    Returns:
        HTML string for the pair card, ready for st.markdown.
    """
    color = _SEVERITY_COLOR[result.severity]
    anim_class = _severity_css_class(result.severity)
    pair_escaped = _html.escape(result.pair)

    badge_style = (
        f"color:{color};"
        f"border-color:{color};"
        "background:transparent;"
    )

    return f"""
<div class="pair-card {anim_class}" style="border-left-color:{color};">
  <p class="pair-name">{pair_escaped}</p>
  <p class="pair-metric">60d corr &nbsp;<span class="val">{result.current_corr_60d:+.3f}</span></p>
  <p class="pair-metric">z-score &nbsp;<span class="val">{result.z_score:+.2f}</span></p>
  <p class="pair-metric">μ {result.hist_mean:+.3f} &nbsp; σ {result.hist_std:.3f}</p>
  <span class="severity-badge" style="{badge_style}">{_html.escape(result.severity)}</span>
</div>
"""


def _fmt_return(val: float) -> str:
    """Format a return value as a colored HTML span.

    Args:
        val: Fractional return (e.g. 0.05 = 5%).

    Returns:
        HTML <span> string with contextual color class.
    """
    pct = val * 100
    css_class = "ctx-pos" if pct > 0.05 else ("ctx-neg" if pct < -0.05 else "ctx-neu")
    sign = "+" if pct > 0 else ""
    return f'<span class="{css_class}">{sign}{pct:.1f}%</span>'


def _historical_context_html(
    ctx_df: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    severity: str,
) -> str:
    """Render the historical context table as an HTML card.

    Args:
        ctx_df:   DataFrame from _compute_historical_context.
        ticker_a: Name for first ticker column header.
        ticker_b: Name for second ticker column header.
        severity: Severity string to determine border color.

    Returns:
        HTML string for the context card.
    """
    color = _SEVERITY_COLOR[severity]
    ta = _html.escape(ticker_a)
    tb = _html.escape(ticker_b)

    header_row = f"""
<div class="ctx-row header-row">
  <span>DATE</span><span>Z</span>
  <span>+5d {ta}</span><span>+5d {tb}</span>
  <span>+10d {ta}</span><span>+10d {tb}</span>
  <span>+20d {ta}</span><span>+20d {tb}</span>
</div>"""

    data_rows = ""
    for _, row in ctx_df.iterrows():
        data_rows += f"""
<div class="ctx-row">
  <span>{_html.escape(str(row['date']))}</span>
  <span>{row['z']:+.2f}</span>
  {_fmt_return(row['r5a'])}
  {_fmt_return(row['r5b'])}
  {_fmt_return(row['r10a'])}
  {_fmt_return(row['r10b'])}
  {_fmt_return(row['r20a'])}
  {_fmt_return(row['r20b'])}
</div>"""

    # Averages row
    avg = ctx_df[["r5a", "r5b", "r10a", "r10b", "r20a", "r20b"]].mean()
    avg_row = f"""
<div class="ctx-row avg-row">
  <span>AVG ({len(ctx_df)})</span><span>—</span>
  {_fmt_return(avg['r5a'])}
  {_fmt_return(avg['r5b'])}
  {_fmt_return(avg['r10a'])}
  {_fmt_return(avg['r10b'])}
  {_fmt_return(avg['r20a'])}
  {_fmt_return(avg['r20b'])}
</div>"""

    return f"""
<div class="ctx-card" style="border-left-color:{color};">
  <p class="ctx-header">Historical Context — Similar Z-Score Periods (±0.5)</p>
  {header_row}
  {data_rows}
  {avg_row}
  <p class="ctx-caveat">Past correlation breaks do not guarantee future outcomes.</p>
</div>
"""


# ---------------------------------------------------------------------------
# Plotly chart builders
# ---------------------------------------------------------------------------


def _main_correlation_chart(result: PairResult) -> go.Figure:
    """Build the main 60-day rolling correlation chart with SOC styling.

    Includes mean line, ±2σ dashed lines, area fill below −2σ, and
    shaded columns for historical break periods (z < -1.5).

    Args:
        result: PairResult containing corr_60d series and stats.

    Returns:
        Plotly Figure object.
    """
    layout = get_plotly_layout(theme="dark")
    layout["paper_bgcolor"] = _BG
    layout["plot_bgcolor"] = _BG
    layout["xaxis"]["gridcolor"] = "rgba(255,255,255,0.02)"
    layout["yaxis"]["gridcolor"] = "rgba(255,255,255,0.02)"
    layout["font"]["family"] = _FONT
    layout["margin"] = {"l": 50, "r": 20, "t": 40, "b": 40}

    corr = result.corr_60d.dropna()
    dates = corr.index
    mu = result.hist_mean
    sigma = result.hist_std
    upper_2s = mu + 2 * sigma
    lower_2s = mu - 2 * sigma

    fig = go.Figure()

    # Shaded break-period columns (z < -1.5 → corr < mu - 1.5*sigma)
    break_threshold = mu - 1.5 * sigma
    in_break = False
    break_start = None
    shapes = []
    for i, (dt, val) in enumerate(corr.items()):
        if val < break_threshold and not in_break:
            in_break = True
            break_start = dt
        elif val >= break_threshold and in_break:
            in_break = False
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=break_start,
                    x1=dt,
                    y0=0,
                    y1=1,
                    fillcolor="rgba(239,68,68,0.04)",
                    line_width=0,
                    layer="below",
                )
            )
    if in_break:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="paper",
                x0=break_start,
                x1=dates[-1],
                y0=0,
                y1=1,
                fillcolor="rgba(239,68,68,0.04)",
                line_width=0,
                layer="below",
            )
        )

    # Area fill below −2σ threshold
    fill_y = corr.clip(upper=lower_2s)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=[lower_2s] * len(dates),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=fill_y.values,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(239,68,68,0.08)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Main 60d correlation line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=corr.values,
            mode="lines",
            name="60d Rolling Corr",
            line=dict(color="#94a3b8", width=1.5),
        )
    )

    # Historical mean
    fig.add_hline(
        y=mu,
        line=dict(color="#334155", width=1, dash="dash"),
        annotation_text=f"μ={mu:.3f}",
        annotation_font_color="#475569",
        annotation_font_size=10,
    )

    # +2σ and −2σ threshold lines
    for y_val, label in [(upper_2s, "+2σ"), (lower_2s, "−2σ")]:
        fig.add_hline(
            y=y_val,
            line=dict(color="rgba(239,68,68,0.30)", width=1, dash="dash"),
            annotation_text=label,
            annotation_font_color="rgba(239,68,68,0.6)",
            annotation_font_size=10,
        )

    layout["shapes"] = shapes
    layout["title"] = dict(
        text=f"{_html.escape(result.pair)} — 60-Day Rolling Correlation",
        font=dict(family=_FONT, size=13, color="#94a3b8"),
        x=0,
        xanchor="left",
    )
    fig.update_layout(**layout)
    return fig


def _mini_chart(series: pd.Series, title: str) -> go.Figure:
    """Build a compact mini chart for 20d or 60d rolling correlation.

    Args:
        series: Rolling correlation pd.Series (may contain NaNs).
        title:  Chart title string.

    Returns:
        Plotly Figure with minimal decoration.
    """
    layout = get_plotly_layout(theme="dark")
    layout["paper_bgcolor"] = _BG
    layout["plot_bgcolor"] = _BG
    layout["xaxis"]["gridcolor"] = "rgba(255,255,255,0.02)"
    layout["yaxis"]["gridcolor"] = "rgba(255,255,255,0.02)"
    layout["font"]["family"] = _FONT
    layout["margin"] = {"l": 40, "r": 10, "t": 30, "b": 30}

    clean = series.dropna()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=clean.index,
            y=clean.values,
            mode="lines",
            line=dict(color="#94a3b8", width=1),
            showlegend=False,
        )
    )
    layout["title"] = dict(
        text=title,
        font=dict(family=_FONT, size=11, color="#475569"),
        x=0,
        xanchor="left",
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Alert log renderer
# ---------------------------------------------------------------------------


def _render_alert_log() -> None:
    """Render the alert log expander from logs/alerts.json."""
    with st.expander("Alert Log", expanded=False):
        try:
            with open(_ALERTS_PATH, "r") as fh:
                alerts: list[dict] = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError):
            alerts = []

        if not alerts:
            st.markdown(
                f'<p style="font-family:{_FONT};font-size:11px;color:{_MUTED};">'
                "No alerts logged.</p>",
                unsafe_allow_html=True,
            )
            return

        for alert in reversed(alerts[-50:]):  # most recent first, cap at 50
            ts = alert.get("timestamp", "—")
            pair = _html.escape(str(alert.get("pair", "—")))
            z = alert.get("z_score", 0.0)
            sev = alert.get("severity", "—")
            corr = alert.get("corr_60d", 0.0)
            sev_color = _SEVERITY_COLOR.get(sev, _MUTED)
            st.markdown(
                f'<div class="log-row">'
                f'<span style="color:{_MUTED};min-width:160px;">{_html.escape(str(ts)[:19])}</span>'
                f'<span class="lval" style="min-width:120px;">{pair}</span>'
                f'<span style="color:{_MUTED};">z=</span><span class="lval">{z:+.2f}</span>'
                f'&nbsp;&nbsp;<span style="color:{_MUTED};">corr=</span><span class="lval">{corr:+.3f}</span>'
                f'&nbsp;&nbsp;<span style="color:{sev_color};letter-spacing:2px;">{_html.escape(sev)}</span>'
                f"</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def _render_sidebar() -> tuple[list[str], int]:
    """Render the SOC dashboard sidebar and return user inputs.

    Returns:
        Tuple of (pairs_list, lookback_years).
    """
    st.sidebar.markdown(
        f'<p style="font-family:{_FONT};font-size:10px;letter-spacing:3px;'
        f'text-transform:uppercase;color:{_MUTED};margin-bottom:8px;">⚡ SOC PANEL</p>',
        unsafe_allow_html=True,
    )

    pairs_input = st.sidebar.text_area(
        "Asset Pairs (one per line, slash-separated)",
        value="SPY/QQQ\nGLD/TLT\nSPY/IWM\nBTC-USD/ETH-USD\nSPY/EEM",
        height=140,
        help="Format: TICKER_A/TICKER_B",
    )

    lookback_years = st.sidebar.slider(
        "Lookback (years)",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
    )

    run = st.sidebar.button("Check Correlations", type="primary", use_container_width=True)

    # Parse pairs from text input
    raw_lines = [ln.strip() for ln in pairs_input.strip().splitlines() if ln.strip()]
    pairs: list[str] = []
    for ln in raw_lines:
        if "/" in ln:
            parts = ln.split("/", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                pairs.append(f"{parts[0].strip()}/{parts[1].strip()}")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f'<p style="font-family:{_FONT};font-size:10px;color:{_MUTED};">Dashboard 7 / 7</p>',
        unsafe_allow_html=True,
    )

    if not run and "cb_results" not in st.session_state:
        return [], lookback_years

    return pairs, lookback_years


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the Correlation Break Detector dashboard."""

    # --- Masthead ---
    st.markdown(
        f'<h1 class="soc-masthead">⚡ Correlation Break Detector</h1>'
        f'<p class="soc-subtitle">Security Operations Center — Regime Break Surveillance</p>',
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.04);margin:8px 0 20px 0;'>", unsafe_allow_html=True)

    # --- Sidebar ---
    pairs, lookback_years = _render_sidebar()

    # Check if user pressed the button
    run_pressed = st.sidebar.button(
        "↻ Re-check",
        key="recheck_btn",
        help="Force a fresh computation (bypasses cache)",
        use_container_width=True,
    )
    _ = run_pressed  # re-render triggers recomputation via Streamlit's natural rerun

    if not pairs:
        st.markdown(
            f'<p style="font-family:{_FONT};font-size:13px;color:{_MUTED};text-align:center;margin-top:60px;">'
            "Configure pairs in the sidebar and press [ Check Correlations ] to begin.</p>",
            unsafe_allow_html=True,
        )
        _render_alert_log()
        return

    start, end = date_range_default(years=lookback_years)

    # --- Compute all pairs ---
    results: list[PairResult | str] = []
    status_placeholder = st.empty()

    with st.spinner("Computing correlations..."):
        for pair_str in pairs:
            parts = pair_str.split("/", 1)
            ta, tb = parts[0].strip(), parts[1].strip()
            res = _compute_pair_result(pair_str, ta, tb, start, end)
            results.append(res)

    status_placeholder.empty()

    # --- Persist alerts to logs/alerts.json (once per page run, not re-render) ---
    run_key = f"alerts_saved_{start}_{end}_{'_'.join(pairs)}"
    if run_key not in st.session_state:
        st.session_state[run_key] = True
        for res in results:
            if isinstance(res, PairResult) and res.severity != "Normal":
                _append_alert(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "pair": res.pair,
                        "z_score": round(res.z_score, 4),
                        "severity": res.severity,
                        "corr_60d": round(res.current_corr_60d, 4),
                    }
                )

    # --- Pair status grid (top) ---
    st.markdown(
        f'<p style="font-family:{_FONT};font-size:10px;letter-spacing:3px;'
        f'text-transform:uppercase;color:{_MUTED};margin-bottom:6px;">STATUS GRID</p>',
        unsafe_allow_html=True,
    )

    grid_cols = st.columns(len(results))
    valid_results: list[PairResult] = []

    for col, res in zip(grid_cols, results):
        with col:
            if isinstance(res, str):
                st.markdown(
                    f'<div class="pair-card" style="border-left-color:{_MUTED};">'
                    f'<p class="pair-name">{_html.escape(pairs[results.index(res)])}</p>'
                    f'<p class="pair-metric" style="color:{_EXTREME_COLOR};">ERROR</p>'
                    f'<p style="font-size:10px;color:{_MUTED};">{_html.escape(res)}</p>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(_pair_card_html(res), unsafe_allow_html=True)
                valid_results.append(res)

    if not valid_results:
        st.error("All pairs failed to load. Check tickers and network connectivity.")
        _render_alert_log()
        return

    # --- Pair selector for detail view ---
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

    valid_pair_labels = [r.pair for r in valid_results]
    # Default to the most severe pair
    severity_rank = {s: i for i, s in enumerate(_SEVERITY_ORDER)}
    sorted_by_severity = sorted(
        valid_results,
        key=lambda r: severity_rank.get(r.severity, 0),
        reverse=True,
    )
    default_pair = sorted_by_severity[0].pair

    selected_pair_label = st.selectbox(
        "Select pair for detail view",
        options=valid_pair_labels,
        index=valid_pair_labels.index(default_pair),
        key="selected_pair",
    )
    selected = next(r for r in valid_results if r.pair == selected_pair_label)

    # --- Main correlation chart ---
    st.markdown(
        f'<p style="font-family:{_FONT};font-size:10px;letter-spacing:3px;'
        f'text-transform:uppercase;color:{_MUTED};margin:16px 0 6px 0;">CORRELATION TIMELINE</p>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        _main_correlation_chart(selected),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    # --- Two small side-by-side mini charts ---
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(
            _mini_chart(selected.corr_20d, "20-Day Rolling Correlation"),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    with col_right:
        st.plotly_chart(
            _mini_chart(selected.corr_60d, "60-Day Rolling Correlation"),
            use_container_width=True,
            config={"displayModeBar": False},
        )

    # --- Historical context (only for Notable or worse) ---
    if selected.severity != "Normal":
        ctx = _compute_historical_context(
            pair=selected.pair,
            ticker_a=selected.ticker_a,
            ticker_b=selected.ticker_b,
            start=start,
            end=end,
            current_z=selected.z_score,
        )
        if isinstance(ctx, str):
            st.markdown(
                f'<p style="font-family:{_FONT};font-size:12px;color:{_MUTED};">'
                f"Historical context: {_html.escape(ctx)}</p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                _historical_context_html(ctx, selected.ticker_a, selected.ticker_b, selected.severity),
                unsafe_allow_html=True,
            )

    # --- Alert log ---
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    _render_alert_log()


if __name__ == "__main__":
    main()
else:
    # Streamlit runs the module top-level; call main() outside __main__ guard
    main()
