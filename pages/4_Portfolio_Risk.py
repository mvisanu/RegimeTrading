"""
pages/4_Portfolio_Risk.py — Premium Fintech Portfolio Risk Dashboard.

Design language: premium fintech — gradient cards, Plus Jakarta Sans + JetBrains Mono,
indigo primary (#6366f1), hover-lift cards, subtle glow on the portfolio total.

Displays:
  - Top stats bar: total portfolio value, total P&L, summary (positions / regimes / market)
  - Left column (60%): position cards with regime badge, P&L bar, entry→current price
  - Right column (40%): correlation heatmap, stress tests, watchlist

Sidebar: Run Analysis button, portfolio editor (text area + CSV upload), watchlist tickers.
"""

from __future__ import annotations

import html as _html
import io
import logging
import sys
import os
from datetime import datetime, date
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# --- Must be the FIRST Streamlit call ---
st.set_page_config(
    page_title="Portfolio Risk",
    page_icon="💼",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------
# Add project root to path when running from pages/ subfolder
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import load_ohlcv, date_range_default
from core.design_system import REGIME_COLORS, get_plotly_layout, regime_badge
from core.hmm_utils import fit_and_filter, RegimeResult
from core import verify as _verify

# ---------------------------------------------------------------------------
# Design tokens — Premium Fintech palette (scoped to this page only)
# ---------------------------------------------------------------------------
_BG = "#0e1016"
_CARD_BG_FROM = "#161923"
_CARD_BG_TO = "#1a1e2e"
_CARD_BORDER = "rgba(255,255,255,0.04)"
_PRIMARY = "#6366f1"       # indigo
_SUCCESS = "#10b981"       # teal/emerald
_DANGER = "#f43f5e"        # rose
_WARNING = "#f59e0b"       # amber
_TEXT_PRIMARY = "#f1f5f9"
_TEXT_LABEL = "#94a3b8"
_CARD_RADIUS = "16px"
_CARD_SHADOW = "0 4px 24px rgba(0,0,0,0.3)"
_HOVER_BORDER = "rgba(99,102,241,0.2)"

# ---------------------------------------------------------------------------
# Default portfolio data
# ---------------------------------------------------------------------------
_DEFAULT_POSITIONS_CSV = """ticker,shares,entry,current
SPY,100,540,558
QQQ,50,480,495
AAPL,75,210,218
GLD,40,235,242
TLT,60,88,85
"""

_TICKER_NAMES: dict[str, str] = {
    "SPY": "SPDR S&P 500 ETF",
    "QQQ": "Invesco QQQ Trust",
    "AAPL": "Apple Inc.",
    "GLD": "SPDR Gold Shares",
    "TLT": "iShares 20Y Treasury",
    "IWM": "iShares Russell 2000",
    "EEM": "iShares MSCI Emerging",
    "VXX": "iPath VIX Short-Term",
    "XLF": "Financial Select Sector",
    "XLE": "Energy Select Sector",
}

_STRESS_SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "2008 Crisis",
        "returns": {"SPY": -0.56, "QQQ": -0.54, "AAPL": -0.61, "GLD": 0.21, "TLT": 0.33},
    },
    {
        "name": "2020 COVID",
        "returns": {"SPY": -0.34, "QQQ": -0.28, "AAPL": -0.31, "GLD": -0.03, "TLT": 0.21},
    },
    {
        "name": "2022 Bear",
        "returns": {"SPY": -0.25, "QQQ": -0.33, "AAPL": -0.30, "GLD": -0.04, "TLT": -0.31},
    },
]

# ---------------------------------------------------------------------------
# CSS injection — Premium Fintech (scoped to this page)
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Page base */
.stApp {
    background-color: #0e1016;
    color: #f1f5f9;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.stSidebar {
    background-color: #11141e !important;
    border-right: 1px solid rgba(255,255,255,0.05);
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1rem;
}

.stSidebar label,
.stSidebar .stMarkdown p {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: #94a3b8 !important;
    font-size: 0.82rem !important;
}

/* Inputs */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background-color: #161923;
    border: 1px solid rgba(255,255,255,0.08);
    color: #f1f5f9;
    font-family: 'JetBrains Mono', monospace;
    border-radius: 8px;
    font-size: 0.82rem;
}

/* Run button */
.stButton > button {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(99,102,241,0.08));
    border: 1px solid rgba(99,102,241,0.45);
    color: #a5b4fc;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.5rem 1.4rem;
    letter-spacing: 0.02em;
    width: 100%;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(99,102,241,0.15));
    border-color: #6366f1;
    color: #ffffff;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(99,102,241,0.25);
}

/* Top stats bar */
.top-stats-bar {
    display: flex;
    gap: 1rem;
    margin-bottom: 1.5rem;
    align-items: stretch;
}

.stat-card {
    background: linear-gradient(135deg, #161923, #1a1e2e);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 16px;
    padding: 1.4rem 1.8rem;
    flex: 1;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: border-color 0.2s ease, transform 0.2s ease;
}

.stat-card:hover {
    border-color: rgba(99,102,241,0.2);
    transform: translateY(-2px);
}

.stat-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 500;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.5rem;
}

.stat-value-large {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3rem;
    font-weight: 500;
    color: #f1f5f9;
    line-height: 1;
    text-shadow: 0 0 40px rgba(99,102,241,0.4);
    letter-spacing: -0.02em;
}

.stat-value-pnl {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.2rem;
    font-weight: 500;
    line-height: 1;
    letter-spacing: -0.02em;
}

.stat-value-summary {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.95rem;
    font-weight: 500;
    color: #cbd5e1;
    line-height: 1.6;
}

/* Position cards */
.position-card {
    background: linear-gradient(135deg, #161923, #1a1e2e);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
    transition: border-color 0.2s ease, transform 0.2s ease;
}

.position-card:hover {
    border-color: rgba(99,102,241,0.2);
    transform: translateY(-2px);
}

.position-ticker {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.1rem;
}

.position-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.72rem;
    color: #64748b;
    margin-bottom: 0.6rem;
}

.position-price-row {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #94a3b8;
    margin: 0.35rem 0;
}

.position-price-current {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1rem;
    font-weight: 500;
    color: #f1f5f9;
}

.position-pnl-positive {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 500;
    color: #10b981;
}

.position-pnl-negative {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 500;
    color: #f43f5e;
}

/* P&L bar */
.pnl-bar-container {
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    margin-top: 0.5rem;
    overflow: hidden;
    position: relative;
}

.pnl-bar-positive {
    height: 100%;
    border-radius: 2px;
    background: #10b981;
    position: absolute;
    left: 0;
}

.pnl-bar-negative {
    height: 100%;
    border-radius: 2px;
    background: #f43f5e;
    position: absolute;
    right: 0;
}

/* Regime info row */
.regime-info-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 0.4rem 0;
    flex-wrap: wrap;
}

.days-in-regime {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #64748b;
}

/* Section headers (fintech style) */
.section-header-fintech {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.72rem;
    font-weight: 600;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.6rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(99,102,241,0.2);
}

/* Right panel sub-cards */
.right-card {
    background: linear-gradient(135deg, #161923, #1a1e2e);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}

/* Stress test rows */
.stress-row-safe {
    background: rgba(16,185,129,0.06);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem;
    border-left: 3px solid #10b981;
}

.stress-row-amber {
    background: rgba(245,158,11,0.06);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem;
    border-left: 3px solid #f59e0b;
}

.stress-row-danger {
    background: rgba(244,63,94,0.07);
    border-radius: 8px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem;
    border-left: 3px solid #f43f5e;
}

.stress-scenario-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    color: #cbd5e1;
    margin-bottom: 0.2rem;
}

.stress-loss-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.2rem;
    font-weight: 500;
}

.stress-bar-container {
    height: 3px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    margin-top: 0.3rem;
    overflow: hidden;
}

/* Watchlist rows */
.watchlist-row {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}

.watchlist-ticker {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 500;
    color: #f1f5f9;
    min-width: 42px;
}

.watchlist-price {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #94a3b8;
    min-width: 54px;
    text-align: right;
}

.watchlist-conf-bar {
    flex: 1;
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
}

/* Divider */
.pf-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin: 0.5rem 0;
}

/* Spinner override */
.stSpinner > div {
    border-top-color: #6366f1 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Helper: parse positions CSV / text
# ---------------------------------------------------------------------------

def parse_positions(raw: str) -> pd.DataFrame:
    """Parse a CSV string into a positions DataFrame.

    Parameters
    ----------
    raw:
        CSV text with columns: ticker, shares, entry, current.

    Returns
    -------
    pd.DataFrame with columns: ticker (str), shares (int), entry (float), current (float).
    Skips rows that cannot be parsed.
    """
    try:
        df = pd.read_csv(io.StringIO(raw.strip()))
        df.columns = [c.strip().lower() for c in df.columns]
        required = {"ticker", "shares", "entry", "current"}
        if not required.issubset(set(df.columns)):
            st.error(f"Portfolio CSV must have columns: {sorted(required)}")
            return pd.DataFrame(columns=list(required))
        df["ticker"] = df["ticker"].str.strip().str.upper()
        df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
        df["entry"] = pd.to_numeric(df["entry"], errors="coerce")
        df["current"] = pd.to_numeric(df["current"], errors="coerce")
        df = df.dropna(subset=["entry", "current"])
        df = df[df["shares"] > 0]
        return df.reset_index(drop=True)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to parse portfolio: {exc}")
        return pd.DataFrame(columns=["ticker", "shares", "entry", "current"])


# ---------------------------------------------------------------------------
# Helper: market open/closed check (Eastern Time)
# ---------------------------------------------------------------------------

def is_market_open() -> bool:
    """Return True if the NYSE is currently open (Mon–Fri, 09:30–16:00 ET)."""
    et = ZoneInfo("America/New_York")
    now = datetime.now(et)
    if now.weekday() >= 5:
        return False
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now < market_close


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def cached_load_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV and normalise column names to lowercase."""
    df = load_ohlcv(ticker, start, end)
    df.columns = [c.lower() for c in df.columns]
    return df


@st.cache_data(ttl=3600)
def cached_fit_regime(ticker: str, start: str, end: str) -> dict[str, Any]:
    """Fit HMM on 1y of data and return serialisable regime info for one ticker.

    Returns
    -------
    dict with keys: regime (str), confidence (float), days_in_regime (int).
    On error returns regime="Uncertain", confidence=0.5, days_in_regime=0.
    """
    try:
        df = cached_load_ohlcv(ticker, start, end)
        if len(df) < 30:
            logging.getLogger(__name__).warning(
                "cached_fit_regime(%s): insufficient data (%d rows)", ticker, len(df)
            )
            return {"regime": "Uncertain", "confidence": 0.5, "days_in_regime": 0}
        result: RegimeResult = fit_and_filter(df)
        labels = result.stable_labels
        conf_arr = result.confidence

        # Latest regime
        latest_regime = labels[-1] if labels else "Uncertain"
        latest_conf = float(conf_arr[-1]) if len(conf_arr) > 0 else 0.5

        # Days in current regime: count trailing consecutive identical labels
        days = 0
        for lbl in reversed(labels):
            if lbl == latest_regime:
                days += 1
            else:
                break

        return {
            "regime": latest_regime,
            "confidence": latest_conf,
            "days_in_regime": days,
        }
    except Exception:  # noqa: BLE001
        logging.getLogger(__name__).warning(
            "cached_fit_regime(%s): HMM fit failed, returning Uncertain", ticker
        )
        return {"regime": "Uncertain", "confidence": 0.5, "days_in_regime": 0}


@st.cache_data(ttl=3600)
def cached_latest_price(ticker: str, start: str, end: str) -> float | None:
    """Return the most recent closing price for a ticker, or None on failure."""
    try:
        df = cached_load_ohlcv(ticker, start, end)
        return float(df["close"].iloc[-1])
    except Exception:  # noqa: BLE001
        return None


@st.cache_data(ttl=3600)
def cached_correlation_matrix(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Compute 60-day rolling correlation matrix from daily returns.

    Parameters
    ----------
    tickers:
        List of ticker symbols.
    start, end:
        Date range strings (ISO 8601).

    Returns
    -------
    pd.DataFrame — correlation matrix, rows/cols indexed by ticker.
    Returns identity matrix of appropriate size on failure.
    """
    try:
        closes: dict[str, pd.Series] = {}
        for ticker in tickers:
            df = cached_load_ohlcv(ticker, start, end)
            closes[ticker] = df["close"]

        price_df = pd.DataFrame(closes)
        returns = price_df.pct_change().dropna()

        # Use the last 60 trading days
        recent = returns.tail(60)
        corr = recent.corr()
        return corr
    except Exception:  # noqa: BLE001
        return pd.DataFrame(
            np.eye(len(tickers)),
            index=tickers,
            columns=tickers,
        )


# ---------------------------------------------------------------------------
# HTML component builders
# ---------------------------------------------------------------------------

def _pnl_bar_html(pnl_pct: float) -> str:
    """Return a 4px-tall P&L bar HTML. Positive extends right, negative extends left."""
    max_scale = 20.0  # cap at ±20 % for visual scale
    clamped = max(-max_scale, min(max_scale, pnl_pct * 100))
    width_pct = min(abs(clamped) / max_scale * 100, 100)

    if clamped >= 0:
        bar_html = (
            f'<div class="pnl-bar-positive" '
            f'style="width:{width_pct:.1f}%;"></div>'
        )
    else:
        bar_html = (
            f'<div class="pnl-bar-negative" '
            f'style="width:{width_pct:.1f}%;"></div>'
        )
    return f'<div class="pnl-bar-container">{bar_html}</div>'


def _position_card_html(
    ticker: str,
    shares: int,
    entry: float,
    current: float,
    regime: str,
    confidence: float,
    days_in_regime: int,
) -> str:
    """Return the full HTML for a position card."""
    name = _TICKER_NAMES.get(ticker, ticker)
    ticker = _html.escape(ticker)
    name = _html.escape(name)
    pnl_dollar = shares * (current - entry)
    pnl_pct = (current - entry) / entry

    pnl_sign = "+" if pnl_dollar >= 0 else ""
    pnl_class = "position-pnl-positive" if pnl_dollar >= 0 else "position-pnl-negative"
    badge_html = regime_badge(regime, confidence, glow=True)
    bar_html = _pnl_bar_html(pnl_pct)

    mkt_value = shares * current

    return f"""
<div class="position-card">
  <div style="display:flex; justify-content:space-between; align-items:flex-start;">
    <div>
      <div class="position-ticker">{ticker}</div>
      <div class="position-name">{name}</div>
    </div>
    <div style="text-align:right;">
      <div class="position-price-current">${current:,.2f}</div>
      <div class="days-in-regime">{shares:,} shares &nbsp;·&nbsp; ${mkt_value:,.0f}</div>
    </div>
  </div>
  <div class="regime-info-row">
    {badge_html}
    <span class="days-in-regime">{days_in_regime}d in regime</span>
  </div>
  <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.4rem;">
    <span class="position-price-row">${entry:,.2f} &nbsp;→&nbsp; <span class="position-price-current">${current:,.2f}</span></span>
    <span class="{pnl_class}">{pnl_sign}${pnl_dollar:,.0f} &nbsp;({pnl_sign}{pnl_pct*100:.1f}%)</span>
  </div>
  {bar_html}
</div>
"""


def _stress_row_html(
    scenario_name: str,
    portfolio_loss: float,
    loss_pct: float,
    max_abs_loss: float,
) -> str:
    """Return HTML for one stress test row."""
    abs_loss_pct = abs(loss_pct * 100)

    if abs_loss_pct < 10:
        row_class = "stress-row-safe"
        color = _SUCCESS
    elif abs_loss_pct < 20:
        row_class = "stress-row-amber"
        color = _WARNING
    else:
        row_class = "stress-row-danger"
        color = _DANGER

    if portfolio_loss >= 0:
        loss_display = f'+${portfolio_loss:,.0f}'
    else:
        loss_display = f'−${abs(portfolio_loss):,.0f}'

    # damage bar width proportional to abs loss vs max scenario loss
    bar_width = min(abs(portfolio_loss) / (max_abs_loss + 1) * 100, 100)

    loss_pct_display = f"{'+' if loss_pct >= 0 else ''}{loss_pct*100:.1f}%"

    return f"""
<div class="{row_class}">
  <div style="display:flex; justify-content:space-between; align-items:center;">
    <div>
      <div class="stress-scenario-name">{scenario_name}</div>
      <div class="stress-bar-container">
        <div style="height:100%; width:{bar_width:.1f}%; background:{color}; border-radius:2px;"></div>
      </div>
    </div>
    <div style="text-align:right;">
      <div class="stress-loss-value" style="color:{color};">{loss_display}</div>
      <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#64748b;">{loss_pct_display}</div>
    </div>
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Plotly: correlation heatmap
# ---------------------------------------------------------------------------

def _build_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Build a Plotly correlation heatmap with fintech color scale."""
    tickers = list(corr.columns)
    z = corr.values

    # Annotations — show value in each cell; flag high-correlation pairs
    annotations = []
    shapes = []
    n = len(tickers)

    for i in range(n):
        for j in range(n):
            val = z[i, j]
            annotations.append(
                dict(
                    x=j,
                    y=i,
                    text=f"{val:.2f}",
                    showarrow=False,
                    font=dict(
                        color="#ffffff" if abs(val) > 0.5 else "#94a3b8",
                        family="JetBrains Mono, monospace",
                        size=10,
                    ),
                )
            )
            # Rose border for high correlation (excluding diagonal)
            if i != j and abs(val) > 0.85:
                shapes.append(
                    dict(
                        type="rect",
                        x0=j - 0.5,
                        y0=i - 0.5,
                        x1=j + 0.5,
                        y1=i + 0.5,
                        line=dict(color="#f43f5e", width=2),
                        fillcolor="rgba(0,0,0,0)",
                    )
                )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=tickers,
            y=tickers,
            colorscale=[[0, "#0d1b3e"], [0.5, "#6366f1"], [1, "#ffffff"]],
            zmin=-1,
            zmax=1,
            showscale=True,
            colorbar=dict(
                thickness=10,
                len=0.8,
                tickfont=dict(
                    family="JetBrains Mono, monospace",
                    color="#94a3b8",
                    size=9,
                ),
                tickvals=[-1, -0.5, 0, 0.5, 1],
            ),
        )
    )

    layout = get_plotly_layout(theme="dark")
    layout.update(
        {
            "paper_bgcolor": _CARD_BG_FROM,
            "plot_bgcolor": _CARD_BG_FROM,
            "annotations": annotations,
            "shapes": shapes,
            "height": 260,
            "margin": {"l": 40, "r": 40, "t": 20, "b": 40},
            "xaxis": {
                **layout.get("xaxis", {}),
                "showgrid": False,
                "tickfont": dict(family="JetBrains Mono, monospace", color="#94a3b8", size=10),
            },
            "yaxis": {
                **layout.get("yaxis", {}),
                "showgrid": False,
                "autorange": "reversed",
                "tickfont": dict(family="JetBrains Mono, monospace", color="#94a3b8", size=10),
            },
        }
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Compute derived portfolio metrics
# ---------------------------------------------------------------------------

def compute_portfolio_metrics(positions: pd.DataFrame) -> dict[str, float]:
    """Return total_value, total_pnl, total_pnl_pct for the portfolio.

    Parameters
    ----------
    positions:
        DataFrame with columns: ticker, shares, entry, current.

    Returns
    -------
    dict with keys: total_value, total_cost, total_pnl, total_pnl_pct.
    """
    positions = positions.copy()
    positions["market_value"] = positions["shares"] * positions["current"]
    positions["cost_basis"] = positions["shares"] * positions["entry"]
    total_value = float(positions["market_value"].sum())
    total_cost = float(positions["cost_basis"].sum())
    total_pnl = total_value - total_cost
    total_pnl_pct = total_pnl / total_cost if total_cost > 0 else 0.0
    return {
        "total_value": total_value,
        "total_cost": total_cost,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
    }


def compute_stress_results(
    positions: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Apply hardcoded stress scenarios to the current portfolio.

    Parameters
    ----------
    positions:
        DataFrame with columns: ticker, shares, current.

    Returns
    -------
    list of dicts: name, portfolio_loss, loss_pct, abs_loss.
    """
    total_value = float((positions["shares"] * positions["current"]).sum())
    results = []
    for scenario in _STRESS_SCENARIOS:
        loss = 0.0
        for _, row in positions.iterrows():
            ticker = row["ticker"]
            drawdown = scenario["returns"].get(ticker, 0.0)
            loss += row["shares"] * row["current"] * drawdown
        loss_pct = loss / total_value if total_value > 0 else 0.0
        results.append(
            {
                "name": scenario["name"],
                "portfolio_loss": loss,
                "loss_pct": loss_pct,
                "abs_loss": abs(loss),
            }
        )
    return results


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Plus Jakarta Sans\',sans-serif; font-size:1rem; '
        'font-weight:700; color:#a5b4fc; letter-spacing:0.01em; margin-bottom:1rem;">'
        "💼 Portfolio Risk</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Portfolio Positions**")
    st.caption("Edit CSV: ticker, shares, entry, current")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    if uploaded is not None:
        try:
            raw_csv = uploaded.read().decode("utf-8")
        except UnicodeDecodeError:
            st.error(
                "CSV file must be UTF-8 encoded. "
                "Re-save from Excel as 'CSV UTF-8 (Comma delimited)'."
            )
            raw_csv = _DEFAULT_POSITIONS_CSV
    else:
        raw_csv = _DEFAULT_POSITIONS_CSV

    portfolio_text = st.text_area(
        "Positions CSV",
        value=raw_csv.strip(),
        height=160,
        label_visibility="collapsed",
        key="portfolio_csv",
    )

    st.markdown("---")
    st.markdown("**Watchlist**")
    watchlist_input = st.text_input(
        "Tickers (comma-separated)",
        value="IWM, EEM",
        key="watchlist_tickers",
    )

    st.markdown("---")
    run_btn = st.button("Run Analysis", key="run_analysis", type="primary")

# ---------------------------------------------------------------------------
# Parse positions
# ---------------------------------------------------------------------------

positions_df = parse_positions(portfolio_text)
if positions_df.empty:
    st.warning("No valid positions found. Check the CSV format in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Page title
# ---------------------------------------------------------------------------
st.markdown(
    '<h1 style="font-family:\'Plus Jakarta Sans\',sans-serif; font-size:1.6rem; '
    'font-weight:700; color:#f1f5f9; margin:0 0 1.2rem 0; letter-spacing:-0.01em;">'
    "Portfolio Risk</h1>",
    unsafe_allow_html=True,
)

if not _verify.LOOKAHEAD_CHECK_PASSED:
    st.error("Look-ahead bias check FAILED. HMM results may be unreliable.")
    st.stop()

# ---------------------------------------------------------------------------
# Run analysis block
# ---------------------------------------------------------------------------

# Initialise session state
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False
if "regime_data" not in st.session_state:
    st.session_state["regime_data"] = {}
if "corr_matrix" not in st.session_state:
    st.session_state["corr_matrix"] = None
if "watchlist_data" not in st.session_state:
    st.session_state["watchlist_data"] = {}

if run_btn:
    start_1y, end_today = date_range_default(years=1)
    start_60d = (date.today() - pd.Timedelta(days=90)).isoformat()

    # --- Regime detection per position ---
    with st.spinner("Detecting regimes..."):
        regime_data: dict[str, dict[str, Any]] = {}
        for _, row in positions_df.iterrows():
            ticker = row["ticker"]
            regime_data[ticker] = cached_fit_regime(ticker, start_1y, end_today)
        st.session_state["regime_data"] = regime_data

    # --- Correlation matrix ---
    tickers_list = positions_df["ticker"].tolist()
    with st.spinner("Computing correlations..."):
        corr = cached_correlation_matrix(tickers_list, start_60d, end_today)
        st.session_state["corr_matrix"] = corr

    # --- Watchlist ---
    raw_watchlist = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
    watchlist_data: dict[str, dict[str, Any]] = {}
    with st.spinner("Loading watchlist..."):
        for ticker in raw_watchlist:
            price = cached_latest_price(ticker, start_1y, end_today)
            regime_info = cached_fit_regime(ticker, start_1y, end_today)
            watchlist_data[ticker] = {"price": price, **regime_info}
        st.session_state["watchlist_data"] = watchlist_data

    st.session_state["analysis_done"] = True

# ---------------------------------------------------------------------------
# Compute portfolio-level metrics (always, from positions_df)
# ---------------------------------------------------------------------------

pf_metrics = compute_portfolio_metrics(positions_df)
stress_results = compute_stress_results(positions_df)
regime_data = st.session_state.get("regime_data", {})

# Favorable regimes count
favorable_regimes = {"Low Vol", "Medium Vol"}
n_favorable = sum(
    1
    for _, row in positions_df.iterrows()
    if regime_data.get(row["ticker"], {}).get("regime", "Uncertain") in favorable_regimes
)

market_status = "Market open" if is_market_open() else "Market closed"
n_positions = len(positions_df)

# ---------------------------------------------------------------------------
# Top stats bar
# ---------------------------------------------------------------------------

pnl = pf_metrics["total_pnl"]
pnl_pct = pf_metrics["total_pnl_pct"]
pnl_color = _SUCCESS if pnl >= 0 else _DANGER
pnl_arrow = "↑" if pnl >= 0 else "↓"
pnl_sign = "+" if pnl >= 0 else ""

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.markdown(
        f"""
<div class="stat-card">
  <div class="stat-label">Portfolio Value</div>
  <div class="stat-value-large">${pf_metrics['total_value']:,.0f}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with col_b:
    st.markdown(
        f"""
<div class="stat-card">
  <div class="stat-label">Total P&amp;L</div>
  <div class="stat-value-pnl" style="color:{pnl_color};">
    {pnl_arrow} {pnl_sign}${abs(pnl):,.0f}
    <span style="font-size:1rem; font-weight:400; opacity:0.75;">&nbsp;{pnl_sign}{pnl_pct*100:.1f}%</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with col_c:
    st.markdown(
        f"""
<div class="stat-card">
  <div class="stat-label">Overview</div>
  <div class="stat-value-summary">
    {n_positions} positions<br>
    <span style="color:{_SUCCESS};">{n_favorable} in favorable regime</span><br>
    <span style="color:{'#10b981' if 'open' in market_status else '#f59e0b'};">{market_status}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<hr class='pf-divider'>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main body: 60% left / 40% right
# ---------------------------------------------------------------------------

if not st.session_state["analysis_done"]:
    st.info("Click **Run Analysis** in the sidebar to load regime data, correlations, and watchlist.")

left_col, right_col = st.columns([3, 2])

# ── Left column: position cards ──────────────────────────────────────────────
with left_col:
    st.markdown('<div class="section-header-fintech">Positions</div>', unsafe_allow_html=True)

    for _, row in positions_df.iterrows():
        ticker: str = row["ticker"]
        shares: int = int(row["shares"])
        entry: float = float(row["entry"])
        current: float = float(row["current"])

        rdata = regime_data.get(ticker, {})
        regime = rdata.get("regime", "Uncertain")
        confidence = rdata.get("confidence", 0.5)
        days_in = rdata.get("days_in_regime", 0)

        card_html = _position_card_html(
            ticker=ticker,
            shares=shares,
            entry=entry,
            current=current,
            regime=regime,
            confidence=confidence,
            days_in_regime=days_in,
        )
        st.markdown(card_html, unsafe_allow_html=True)

# ── Right column: correlation, stress tests, watchlist ───────────────────────
with right_col:

    # ── Correlation heatmap ─────────────────────────────────────────────────
    st.markdown('<div class="section-header-fintech">60-Day Return Correlation</div>', unsafe_allow_html=True)
    corr_matrix = st.session_state.get("corr_matrix")

    with st.container():
        st.markdown('<div class="right-card">', unsafe_allow_html=True)
        if corr_matrix is not None:
            # Flag high-correlation pairs
            tickers_list = positions_df["ticker"].tolist()
            high_corr_pairs = []
            n = len(corr_matrix.columns)
            for i in range(n):
                for j in range(i + 1, n):
                    val = corr_matrix.iloc[i, j]
                    if abs(val) > 0.85:
                        ti = corr_matrix.columns[i]
                        tj = corr_matrix.columns[j]
                        high_corr_pairs.append(f"{ti}/{tj}: {val:.2f}")

            fig_corr = _build_correlation_heatmap(corr_matrix)
            st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})

            if high_corr_pairs:
                pairs_str = "  ·  ".join(high_corr_pairs)
                st.markdown(
                    f'<div style="font-family:\'JetBrains Mono\',monospace; font-size:0.7rem; '
                    f'color:#f43f5e; margin-top:-0.3rem;">⚠ High correlation: {pairs_str}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div style="font-family:\'Plus Jakarta Sans\',sans-serif; color:#64748b; '
                'font-size:0.82rem; padding:1rem 0; text-align:center;">'
                "Run analysis to compute correlations.</div>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Stress tests ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header-fintech" style="margin-top:1rem;">Stress Tests</div>', unsafe_allow_html=True)
    st.markdown('<div class="right-card">', unsafe_allow_html=True)

    max_abs_loss = max((abs(r["portfolio_loss"]) for r in stress_results), default=1.0)
    for sr in stress_results:
        row_html = _stress_row_html(
            scenario_name=sr["name"],
            portfolio_loss=sr["portfolio_loss"],
            loss_pct=sr["loss_pct"],
            max_abs_loss=max_abs_loss,
        )
        st.markdown(row_html, unsafe_allow_html=True)

    covered = set(_STRESS_SCENARIOS[0]["returns"].keys())
    uncovered = [row["ticker"] for _, row in positions_df.iterrows() if row["ticker"] not in covered]
    if uncovered:
        st.caption(
            f"Stress impact is 0% for tickers not in historical scenarios: {', '.join(uncovered)}"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Watchlist ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header-fintech" style="margin-top:1rem;">Watchlist</div>', unsafe_allow_html=True)
    st.markdown('<div class="right-card">', unsafe_allow_html=True)

    watchlist_data = st.session_state.get("watchlist_data", {})
    raw_watchlist_tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]

    if not watchlist_data and raw_watchlist_tickers:
        st.markdown(
            '<div style="font-family:\'Plus Jakarta Sans\',sans-serif; color:#64748b; '
            'font-size:0.82rem; padding:0.8rem 0; text-align:center;">'
            "Run analysis to load watchlist.</div>",
            unsafe_allow_html=True,
        )
    elif not raw_watchlist_tickers:
        st.markdown(
            '<div style="font-family:\'Plus Jakarta Sans\',sans-serif; color:#64748b; '
            'font-size:0.82rem; padding:0.8rem 0;">'
            "No watchlist tickers configured.</div>",
            unsafe_allow_html=True,
        )
    else:
        for wt in raw_watchlist_tickers:
            wdata = watchlist_data.get(wt)
            if wdata is None:
                # Not yet loaded — show placeholder
                st.markdown(
                    f'<div class="watchlist-row">'
                    f'<span class="watchlist-ticker">{_html.escape(wt)}</span>'
                    f'<span style="font-family:\'JetBrains Mono\',monospace; font-size:0.72rem; color:#64748b;">pending…</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                continue

            w_price = wdata.get("price")
            w_regime = wdata.get("regime", "Uncertain")
            w_conf = wdata.get("confidence", 0.5)
            w_color = REGIME_COLORS.get(w_regime, REGIME_COLORS["Uncertain"])
            w_badge = regime_badge(w_regime, w_conf, glow=False)
            price_str = f"${w_price:,.2f}" if w_price is not None else "N/A"
            conf_pct = min(w_conf * 100, 100)

            st.markdown(
                f"""
<div class="watchlist-row">
  <span class="watchlist-ticker">{_html.escape(wt)}</span>
  <span class="watchlist-price">{price_str}</span>
  {w_badge}
  <div class="watchlist-conf-bar">
    <div style="height:100%; width:{conf_pct:.0f}%; background:{w_color}; border-radius:2px;"></div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)
