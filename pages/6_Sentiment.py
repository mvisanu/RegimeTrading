"""
pages/6_Sentiment.py
====================
Dashboard 6 — Sentiment Analysis ("Newsroom Briefing").

Design language: Newsroom serif. Newsreader for headlines/headers, Inter for
body copy, JetBrains Mono for numbers. Near-white editorial palette on a
deep ``#111318`` background with 4px card radius and no decorative glow.

Background:  #111318   Card bg: #1a1d25   Primary: #e2e8f0
Bullish:     #22c55e   Bearish: #ef4444   Neutral:  #64748b
Card radius: 4px        Divider: 1px #2a2d35

Data source: Google News RSS, scored with NLTK VADER.
NO HMM — this dashboard does not use core.hmm_utils at all.
"""

from __future__ import annotations

import html as _html
import math
import time
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import feedparser
import nltk
import plotly.graph_objects as go
import streamlit as st

from core.design_system import REGIME_COLORS, get_plotly_layout  # noqa: F401 — project convention

# ---------------------------------------------------------------------------
# VADER lexicon — download once at module load
# ---------------------------------------------------------------------------

nltk.download("vader_lexicon", quiet=True)

from nltk.sentiment import SentimentIntensityAnalyzer  # noqa: E402 — after download

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="📰",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

_BG = "#111318"
_CARD_BG = "#1a1d25"
_CARD_BORDER = "rgba(255,255,255,0.06)"
_PRIMARY = "#e2e8f0"
_BULLISH = "#22c55e"
_BEARISH = "#ef4444"
_NEUTRAL = "#64748b"
_DIVIDER = "#2a2d35"
_CARD_RADIUS = "4px"

_GOOGLE_FONTS_URL = (
    "https://fonts.googleapis.com/css2?"
    "family=Newsreader:ital,wght@0,400;0,600;1,400"
    "&family=Inter:wght@400;500"
    "&family=JetBrains+Mono:wght@400"
    "&display=swap"
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

_CSS = f"""
<style>
@import url('{_GOOGLE_FONTS_URL}');

html, body, [class*="css"] {{
    background-color: {_BG} !important;
    color: {_PRIMARY};
    font-family: 'Inter', sans-serif;
}}

/* Streamlit chrome overrides */
section[data-testid="stSidebar"] {{
    background-color: {_CARD_BG} !important;
    border-right: 1px solid {_DIVIDER};
}}

.block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}

/* Newsroom heading style */
.newsroom-masthead {{
    font-family: 'Newsreader', Georgia, serif;
    font-size: 28px;
    font-weight: 600;
    letter-spacing: 4px;
    color: {_PRIMARY};
    text-transform: uppercase;
    margin: 0;
    padding: 0;
}}

.newsroom-dateline {{
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    color: {_NEUTRAL};
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 4px;
}}

.newsroom-rule {{
    border: none;
    border-top: 1px solid {_DIVIDER};
    margin: 12px 0 20px 0;
}}

/* Ticker summary card */
.ticker-card {{
    background: {_CARD_BG};
    border: 1px solid {_CARD_BORDER};
    border-radius: {_CARD_RADIUS};
    padding: 16px 14px 12px 14px;
    cursor: pointer;
    transition: background 0.15s;
}}

.ticker-card:hover {{
    background: #1f2330;
}}

.ticker-symbol {{
    font-family: 'Inter', sans-serif;
    font-size: 24px;
    font-weight: 700;
    color: {_PRIMARY};
    margin: 0 0 6px 0;
}}

.ticker-article-count {{
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    color: {_NEUTRAL};
    margin-top: 6px;
}}

/* Article row */
.article-row {{
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 10px 0;
    border-bottom: 1px solid {_DIVIDER};
}}

.article-row:last-child {{
    border-bottom: none;
}}

.article-dot {{
    width: 4px;
    min-width: 4px;
    height: 4px;
    border-radius: 50%;
    margin-top: 7px;
}}

.article-meta {{
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    color: {_NEUTRAL};
    font-variant: small-caps;
    line-height: 1.4;
}}

.article-headline {{
    font-family: 'Inter', sans-serif;
    font-size: 15px;
    color: {_PRIMARY};
    font-weight: 400;
    line-height: 1.45;
    margin: 2px 0;
}}

.article-score {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    white-space: nowrap;
}}

.detail-section-header {{
    font-family: 'Newsreader', Georgia, serif;
    font-size: 20px;
    font-weight: 600;
    color: {_PRIMARY};
    margin: 0 0 4px 0;
}}

.detail-rule {{
    border: none;
    border-top: 1px solid {_DIVIDER};
    margin: 0 0 16px 0;
}}
</style>
"""

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ArticleData:
    """Parsed and scored article from RSS feed."""

    title: str
    source: str
    published: datetime
    snippet: str
    compound_score: float
    weight: float


@dataclass
class TickerSentiment:
    """Aggregated sentiment result for a single ticker."""

    ticker: str
    weighted_score: float
    momentum: float
    article_count: int
    articles: list[ArticleData] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_sia() -> SentimentIntensityAnalyzer:
    """Return a cached SentimentIntensityAnalyzer instance (one per process)."""
    return SentimentIntensityAnalyzer()


def _score_color(score: float) -> str:
    """Return CSS color for a compound sentiment score."""
    if score > 0.05:
        return _BULLISH
    if score < -0.05:
        return _BEARISH
    return _NEUTRAL


def _momentum_arrow(momentum: float) -> str:
    """Return a directional arrow character for the momentum value."""
    if momentum > 0.05:
        return "↑"
    if momentum < -0.05:
        return "↓"
    return "→"


def _momentum_color(momentum: float) -> str:
    """Return CSS color for a momentum value."""
    if momentum > 0.05:
        return _BULLISH
    if momentum < -0.05:
        return _BEARISH
    return _NEUTRAL


def _parse_published(entry: Any) -> datetime:
    """Convert feedparser ``published_parsed`` (time.struct_time) to datetime.

    Returns a timezone-aware UTC datetime. Falls back to ``datetime.now(UTC)``
    if the field is absent or malformed.
    """
    parsed = getattr(entry, "published_parsed", None)
    if parsed is None:
        return datetime.now(timezone.utc)
    try:
        return datetime(*parsed[:6], tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


def _article_weight(published: datetime) -> float:
    """Return recency weight for an article based on age.

    - Last 24 hours  → 2×
    - Older than 3 days → 0.5×
    - Otherwise → 1×
    """
    now = datetime.now(timezone.utc)
    # Ensure published is timezone-aware for comparison
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    age_hours = (now - published).total_seconds() / 3600.0
    if age_hours <= 24.0:
        return 2.0
    if age_hours > 72.0:
        return 0.5
    return 1.0


# ---------------------------------------------------------------------------
# RSS + scoring
# ---------------------------------------------------------------------------

def _empty_raw(ticker: str) -> dict:
    return {"ticker": ticker, "weighted_score": 0.0, "momentum": 0.0, "article_count": 0, "articles": []}


@st.cache_data(ttl=900)
def _fetch_and_score_raw(ticker: str) -> dict:
    """Cached fetch — returns plain dicts so pickle can always serialize them."""
    safe_ticker = urllib.parse.quote_plus(ticker)
    url = f"https://news.google.com/rss/search?q={safe_ticker}+stock"

    try:
        feed = feedparser.parse(url)
    except Exception:
        return _empty_raw(ticker)

    # Bozo flag means feedparser encountered a parse error; if there are also
    # no entries the feed is unusable (e.g. network failure or blocked host).
    if getattr(feed, "bozo", False) and not list(getattr(feed, "entries", [])):
        return _empty_raw(ticker)

    entries = list(getattr(feed, "entries", []))

    sia = _get_sia()
    raw_articles: list[dict] = []

    for entry in entries:
        # Unescape HTML entities BEFORE scoring — feedparser sometimes delivers
        # entity-encoded RSS text (e.g. &amp;, &#39;) which VADER mis-scores.
        raw_title: str = _html.unescape(getattr(entry, "title", "") or "")
        raw_snippet: str = _html.unescape(getattr(entry, "summary", "") or "")
        source_obj = getattr(entry, "source", None)
        source: str = getattr(source_obj, "title", "Unknown") if source_obj else "Unknown"
        published = _parse_published(entry)

        text = raw_title + " " + raw_snippet
        compound = sia.polarity_scores(text)["compound"]
        weight = _article_weight(published)

        raw_articles.append({
            "title": raw_title,
            "source": source,
            "published": published.isoformat(),
            "snippet": raw_snippet,
            "compound_score": compound,
            "weight": weight,
        })

    if not raw_articles:
        return _empty_raw(ticker)

    total_weight = sum(a["weight"] for a in raw_articles)
    weighted_score = (
        sum(a["compound_score"] * a["weight"] for a in raw_articles) / total_weight
        if total_weight > 0 else 0.0
    )

    recent = [a["compound_score"] for a in raw_articles if a["weight"] == 2.0]
    old = [a["compound_score"] for a in raw_articles if a["weight"] == 0.5]
    momentum = (sum(recent) / len(recent) if recent else 0.0) - (sum(old) / len(old) if old else 0.0)

    return {
        "ticker": ticker,
        "weighted_score": weighted_score,
        "momentum": momentum,
        "article_count": len(raw_articles),
        "articles": raw_articles,
    }


def fetch_and_score(ticker: str) -> TickerSentiment:
    """Fetch and score sentiment for *ticker*, returning a typed result.

    The network fetch and scoring are cached via :func:`_fetch_and_score_raw`;
    this wrapper converts the cache-safe dict back into a :class:`TickerSentiment`.
    """
    raw = _fetch_and_score_raw(ticker)
    articles = [
        ArticleData(
            title=a["title"],
            source=a["source"],
            published=datetime.fromisoformat(a["published"]),
            snippet=a["snippet"],
            compound_score=a["compound_score"],
            weight=a["weight"],
        )
        for a in raw["articles"]
    ]
    return TickerSentiment(
        ticker=raw["ticker"],
        weighted_score=raw["weighted_score"],
        momentum=raw["momentum"],
        article_count=raw["article_count"],
        articles=articles,
    )


# ---------------------------------------------------------------------------
# SVG gauge
# ---------------------------------------------------------------------------

def _sentiment_gauge_svg(score: float, size: int = 80) -> str:
    """Return inline SVG of a semicircular sentiment gauge with needle.

    The gauge spans 180° (left = bearish, right = bullish). The needle
    extends from the centre to the arc at the angle corresponding to *score*.

    Args:
        score: Compound VADER score in [-1, 1].
        size:  Width of the SVG in pixels; height is ``size // 2 + 10``.

    Returns:
        HTML ``<svg>`` string suitable for embedding via ``st.markdown``.
    """
    cx = size / 2
    cy = size / 2
    r = size * 0.4

    # score=-1 → 180° (left), score=0 → 90° (top), score=1 → 0° (right)
    angle_deg = (1.0 - score) / 2.0 * 180.0
    angle_rad = math.radians(angle_deg)

    # Needle tip on the arc (SVG y-axis is downward)
    needle_x = cx + r * math.cos(math.pi - angle_rad)
    needle_y = cy - r * math.sin(math.pi - angle_rad)

    color = _score_color(score)

    h = size // 2 + 10
    return (
        f'<svg width="{size}" height="{h}" viewBox="0 0 {size} {h}">'
        f'<path d="M {size*0.1},{cy} A {r},{r} 0 0 1 {size*0.9},{cy}" '
        f'fill="none" stroke="{_DIVIDER}" stroke-width="6"/>'
        f'<line x1="{cx}" y1="{cy}" x2="{needle_x}" y2="{needle_y}" '
        f'stroke="{color}" stroke-width="2" stroke-linecap="round"/>'
        f'<circle cx="{cx}" cy="{cy}" r="3" fill="{color}"/>'
        f'</svg>'
    )


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _render_ticker_card(result: TickerSentiment, selected: bool) -> str:
    """Return HTML for a single ticker summary card.

    Args:
        result:   Aggregated sentiment for the ticker.
        selected: When ``True`` a stronger border is applied to indicate focus.

    Returns:
        HTML string for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    score = result.weighted_score
    border_color = _score_color(score)
    extra_bg = f"background:{_CARD_BG};" if not selected else f"background:#1f2330;"

    arrow = _momentum_arrow(result.momentum)
    arrow_color = _momentum_color(result.momentum)
    score_color = _score_color(score)
    gauge_svg = _sentiment_gauge_svg(score, size=80)

    score_str = f"{score:+.3f}"
    ticker_safe = _html.escape(result.ticker)

    return (
        f'<div class="ticker-card" style="border-left:3px solid {border_color};{extra_bg}">'
        f'<div class="ticker-symbol">{ticker_safe}</div>'
        f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
        f'{gauge_svg}'
        f'</div>'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:18px;'
        f'font-weight:400;color:{score_color};">{score_str}</span>'
        f'<span style="font-size:18px;color:{arrow_color};">{arrow}</span>'
        f'</div>'
        f'<div class="ticker-article-count">{result.article_count} articles</div>'
        f'</div>'
    )


def _render_article_row(article: ArticleData, last: bool = False) -> str:
    """Return HTML for a single compact article row in the detail panel.

    Args:
        article: Scored article to render.
        last:    When ``True`` the bottom divider line is suppressed.

    Returns:
        HTML string for ``st.markdown(..., unsafe_allow_html=True)``.
    """
    dot_color = _score_color(article.compound_score)
    score_color = dot_color
    border = "none" if last else f"1px solid {_DIVIDER}"

    title_safe = _html.escape(article.title)
    source_safe = _html.escape(article.source)
    score_str = f"{article.compound_score:+.3f}"

    try:
        date_str = article.published.strftime("%b %d, %Y")
    except Exception:
        date_str = "—"

    return (
        f'<div style="display:flex;align-items:flex-start;gap:10px;'
        f'padding:10px 0;border-bottom:{border};">'
        # dot
        f'<div style="width:4px;min-width:4px;height:4px;border-radius:50%;'
        f'background:{dot_color};margin-top:7px;"></div>'
        # text block
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-family:\'Inter\',sans-serif;font-size:11px;'
        f'color:{_NEUTRAL};font-variant:small-caps;line-height:1.4;">'
        f'{source_safe} &middot; {date_str}</div>'
        f'<div style="font-family:\'Inter\',sans-serif;font-size:15px;'
        f'color:{_PRIMARY};font-weight:400;line-height:1.45;margin:2px 0;">'
        f'{title_safe}</div>'
        f'</div>'
        # score
        f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:12px;'
        f'color:{score_color};white-space:nowrap;padding-top:6px;">{score_str}</div>'
        f'</div>'
    )


def _render_aggregate_bar(results: list[TickerSentiment]) -> go.Figure:
    """Return a Plotly horizontal bar chart of aggregate scores centered at 0.

    Args:
        results: Sentiment results for all tickers.

    Returns:
        Plotly :class:`go.Figure`.
    """
    tickers = [r.ticker for r in results]
    scores = [r.weighted_score for r in results]
    colors = [_score_color(s) for s in scores]

    layout = get_plotly_layout("dark")
    layout.update(
        {
            "paper_bgcolor": _BG,
            "plot_bgcolor": _BG,
            "height": 220,
            "margin": {"l": 80, "r": 40, "t": 30, "b": 40},
            "xaxis": {
                **layout["xaxis"],
                "range": [-1.05, 1.05],
                "zeroline": True,
                "zerolinecolor": _NEUTRAL,
                "zerolinewidth": 1,
                "tickfont": {"color": _NEUTRAL, "family": "JetBrains Mono", "size": 11},
                "title": None,
                "showgrid": True,
                "gridcolor": _DIVIDER,
            },
            "yaxis": {
                **layout["yaxis"],
                "tickfont": {"color": _PRIMARY, "family": "Inter", "size": 12},
                "showgrid": False,
            },
        }
    )

    fig = go.Figure(
        data=[
            go.Bar(
                x=scores,
                y=tickers,
                orientation="h",
                marker_color=colors,
                text=[f"{s:+.3f}" for s in scores],
                textfont={"family": "JetBrains Mono", "size": 11, "color": _PRIMARY},
                textposition="outside",
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(**layout)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point — renders Dashboard 6: Sentiment Analysis."""

    # Inject global CSS
    st.markdown(_CSS, unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Sidebar
    # -----------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            f'<div style="font-family:\'Newsreader\',serif;font-size:18px;'
            f'font-weight:600;color:{_PRIMARY};margin-bottom:12px;">Sentiment Settings</div>',
            unsafe_allow_html=True,
        )

        tickers_raw: str = st.text_input(
            "Tickers (comma-separated)",
            value="SPY, AAPL, NVDA, TSLA, BTC-USD",
            help="Enter ticker symbols separated by commas.",
        )

        refresh = st.button("Refresh Sentiment", use_container_width=True)

        if refresh:
            _fetch_and_score_raw.clear()
            st.rerun()

        st.markdown('<hr style="border-top:1px solid #2a2d35;margin:12px 0;">', unsafe_allow_html=True)
        last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        st.markdown(
            f'<div style="font-size:11px;color:{_NEUTRAL};">Last updated<br>{last_updated}</div>',
            unsafe_allow_html=True,
        )

    # Parse tickers
    tickers: list[str] = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
    if not tickers:
        st.warning("Enter at least one ticker symbol in the sidebar.")
        return

    # -----------------------------------------------------------------------
    # Fetch sentiment for all tickers
    # -----------------------------------------------------------------------
    results: list[TickerSentiment] = []
    empty_tickers: list[str] = []

    with st.spinner("Fetching news and scoring sentiment…"):
        for ticker in tickers:
            result = fetch_and_score(ticker)
            results.append(result)
            if result.article_count == 0:
                empty_tickers.append(ticker)

    if empty_tickers:
        st.warning(
            "No articles returned for: "
            + ", ".join(_html.escape(t) for t in empty_tickers)
            + ". Google News may be rate-limiting requests or the ticker format is unrecognised."
        )

    # -----------------------------------------------------------------------
    # 1. Header section
    # -----------------------------------------------------------------------
    total_articles = sum(r.article_count for r in results)
    now = datetime.now(timezone.utc)

    # Cross-platform date format (%-d fails on Windows)
    try:
        date_display = now.strftime("%A, %B %-d, %Y")
    except ValueError:
        date_display = now.strftime("%A, %B %d, %Y").replace(" 0", " ")

    col_title, col_date = st.columns([3, 1])
    with col_title:
        st.markdown(
            '<div class="newsroom-masthead">Market Sentiment Briefing</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="newsroom-dateline">{total_articles} articles analyzed</div>',
            unsafe_allow_html=True,
        )
    with col_date:
        st.markdown(
            f'<div style="text-align:right;font-family:\'Newsreader\',serif;'
            f'font-size:14px;color:{_NEUTRAL};padding-top:8px;">'
            f'{_html.escape(date_display)}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="newsroom-rule">', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 2. Ticker summary cards (horizontal row)
    # -----------------------------------------------------------------------
    # Use session state to track selected ticker
    if "selected_ticker_idx" not in st.session_state:
        st.session_state["selected_ticker_idx"] = 0

    n_tickers = len(results)
    card_cols = st.columns(n_tickers if n_tickers <= 6 else 6)

    if n_tickers > 6:
        st.caption(f"Showing first 6 of {n_tickers} tickers. Reduce the list to see all.")

    for i, (col, result) in enumerate(zip(card_cols, results)):
        with col:
            selected = st.session_state["selected_ticker_idx"] == i
            card_html = _render_ticker_card(result, selected=selected)
            st.markdown(card_html, unsafe_allow_html=True)
            if st.button(
                result.ticker,
                key=f"ticker_btn_{i}",
                use_container_width=True,
                help=f"Show detail for {result.ticker}",
            ):
                st.session_state["selected_ticker_idx"] = i
                st.rerun()

    st.markdown("<div style='margin-top:8px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 3. Detail panel (selected ticker)
    # -----------------------------------------------------------------------
    selected_idx = st.session_state["selected_ticker_idx"]
    # Guard against index out of range (e.g. user reduced ticker list)
    if selected_idx >= len(results):
        selected_idx = 0
        st.session_state["selected_ticker_idx"] = 0

    detail_result = results[selected_idx]

    st.markdown(
        f'<div class="detail-section-header">'
        f'{_html.escape(detail_result.ticker)} — Top Stories'
        f'</div>'
        f'<hr class="detail-rule">',
        unsafe_allow_html=True,
    )

    if not detail_result.articles:
        st.info(f"No articles found for {_html.escape(detail_result.ticker)}.")
    else:
        # Top 5 by abs(score)
        sorted_articles = sorted(
            detail_result.articles,
            key=lambda a: abs(a.compound_score),
            reverse=True,
        )
        top5 = sorted_articles[:5]
        rest = sorted_articles[5:]

        # Render top 5
        top5_html = "".join(
            _render_article_row(a, last=(i == len(top5) - 1 and not rest))
            for i, a in enumerate(top5)
        )
        st.markdown(
            f'<div style="background:{_CARD_BG};border:1px solid {_CARD_BORDER};'
            f'border-radius:{_CARD_RADIUS};padding:0 16px;">'
            f'{top5_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Remaining articles in an expander
        if rest:
            with st.expander(f"Show {len(rest)} more articles"):
                rest_html = "".join(
                    _render_article_row(a, last=(i == len(rest) - 1))
                    for i, a in enumerate(rest)
                )
                st.markdown(
                    f'<div style="background:{_CARD_BG};border:1px solid {_CARD_BORDER};'
                    f'border-radius:{_CARD_RADIUS};padding:0 16px;">'
                    f'{rest_html}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # -----------------------------------------------------------------------
    # 4. Aggregate sentiment bar (bottom)
    # -----------------------------------------------------------------------
    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="font-family:\'Newsreader\',serif;font-size:16px;font-weight:600;'
        f'color:{_PRIMARY};margin-bottom:8px;">Aggregate Sentiment Overview</div>',
        unsafe_allow_html=True,
    )

    if results:
        fig = _render_aggregate_bar(results)
        st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------------------------
    # 5. Disclaimer
    # -----------------------------------------------------------------------
    st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
    st.markdown('<hr style="border-top:1px solid #2a2d35;margin:0 0 8px 0;">', unsafe_allow_html=True)
    st.caption(
        "⚠ Automated sentiment analysis has known limitations. "
        "Scores should not be used as the sole basis for trading decisions."
    )


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

main()
