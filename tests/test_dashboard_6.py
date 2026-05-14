"""
tests/test_dashboard_6.py — Unit tests for pages/6_Sentiment.py.

Dashboard 6: Newsroom Briefing Sentiment Analysis.
Design language: #111318 bg, Newsreader serif, Inter body, JetBrains Mono nums, 4px radius.
Tests: importability, _score_color, _momentum_arrow, _momentum_color,
       _article_weight, _parse_published, _sentiment_gauge_svg, fetch_and_score
       (mocked network), design compliance, no-HMM guarantee.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from tests.conftest_dashboard_helpers import load_dashboard, make_st_mock

# feedparser must be mocked before import so network is never touched
_feedparser_mock = MagicMock()
_feedparser_mock.parse.return_value = MagicMock(
    bozo=False,
    entries=[],
)
sys.modules["feedparser"] = _feedparser_mock

# NLTK must be available (installed with project); mock download call
import types
_nltk_mod = sys.modules.get("nltk") or MagicMock()
if hasattr(_nltk_mod, "download"):
    # nltk is real — patch its download to be silent
    pass  # it's already quiet=True in the dashboard

_ST = make_st_mock()
_d6 = load_dashboard(6, _ST)

_score_color = _d6._score_color
_momentum_arrow = _d6._momentum_arrow
_momentum_color = _d6._momentum_color
_article_weight = _d6._article_weight
_parse_published = _d6._parse_published
_sentiment_gauge_svg = _d6._sentiment_gauge_svg
TickerSentiment = _d6.TickerSentiment
ArticleData = _d6.ArticleData
_render_ticker_card = _d6._render_ticker_card
_render_article_row = _d6._render_article_row
_render_aggregate_bar = _d6._render_aggregate_bar

_SRC = (_REPO / "pages" / "6_Sentiment.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# T-D6-001: Importability
# ---------------------------------------------------------------------------

class TestImportability:
    def test_module_loads(self):
        assert _d6 is not None

    def test_required_names_present(self):
        for name in ("_score_color", "_momentum_arrow", "_article_weight",
                     "_parse_published", "_sentiment_gauge_svg",
                     "TickerSentiment", "ArticleData", "fetch_and_score"):
            assert hasattr(_d6, name), f"Missing: {name}"


# ---------------------------------------------------------------------------
# T-D6-002: _score_color
# ---------------------------------------------------------------------------

class TestScoreColor:
    def test_positive_is_bullish_green(self):
        color = _score_color(0.5)
        assert color == "#22c55e"

    def test_negative_is_bearish_red(self):
        color = _score_color(-0.5)
        assert color == "#ef4444"

    def test_near_zero_positive_is_neutral(self):
        color = _score_color(0.04)
        assert color == "#64748b"

    def test_near_zero_negative_is_neutral(self):
        color = _score_color(-0.04)
        assert color == "#64748b"

    def test_exact_boundary_positive_side(self):
        """score > 0.05 → bullish; score = 0.05 → neutral."""
        assert _score_color(0.05) == "#64748b"
        assert _score_color(0.06) == "#22c55e"

    def test_exact_boundary_negative_side(self):
        """score < -0.05 → bearish; score = -0.05 → neutral."""
        assert _score_color(-0.05) == "#64748b"
        assert _score_color(-0.06) == "#ef4444"

    def test_zero_is_neutral(self):
        assert _score_color(0.0) == "#64748b"


# ---------------------------------------------------------------------------
# T-D6-003: _momentum_arrow
# ---------------------------------------------------------------------------

class TestMomentumArrow:
    def test_positive_momentum_up_arrow(self):
        assert _momentum_arrow(0.1) == "↑"

    def test_negative_momentum_down_arrow(self):
        assert _momentum_arrow(-0.1) == "↓"

    def test_flat_momentum_right_arrow(self):
        assert _momentum_arrow(0.0) == "→"

    def test_boundary_positive(self):
        assert _momentum_arrow(0.05) == "→"
        assert _momentum_arrow(0.06) == "↑"

    def test_boundary_negative(self):
        assert _momentum_arrow(-0.05) == "→"
        assert _momentum_arrow(-0.06) == "↓"


# ---------------------------------------------------------------------------
# T-D6-004: _momentum_color
# ---------------------------------------------------------------------------

class TestMomentumColor:
    def test_positive_is_bullish(self):
        assert _momentum_color(0.1) == "#22c55e"

    def test_negative_is_bearish(self):
        assert _momentum_color(-0.1) == "#ef4444"

    def test_flat_is_neutral(self):
        assert _momentum_color(0.0) == "#64748b"


# ---------------------------------------------------------------------------
# T-D6-005: _article_weight
# ---------------------------------------------------------------------------

class TestArticleWeight:
    def test_recent_24h_weight_is_2(self):
        recent = datetime.now(timezone.utc) - timedelta(hours=12)
        assert _article_weight(recent) == 2.0

    def test_between_24h_and_72h_is_1(self):
        mid = datetime.now(timezone.utc) - timedelta(hours=48)
        assert _article_weight(mid) == 1.0

    def test_older_than_72h_weight_is_half(self):
        old = datetime.now(timezone.utc) - timedelta(hours=100)
        assert _article_weight(old) == 0.5

    def test_exactly_24h_boundary_recent(self):
        boundary = datetime.now(timezone.utc) - timedelta(hours=24)
        assert _article_weight(boundary) == 2.0

    def test_naive_datetime_handled(self):
        """Naive datetime (no tzinfo) must not raise."""
        naive = datetime.now() - timedelta(hours=6)
        result = _article_weight(naive)
        assert result in (0.5, 1.0, 2.0)


# ---------------------------------------------------------------------------
# T-D6-006: _parse_published
# ---------------------------------------------------------------------------

class TestParsePublished:
    def test_returns_datetime(self):
        import time
        entry = MagicMock()
        entry.published_parsed = time.gmtime()
        result = _parse_published(entry)
        assert isinstance(result, datetime)

    def test_missing_field_returns_now(self):
        entry = MagicMock()
        del entry.published_parsed
        entry.published_parsed = None
        result = _parse_published(entry)
        assert isinstance(result, datetime)

    def test_result_is_tz_aware(self):
        import time
        entry = MagicMock()
        entry.published_parsed = time.gmtime()
        result = _parse_published(entry)
        assert result.tzinfo is not None


# ---------------------------------------------------------------------------
# T-D6-007: _sentiment_gauge_svg
# ---------------------------------------------------------------------------

class TestSentimentGaugeSvg:
    def test_returns_string(self):
        assert isinstance(_sentiment_gauge_svg(0.0), str)

    def test_contains_svg_tag(self):
        svg = _sentiment_gauge_svg(0.5)
        assert "<svg" in svg and "</svg>" in svg

    def test_score_positive_one_does_not_crash(self):
        _sentiment_gauge_svg(1.0)

    def test_score_negative_one_does_not_crash(self):
        _sentiment_gauge_svg(-1.0)

    def test_size_parameter_respected(self):
        svg = _sentiment_gauge_svg(0.0, size=120)
        assert 'width="120"' in svg


# ---------------------------------------------------------------------------
# T-D6-008: fetch_and_score with mocked feedparser
# ---------------------------------------------------------------------------

class TestFetchAndScore:
    def test_empty_feed_returns_ticker_sentiment(self):
        """No articles → TickerSentiment with zero score and no articles."""
        _feedparser_mock.parse.return_value = MagicMock(bozo=False, entries=[])
        result = _d6.fetch_and_score.__wrapped__("AAPL") if hasattr(
            _d6.fetch_and_score, "__wrapped__"
        ) else _d6.fetch_and_score("AAPL")
        assert isinstance(result, TickerSentiment)
        assert result.article_count == 0
        assert result.weighted_score == 0.0

    def test_bozo_empty_returns_zero(self):
        """Bozo parse with no entries → zero-article TickerSentiment."""
        _feedparser_mock.parse.return_value = MagicMock(bozo=True, entries=[])
        result = _d6.fetch_and_score.__wrapped__("SPY") if hasattr(
            _d6.fetch_and_score, "__wrapped__"
        ) else _d6.fetch_and_score("SPY")
        assert result.article_count == 0


# ---------------------------------------------------------------------------
# T-D6-009: Design system compliance
# ---------------------------------------------------------------------------

class TestDesignCompliance:
    def test_newsroom_bg_color(self):
        assert "#111318" in _SRC, "Newsroom bg #111318 must appear in dashboard 6"

    def test_newsreader_font_referenced(self):
        assert "Newsreader" in _SRC, "Newsreader serif font must be in dashboard 6"

    def test_inter_font_referenced(self):
        assert "Inter" in _SRC, "Inter body font must be in dashboard 6"

    def test_jetbrains_mono_for_numbers(self):
        assert "JetBrains Mono" in _SRC, "JetBrains Mono must be in dashboard 6"

    def test_card_radius_4px(self):
        assert "4px" in _SRC, "4px card radius must appear in dashboard 6 (editorial style)"

    def test_bullish_green_color(self):
        assert "#22c55e" in _SRC

    def test_bearish_red_color(self):
        assert "#ef4444" in _SRC

    def test_no_hmm_import(self):
        """Dashboard 6 explicitly does NOT use HMM — core.hmm_utils must not be imported."""
        import ast, re
        # Check for actual import statements, not doc comments
        import_lines = [
            line for line in _SRC.splitlines()
            if re.match(r"\s*(import|from)\s+.*hmm_utils", line)
        ]
        assert not import_lines, (
            f"Dashboard 6 must not import core.hmm_utils (no HMM in sentiment dashboard). "
            f"Found: {import_lines}"
        )

    def test_vader_used(self):
        assert "VADER" in _SRC.upper() or "SentimentIntensityAnalyzer" in _SRC, (
            "Dashboard 6 must use NLTK VADER for sentiment scoring"
        )

    def test_google_news_rss_url(self):
        assert "news.google.com/rss" in _SRC, (
            "Dashboard 6 must fetch from Google News RSS"
        )

    def test_no_local_regime_colors_definition(self):
        import re
        assignments = re.findall(r"^REGIME_COLORS\s*=\s*\{", _SRC, re.MULTILINE)
        assert not assignments

    def test_disclaimer_in_source(self):
        assert "disclaimer" in _SRC.lower() or "limitation" in _SRC.lower(), (
            "Dashboard 6 must include a disclaimer about automated analysis limitations"
        )

    def test_recency_weighting_logic(self):
        """24h weight=2x, >72h weight=0.5x must appear in source logic."""
        assert "2.0" in _SRC or "2 *" in _SRC  # recency factor
        assert "0.5" in _SRC  # old articles weight
