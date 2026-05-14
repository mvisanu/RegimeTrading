"""
tests/test_dashboard_5.py — Unit tests for pages/5_Multi_Asset_Backtest.py.

Dashboard 5: Asset-Colored Multi-Asset Regime Backtester.
Design language: #0c0c14 bg, Outfit + Fira Code, --accent CSS variable per asset.
Tests: importability, _ticker_color, ASSET_COLORS contract,
       build_regime_timeline_chart, _compute_stress_drawdown, design compliance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from tests.conftest_dashboard_helpers import load_dashboard, make_st_mock

_ST = make_st_mock()
_d5 = load_dashboard(5, _ST)

_ticker_color = _d5._ticker_color
ASSET_COLORS = _d5.ASSET_COLORS
EXTRA_COLORS = _d5.EXTRA_COLORS
DEFAULT_TICKERS = _d5.DEFAULT_TICKERS
STRESS_PERIODS = _d5.STRESS_PERIODS
build_regime_timeline_chart = _d5.build_regime_timeline_chart
_compute_stress_drawdown = _d5._compute_stress_drawdown

_SRC = (_REPO / "pages" / "5_Multi_Asset_Backtest.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Minimal mock RegimeResult for timeline chart tests
# ---------------------------------------------------------------------------

class _MockRegimeResult:
    def __init__(self, labels):
        self.stable_labels = labels
        self.confidence = np.ones(len(labels)) * 0.8
        self.n_regimes = 3


# ---------------------------------------------------------------------------
# T-D5-001: Importability
# ---------------------------------------------------------------------------

class TestImportability:
    def test_module_loads(self):
        assert _d5 is not None

    def test_required_names(self):
        for name in ("ASSET_COLORS", "DEFAULT_TICKERS", "STRESS_PERIODS",
                     "_ticker_color", "build_regime_timeline_chart",
                     "_compute_stress_drawdown"):
            assert hasattr(_d5, name), f"Missing: {name}"


# ---------------------------------------------------------------------------
# T-D5-002: ASSET_COLORS contract
# ---------------------------------------------------------------------------

class TestAssetColors:
    def test_four_default_tickers_have_colors(self):
        for ticker in ("SPY", "BTC-USD", "GLD", "TLT"):
            assert ticker in ASSET_COLORS, f"{ticker} must be in ASSET_COLORS"

    def test_spy_is_cyan(self):
        assert ASSET_COLORS["SPY"] == "#00d4ff"

    def test_btcusd_is_orange(self):
        assert ASSET_COLORS["BTC-USD"] == "#f7921a"

    def test_gld_is_gold(self):
        assert ASSET_COLORS["GLD"] == "#ffd700"

    def test_tlt_is_purple(self):
        assert ASSET_COLORS["TLT"] == "#a78bfa"

    def test_default_tickers_list_matches_colors(self):
        for ticker in DEFAULT_TICKERS:
            assert ticker in ASSET_COLORS, (
                f"DEFAULT_TICKERS entry {ticker!r} has no color in ASSET_COLORS"
            )


# ---------------------------------------------------------------------------
# T-D5-003: _ticker_color
# ---------------------------------------------------------------------------

class TestTickerColor:
    def test_known_ticker_returns_canonical_color(self):
        assert _ticker_color("SPY") == "#00d4ff"
        assert _ticker_color("GLD") == "#ffd700"

    def test_unknown_ticker_falls_back_to_extra_colors(self):
        color = _ticker_color("NVDA", index=0)
        assert color in EXTRA_COLORS

    def test_unknown_ticker_cycles_extra_colors(self):
        c0 = _ticker_color("UNKNOWN", index=0)
        c1 = _ticker_color("UNKNOWN", index=1)
        assert c0 != c1 or len(EXTRA_COLORS) == 1

    def test_lowercase_ticker_recognized(self):
        """Case-insensitive lookup."""
        assert _ticker_color("spy") == "#00d4ff"

    def test_returns_hex_string(self):
        color = _ticker_color("GLD")
        assert color.startswith("#") and len(color) == 7


# ---------------------------------------------------------------------------
# T-D5-004: _compute_stress_drawdown
# ---------------------------------------------------------------------------

class TestComputeStressDrawdown:
    def _equity(self):
        idx = pd.date_range("2008-01-01", periods=500)
        vals = 100.0 * np.cumprod(1.0 + np.random.default_rng(0).normal(0, 0.01, 500))
        return pd.Series(vals, index=idx)

    def test_returns_float_for_valid_range(self):
        eq = self._equity()
        result = _compute_stress_drawdown(eq, "2008-01-01", "2008-12-31")
        assert isinstance(result, float)

    def test_returns_none_for_insufficient_data(self):
        eq = pd.Series([100.0], index=pd.date_range("2020-01-01", periods=1))
        result = _compute_stress_drawdown(eq, "2020-01-01", "2020-01-01")
        assert result is None

    def test_drawdown_is_non_positive(self):
        eq = self._equity()
        result = _compute_stress_drawdown(eq, "2008-01-01", "2008-12-31")
        if result is not None:
            assert result <= 0.0

    def test_flat_equity_zero_drawdown(self):
        idx = pd.date_range("2020-01-01", periods=100)
        eq = pd.Series(np.ones(100) * 100.0, index=idx)
        result = _compute_stress_drawdown(eq, "2020-01-01", "2020-04-09")
        assert result is not None
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_returns_none_for_range_outside_data(self):
        eq = self._equity()
        result = _compute_stress_drawdown(eq, "2050-01-01", "2050-12-31")
        assert result is None


# ---------------------------------------------------------------------------
# T-D5-005: build_regime_timeline_chart
# ---------------------------------------------------------------------------

class TestBuildRegimeTimelineChart:
    def test_empty_tickers_returns_empty_figure(self):
        import plotly.graph_objects as go
        fig = build_regime_timeline_chart([], {}, {})
        assert isinstance(fig, go.Figure)

    def test_figure_returned_for_valid_input(self):
        import plotly.graph_objects as go
        tickers = ["SPY", "GLD"]
        regime_results = {
            "SPY": _MockRegimeResult(["Low Vol"] * 50 + ["High Vol"] * 50),
            "GLD": _MockRegimeResult(["Medium Vol"] * 100),
        }
        colors = {"SPY": "#00d4ff", "GLD": "#ffd700"}
        fig = build_regime_timeline_chart(tickers, regime_results, colors)
        assert isinstance(fig, go.Figure)

    def test_figure_has_traces(self):
        import plotly.graph_objects as go
        tickers = ["SPY"]
        regime_results = {
            "SPY": _MockRegimeResult(["Low Vol"] * 30 + ["High Vol"] * 30),
        }
        colors = {"SPY": "#00d4ff"}
        fig = build_regime_timeline_chart(tickers, regime_results, colors)
        assert len(fig.data) > 0


# ---------------------------------------------------------------------------
# T-D5-006: STRESS_PERIODS contract
# ---------------------------------------------------------------------------

class TestStressPeriods:
    def test_three_stress_periods(self):
        assert len(STRESS_PERIODS) == 3

    def test_2008_in_stress_periods(self):
        assert any("2008" in k for k in STRESS_PERIODS)

    def test_2020_in_stress_periods(self):
        assert any("2020" in k for k in STRESS_PERIODS)

    def test_2022_in_stress_periods(self):
        assert any("2022" in k for k in STRESS_PERIODS)

    def test_periods_are_start_end_tuples(self):
        for name, (start, end) in STRESS_PERIODS.items():
            assert isinstance(start, str) and isinstance(end, str)
            assert start < end, f"Period {name}: start must be before end"


# ---------------------------------------------------------------------------
# T-D5-007: Design system compliance
# ---------------------------------------------------------------------------

class TestDesignCompliance:
    def test_bg_color(self):
        assert "#0c0c14" in _SRC, "Asset-colored bg #0c0c14 must appear in dashboard 5"

    def test_outfit_font(self):
        assert "Outfit" in _SRC, "Outfit font must be in dashboard 5"

    def test_fira_code_font(self):
        assert "Fira Code" in _SRC, "Fira Code font must be in dashboard 5"

    def test_accent_css_variable(self):
        assert "--accent" in _SRC, (
            "Dashboard 5 must use --accent CSS variable for asset-color theming"
        )

    def test_card_radius_12px(self):
        assert "12px" in _SRC, "Card radius 12px must appear in dashboard 5"

    def test_no_local_regime_colors_definition(self):
        import re
        assignments = re.findall(r"^REGIME_COLORS\s*=\s*\{", _SRC, re.MULTILINE)
        assert not assignments

    def test_lookahead_guard_present(self):
        assert "LOOKAHEAD_CHECK_PASSED" in _SRC or "lookahead" in _SRC.lower(), (
            "Dashboard 5 must check the lookahead bias flag"
        )

    def test_walk_forward_backtest_used(self):
        assert "walk_forward_backtest" in _SRC, (
            "Dashboard 5 must use walk_forward_backtest (no in-sample backtests)"
        )

    def test_viterbi_not_used(self):
        assert "model.predict(" not in _SRC, (
            "Dashboard 5 must not use model.predict() (Viterbi — look-ahead biased)"
        )
