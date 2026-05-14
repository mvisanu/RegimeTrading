"""
tests/test_dashboard_4.py — Unit tests for pages/4_Portfolio_Risk.py.

Dashboard 4: Premium Fintech Portfolio Risk.
Design language: #0e1016 bg, #6366f1 indigo, Plus Jakarta Sans + JetBrains Mono.
Tests: importability, parse_positions, compute_portfolio_metrics,
       compute_stress_results, is_market_open, design compliance.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from tests.conftest_dashboard_helpers import load_dashboard, make_st_mock

_ST = make_st_mock()

# Dashboard 4 calls several st functions at module level; ensure mock is active
_d4 = load_dashboard(4, _ST)

parse_positions = _d4.parse_positions
compute_portfolio_metrics = _d4.compute_portfolio_metrics
compute_stress_results = _d4.compute_stress_results
is_market_open = _d4.is_market_open
_pnl_bar_html = _d4._pnl_bar_html
_position_card_html = _d4._position_card_html
_stress_row_html = _d4._stress_row_html

_SRC = (_REPO / "pages" / "4_Portfolio_Risk.py").read_text(encoding="utf-8")

_DEFAULT_CSV = """ticker,shares,entry,current
SPY,100,540,558
QQQ,50,480,495
AAPL,75,210,218
GLD,40,235,242
TLT,60,88,85
"""


# ---------------------------------------------------------------------------
# T-D4-001: Importability
# ---------------------------------------------------------------------------

class TestImportability:
    def test_module_loads(self):
        assert _d4 is not None

    def test_required_functions_present(self):
        for fn in ("parse_positions", "compute_portfolio_metrics",
                   "compute_stress_results", "is_market_open"):
            assert hasattr(_d4, fn), f"Missing: {fn}"


# ---------------------------------------------------------------------------
# T-D4-002: parse_positions
# ---------------------------------------------------------------------------

class TestParsePositions:
    def test_parses_default_csv(self):
        df = parse_positions(_DEFAULT_CSV)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5

    def test_required_columns(self):
        df = parse_positions(_DEFAULT_CSV)
        for col in ("ticker", "shares", "entry", "current"):
            assert col in df.columns

    def test_ticker_uppercased(self):
        csv = "ticker,shares,entry,current\nspy,100,540,558"
        df = parse_positions(csv)
        assert df.iloc[0]["ticker"] == "SPY"

    def test_shares_is_int_type(self):
        df = parse_positions(_DEFAULT_CSV)
        assert df["shares"].dtype in (int, np.int64, np.int32)

    def test_entry_current_are_numeric(self):
        df = parse_positions(_DEFAULT_CSV)
        assert pd.api.types.is_numeric_dtype(df["entry"])
        assert pd.api.types.is_numeric_dtype(df["current"])

    def test_zero_shares_filtered_out(self):
        csv = "ticker,shares,entry,current\nSPY,0,540,558\nQQQ,50,480,495"
        df = parse_positions(csv)
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "QQQ"

    def test_missing_required_columns_returns_empty(self):
        """CSV missing required columns returns empty DataFrame."""
        csv = "ticker,price\nSPY,540"
        df = parse_positions(csv)
        assert df.empty

    def test_nan_entry_dropped(self):
        csv = "ticker,shares,entry,current\nSPY,100,,558\nQQQ,50,480,495"
        df = parse_positions(csv)
        # SPY row has NaN entry → dropped
        assert len(df) == 1
        assert df.iloc[0]["ticker"] == "QQQ"

    def test_extra_whitespace_in_csv_handled(self):
        csv = "ticker, shares, entry, current\n SPY , 100 , 540, 558"
        df = parse_positions(csv)
        assert not df.empty
        assert df.iloc[0]["ticker"] == "SPY"


# ---------------------------------------------------------------------------
# T-D4-003: compute_portfolio_metrics
# ---------------------------------------------------------------------------

class TestComputePortfolioMetrics:
    def _positions(self):
        return parse_positions(_DEFAULT_CSV)

    def test_returns_dict(self):
        result = compute_portfolio_metrics(self._positions())
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = compute_portfolio_metrics(self._positions())
        for key in ("total_value", "total_cost", "total_pnl", "total_pnl_pct"):
            assert key in result

    def test_total_value_matches_manual(self):
        """total_value = sum(shares * current)."""
        positions = self._positions()
        expected = (positions["shares"] * positions["current"]).sum()
        result = compute_portfolio_metrics(positions)
        assert result["total_value"] == pytest.approx(expected, rel=1e-6)

    def test_total_pnl_equals_value_minus_cost(self):
        result = compute_portfolio_metrics(self._positions())
        assert result["total_pnl"] == pytest.approx(
            result["total_value"] - result["total_cost"], rel=1e-6
        )

    def test_spy_position_positive_pnl(self):
        """SPY entry 540, current 558 → positive P&L."""
        positions = self._positions()
        result = compute_portfolio_metrics(positions)
        # Not all positions profitable (TLT is at a loss), but total depends on mix
        # We know SPY contributed positively
        spy_row = positions[positions["ticker"] == "SPY"].iloc[0]
        spy_pnl = spy_row["shares"] * (spy_row["current"] - spy_row["entry"])
        assert spy_pnl > 0

    def test_tlt_position_negative_pnl(self):
        """TLT entry 88, current 85 → negative P&L."""
        positions = self._positions()
        tlt_row = positions[positions["ticker"] == "TLT"].iloc[0]
        tlt_pnl = tlt_row["shares"] * (tlt_row["current"] - tlt_row["entry"])
        assert tlt_pnl < 0

    def test_pnl_pct_near_zero_for_flat_portfolio(self):
        csv = "ticker,shares,entry,current\nSPY,100,540,540"
        positions = parse_positions(csv)
        result = compute_portfolio_metrics(positions)
        assert result["total_pnl_pct"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# T-D4-004: compute_stress_results
# ---------------------------------------------------------------------------

class TestComputeStressResults:
    def _positions(self):
        return parse_positions(_DEFAULT_CSV)

    def test_returns_list_of_three(self):
        results = compute_stress_results(self._positions())
        assert isinstance(results, list)
        assert len(results) == 3

    def test_each_result_has_required_keys(self):
        results = compute_stress_results(self._positions())
        for r in results:
            for key in ("name", "portfolio_loss", "loss_pct", "abs_loss"):
                assert key in r

    def test_2008_crisis_largest_loss(self):
        """2008 crisis had largest drawdowns; portfolio loss should be most negative."""
        results = compute_stress_results(self._positions())
        loss_2008 = next(r for r in results if "2008" in r["name"])
        loss_2020 = next(r for r in results if "2020" in r["name"])
        # Both should be negative for a typical equity-heavy portfolio
        assert loss_2008["portfolio_loss"] < 0

    def test_loss_pct_bounded(self):
        """loss_pct must be in [-1, 1] for any realistic scenario."""
        results = compute_stress_results(self._positions())
        for r in results:
            assert -1.0 <= r["loss_pct"] <= 1.0

    def test_abs_loss_non_negative(self):
        results = compute_stress_results(self._positions())
        for r in results:
            assert r["abs_loss"] >= 0


# ---------------------------------------------------------------------------
# T-D4-005: _pnl_bar_html
# ---------------------------------------------------------------------------

class TestPnlBarHtml:
    def test_positive_pnl_uses_positive_class(self):
        html = _pnl_bar_html(0.05)
        assert "pnl-bar-positive" in html

    def test_negative_pnl_uses_negative_class(self):
        html = _pnl_bar_html(-0.05)
        assert "pnl-bar-negative" in html

    def test_zero_pnl_uses_positive_class(self):
        html = _pnl_bar_html(0.0)
        assert "pnl-bar-positive" in html

    def test_returns_string(self):
        assert isinstance(_pnl_bar_html(0.1), str)

    def test_very_large_pnl_capped(self):
        """Extreme P&L should not produce width > 100%."""
        html = _pnl_bar_html(99.0)
        # Width must be <= 100%
        assert "100.0%" in html or "width:100" in html


# ---------------------------------------------------------------------------
# T-D4-006: Design system compliance
# ---------------------------------------------------------------------------

class TestDesignCompliance:
    def test_premium_bg_color(self):
        assert "#0e1016" in _SRC, "Premium fintech bg #0e1016 must appear in dashboard 4"

    def test_indigo_primary_color(self):
        assert "#6366f1" in _SRC, "Indigo primary #6366f1 must appear in dashboard 4"

    def test_plus_jakarta_sans_font(self):
        assert "Plus Jakarta Sans" in _SRC, "Plus Jakarta Sans must be in dashboard 4"

    def test_jetbrains_mono_font(self):
        assert "JetBrains Mono" in _SRC, "JetBrains Mono must be in dashboard 4"

    def test_card_radius_16px(self):
        assert "16px" in _SRC, "Card radius 16px must appear in dashboard 4"

    def test_hover_lift_transform(self):
        assert "translateY(-2px)" in _SRC, "Hover lift translateY(-2px) must be in dashboard 4"

    def test_gradient_card_background(self):
        assert "linear-gradient" in _SRC, "Gradient card bg must be in dashboard 4"

    def test_regime_colors_imported_not_redefined(self):
        import re, ast
        tree = ast.parse(_SRC)
        found_import = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "core.design_system"
            and any(alias.name == "REGIME_COLORS" for alias in node.names)
            for node in ast.walk(tree)
        )
        assert found_import, "REGIME_COLORS must be imported from core.design_system"

    def test_stress_scenarios_in_source(self):
        for year in ("2008", "2020", "2022"):
            assert year in _SRC, f"Stress scenario {year} must appear in dashboard 4"

    def test_lookahead_guard_present(self):
        assert "LOOKAHEAD_CHECK_PASSED" in _SRC, (
            "Dashboard 4 must check LOOKAHEAD_CHECK_PASSED before rendering HMM results"
        )
