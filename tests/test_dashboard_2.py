"""
tests/test_dashboard_2.py — Unit tests for pages/2_Monte_Carlo.py.

Dashboard 2: Deep Space Nebula Monte Carlo Simulation.
Design language: #060614 bg, Space Mono + DM Sans, 16px card radius.
Tests: importability, _max_drawdown, run_monte_carlo, generate_demo_trades,
       _compute_sharpe, overfitting flag logic, design compliance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from tests.conftest_dashboard_helpers import load_dashboard, make_st_mock

_ST = make_st_mock()
_d2 = load_dashboard(2, _ST)

_max_drawdown = _d2._max_drawdown
run_monte_carlo = _d2.run_monte_carlo
generate_demo_trades = _d2.generate_demo_trades
_compute_sharpe = _d2._compute_sharpe
render_overfit_warning = _d2.render_overfit_warning

_SRC = (_REPO / "pages" / "2_Monte_Carlo.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# T-D2-001: Importability
# ---------------------------------------------------------------------------

class TestImportability:
    def test_module_loads(self):
        assert _d2 is not None

    def test_required_names_present(self):
        for name in ("run_monte_carlo", "generate_demo_trades", "_max_drawdown",
                     "_compute_sharpe", "render_overfit_warning"):
            assert hasattr(_d2, name), f"Missing: {name}"


# ---------------------------------------------------------------------------
# T-D2-002: _max_drawdown
# ---------------------------------------------------------------------------

class TestMaxDrawdown:
    def test_flat_equity_zero_drawdown(self):
        equity = np.array([100.0, 100.0, 100.0])
        assert _max_drawdown(equity) == pytest.approx(0.0)

    def test_monotone_rising_zero_drawdown(self):
        equity = np.array([100.0, 110.0, 120.0, 130.0])
        assert _max_drawdown(equity) == pytest.approx(0.0)

    def test_known_drawdown(self):
        # Peak 200, trough 100 → drawdown = 50%
        equity = np.array([100.0, 200.0, 100.0])
        assert _max_drawdown(equity) == pytest.approx(0.5, abs=1e-6)

    def test_returns_float(self):
        equity = np.array([100.0, 90.0, 95.0])
        result = _max_drawdown(equity)
        assert isinstance(result, float)

    def test_non_negative(self):
        equity = np.array([100.0, 105.0, 80.0, 95.0])
        assert _max_drawdown(equity) >= 0.0

    def test_single_bar(self):
        assert _max_drawdown(np.array([100.0])) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# T-D2-003: generate_demo_trades
# ---------------------------------------------------------------------------

class TestGenerateDemoTrades:
    def test_returns_list(self):
        trades = generate_demo_trades(100)
        assert isinstance(trades, list)

    def test_correct_length(self):
        for n in (50, 100, 200):
            assert len(generate_demo_trades(n)) == n

    def test_deterministic_with_seed(self):
        a = generate_demo_trades(50, seed=42)
        b = generate_demo_trades(50, seed=42)
        assert a == b

    def test_different_seeds_differ(self):
        a = generate_demo_trades(50, seed=1)
        b = generate_demo_trades(50, seed=2)
        assert a != b

    def test_positive_edge_on_average(self):
        """Demo data is specified to have slight positive edge."""
        trades = generate_demo_trades(1000, seed=42)
        assert np.mean(trades) > 0.0, "Demo trades should have positive expected return"


# ---------------------------------------------------------------------------
# T-D2-004: run_monte_carlo
# ---------------------------------------------------------------------------

class TestRunMonteCarlo:
    def _trades(self):
        return generate_demo_trades(50, seed=42)

    def test_returns_dict(self):
        result = run_monte_carlo(self._trades(), n_sims=50)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        result = run_monte_carlo(self._trades(), n_sims=50)
        for key in ("all_curves", "all_final", "all_max_dd", "orig_equity",
                    "median_final", "p5_final", "p95_final",
                    "p_loss", "p_dd_20", "percentile_rank",
                    "starting_capital", "n_sims", "n_trades"):
            assert key in result, f"Missing key: {key}"

    def test_n_sims_respected(self):
        result = run_monte_carlo(self._trades(), n_sims=75)
        assert result["n_sims"] == 75

    def test_n_trades_matches_input(self):
        trades = self._trades()
        result = run_monte_carlo(trades, n_sims=50)
        assert result["n_trades"] == len(trades)

    def test_p_loss_in_unit_interval(self):
        result = run_monte_carlo(self._trades(), n_sims=100)
        assert 0.0 <= result["p_loss"] <= 1.0

    def test_p_dd_20_in_unit_interval(self):
        result = run_monte_carlo(self._trades(), n_sims=100)
        assert 0.0 <= result["p_dd_20"] <= 1.0

    def test_percentile_rank_in_unit_interval(self):
        result = run_monte_carlo(self._trades(), n_sims=100)
        assert 0.0 <= result["percentile_rank"] <= 1.0

    def test_p5_le_median_le_p95(self):
        result = run_monte_carlo(self._trades(), n_sims=200)
        assert result["p5_final"] <= result["median_final"] <= result["p95_final"]

    def test_all_curves_length_equals_n_sims(self):
        result = run_monte_carlo(self._trades(), n_sims=60)
        assert len(result["all_curves"]) == 60

    def test_each_curve_length_equals_n_trades(self):
        trades = self._trades()
        result = run_monte_carlo(trades, n_sims=20)
        for curve in result["all_curves"]:
            assert len(curve) == len(trades)

    def test_orig_equity_starts_above_zero(self):
        result = run_monte_carlo(self._trades(), n_sims=50)
        assert result["orig_equity"][0] > 0


# ---------------------------------------------------------------------------
# T-D2-005: _compute_sharpe
# ---------------------------------------------------------------------------

class TestComputeSharpe:
    def test_returns_float_for_valid_input(self):
        equity = np.cumprod(1.0 + np.full(252, 0.001))
        result = _compute_sharpe(equity, 1.0)
        assert isinstance(result, float)

    def test_returns_none_for_single_bar(self):
        assert _compute_sharpe(np.array([100.0]), 100.0) is None

    def test_flat_curve_returns_none(self):
        """Zero-std returns → denominator = 0 → None."""
        equity = np.full(50, 100.0)
        assert _compute_sharpe(equity, 100.0) is None

    def test_positive_drift_positive_sharpe(self):
        equity = np.cumprod(1.0 + np.full(252, 0.002))
        sharpe = _compute_sharpe(equity, 1.0)
        assert sharpe is not None and sharpe > 0


# ---------------------------------------------------------------------------
# T-D2-006: Overfitting flag (rank > 90th percentile triggers warning)
# ---------------------------------------------------------------------------

class TestOverfitWarning:
    def test_no_warning_below_90th(self):
        """rank <= 0.90 — render_overfit_warning exits early without rendering HTML."""
        from unittest.mock import MagicMock, patch
        result = {"percentile_rank": 0.85}
        mock_markdown = MagicMock()
        with patch.object(_d2.st, "markdown", mock_markdown):
            render_overfit_warning(result)
        mock_markdown.assert_not_called()

    def test_no_warning_at_exactly_90th(self):
        """rank == 0.90 — exactly at threshold → no warning (threshold is > 0.90)."""
        from unittest.mock import MagicMock, patch
        result = {"percentile_rank": 0.90}
        mock_markdown = MagicMock()
        with patch.object(_d2.st, "markdown", mock_markdown):
            render_overfit_warning(result)
        mock_markdown.assert_not_called()

    def test_warning_shown_above_90th(self):
        """rank > 0.90 — render_overfit_warning must produce HTML output."""
        from unittest.mock import MagicMock, patch
        result = {"percentile_rank": 0.95}
        mock_markdown = MagicMock()
        with patch.object(_d2.st, "markdown", mock_markdown):
            render_overfit_warning(result)
        mock_markdown.assert_called_once()


# ---------------------------------------------------------------------------
# T-D2-007: Design system compliance
# ---------------------------------------------------------------------------

class TestDesignCompliance:
    def test_deep_space_bg_color(self):
        assert "#060614" in _SRC, "Deep space nebula bg #060614 must appear in dashboard 2"

    def test_space_mono_font_referenced(self):
        assert "Space Mono" in _SRC, "Space Mono font must be referenced in dashboard 2"

    def test_dm_sans_font_referenced(self):
        assert "DM Sans" in _SRC, "DM Sans font must be referenced in dashboard 2"

    def test_card_radius_16px(self):
        assert "16px" in _SRC, "Card radius 16px must appear in dashboard 2"

    def test_primary_blue_color(self):
        assert "#4d8eff" in _SRC, "Primary #4d8eff must appear in dashboard 2"

    def test_radial_gradient_background(self):
        assert "radial-gradient" in _SRC, (
            "Dashboard 2 must use radial-gradient for the nebula background"
        )

    def test_no_local_regime_colors_definition(self):
        import re
        assignments = re.findall(r"^REGIME_COLORS\s*=\s*\{", _SRC, re.MULTILINE)
        assert not assignments, "REGIME_COLORS must not be redefined in dashboard 2"
