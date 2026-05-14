"""
tests/test_dashboard_1.py — Unit tests for pages/1_Regime_Detection.py.

Dashboard 1: Bloomberg Terminal Regime Detection.
Design language: Bloomberg terminal — #0e1117 bg, #00d4ff cyan, monospace.
Tests: importability, build_regime_segments, compute_regime_stats, design compliance.
"""

from __future__ import annotations

import os
import sys
import ast
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure repo root is on the path before anything else
_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

# ---------------------------------------------------------------------------
# Streamlit mock — must be in sys.modules BEFORE dashboard import
# ---------------------------------------------------------------------------
from tests.conftest_dashboard_helpers import load_dashboard, make_ohlcv_df, make_st_mock

_ST_MOCK = make_st_mock()

# ---------------------------------------------------------------------------
# Load dashboard module once for the session
# ---------------------------------------------------------------------------
_d1 = load_dashboard(1, _ST_MOCK)
build_regime_segments = _d1.build_regime_segments
compute_regime_stats = _d1.compute_regime_stats

_SRC = ((_REPO / "pages" / "1_Regime_Detection.py").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# T-D1-001: Importability
# ---------------------------------------------------------------------------

class TestImportability:
    def test_module_loads_without_error(self):
        """Dashboard 1 must import without raising any exception."""
        assert _d1 is not None

    def test_required_functions_exported(self):
        """Key functions must be present in the loaded module."""
        for name in ("build_regime_segments", "compute_regime_stats",
                     "render_sidebar", "render_hero_chart",
                     "render_regime_stats", "render_confidence_timeline"):
            assert hasattr(_d1, name), f"Missing function: {name}"


# ---------------------------------------------------------------------------
# T-D1-002: build_regime_segments
# ---------------------------------------------------------------------------

class TestBuildRegimeSegments:
    def test_single_regime(self):
        dates = list(pd.date_range("2021-01-01", periods=5))
        segs = build_regime_segments(dates, ["Low Vol"] * 5)
        assert len(segs) == 1
        start, end, label = segs[0]
        assert label == "Low Vol"
        assert start == dates[0]
        assert end == dates[-1]

    def test_two_regimes(self):
        dates = list(pd.date_range("2021-01-01", periods=6))
        labels = ["Low Vol"] * 3 + ["High Vol"] * 3
        segs = build_regime_segments(dates, labels)
        assert len(segs) == 2
        assert segs[0][2] == "Low Vol"
        assert segs[1][2] == "High Vol"

    def test_alternating_labels_gives_n_segments(self):
        dates = list(pd.date_range("2021-01-01", periods=4))
        labels = ["Low Vol", "High Vol", "Low Vol", "High Vol"]
        segs = build_regime_segments(dates, labels)
        assert len(segs) == 4

    def test_empty_input_returns_empty(self):
        assert build_regime_segments([], []) == []

    def test_all_uncertain(self):
        dates = list(pd.date_range("2021-01-01", periods=3))
        segs = build_regime_segments(dates, ["Uncertain"] * 3)
        assert len(segs) == 1
        assert segs[0][2] == "Uncertain"

    def test_each_segment_is_3_tuple(self):
        dates = list(pd.date_range("2021-01-01", periods=3))
        labels = ["Low Vol", "Low Vol", "High Vol"]
        segs = build_regime_segments(dates, labels)
        for item in segs:
            assert isinstance(item, tuple) and len(item) == 3

    def test_start_end_ordering(self):
        """Each segment's start date must be <= its end date."""
        dates = list(pd.date_range("2021-01-01", periods=10))
        labels = ["Low Vol"] * 5 + ["High Vol"] * 5
        segs = build_regime_segments(dates, labels)
        for start, end, _ in segs:
            assert start <= end


# ---------------------------------------------------------------------------
# T-D1-003: compute_regime_stats
# ---------------------------------------------------------------------------

class TestComputeRegimeStats:
    def test_returns_dict(self):
        df = make_ohlcv_df(60)
        stats = compute_regime_stats(df, ["Low Vol"] * 60)
        assert isinstance(stats, dict)

    def test_keys_match_unique_labels(self):
        df = make_ohlcv_df(60)
        labels = ["Low Vol"] * 30 + ["High Vol"] * 30
        stats = compute_regime_stats(df, labels)
        assert set(stats.keys()) == {"Low Vol", "High Vol"}

    def test_required_subkeys(self):
        df = make_ohlcv_df(80)
        stats = compute_regime_stats(df, ["Low Vol"] * 80)
        entry = stats["Low Vol"]
        for key in ("mean_ret", "mean_vol", "pct_time", "count"):
            assert key in entry

    def test_pct_time_sums_to_100(self):
        df = make_ohlcv_df(90)
        labels = ["Low Vol"] * 30 + ["Medium Vol"] * 30 + ["High Vol"] * 30
        stats = compute_regime_stats(df, labels)
        total = sum(v["pct_time"] for v in stats.values())
        assert abs(total - 100.0) < 1e-6

    def test_count_matches_label_frequency(self):
        df = make_ohlcv_df(50)
        labels = ["Low Vol"] * 20 + ["Extreme Vol"] * 30
        stats = compute_regime_stats(df, labels)
        assert stats["Low Vol"]["count"] == 20
        assert stats["Extreme Vol"]["count"] == 30

    def test_mean_vol_non_negative(self):
        df = make_ohlcv_df(100)
        stats = compute_regime_stats(df, ["Low Vol"] * 100)
        assert stats["Low Vol"]["mean_vol"] >= 0.0

    def test_uncertain_label_handled(self):
        df = make_ohlcv_df(50)
        stats = compute_regime_stats(df, ["Uncertain"] * 50)
        assert "Uncertain" in stats

    def test_single_label_single_entry(self):
        df = make_ohlcv_df(10)
        stats = compute_regime_stats(df, ["Extreme Vol"] * 10)
        assert len(stats) == 1


# ---------------------------------------------------------------------------
# T-D1-004: Design system compliance (source inspection)
# ---------------------------------------------------------------------------

class TestDesignSystemCompliance:
    def test_regime_colors_imported_from_core(self):
        """REGIME_COLORS must be imported from core.design_system, not defined locally."""
        tree = ast.parse(_SRC)
        found = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "core.design_system"
            and any(alias.name == "REGIME_COLORS" for alias in node.names)
            for node in ast.walk(tree)
        )
        assert found, "REGIME_COLORS must be imported from core.design_system"

    def test_no_local_regime_colors_assignment(self):
        """REGIME_COLORS must not be assigned locally in dashboard 1."""
        import re
        assignments = re.findall(r"^REGIME_COLORS\s*=\s*\{", _SRC, re.MULTILINE)
        assert not assignments, "REGIME_COLORS must not be redefined in dashboard 1"

    def test_bloomberg_bg_color_in_css(self):
        assert "#0e1117" in _SRC, "Bloomberg terminal bg #0e1117 must appear in dashboard 1"

    def test_cyan_accent_in_source(self):
        assert "#00d4ff" in _SRC, "Cyan accent #00d4ff must appear in dashboard 1"

    def test_monospace_font_referenced(self):
        assert "monospace" in _SRC.lower(), "Monospace font must be referenced in dashboard 1"

    def test_lookahead_badge_referenced(self):
        upper = _SRC.upper()
        assert "LOOKAHEAD" in upper or "LOOK-AHEAD" in upper, (
            "Dashboard 1 must reference a look-ahead verification badge"
        )

    def test_verify_imported(self):
        """core.verify must be imported (for LOOKAHEAD_CHECK_PASSED guard)."""
        assert "verify" in _SRC, "core.verify must be imported in dashboard 1"

    def test_set_page_config_called(self):
        """st.set_page_config must appear as the first Streamlit call."""
        assert "st.set_page_config" in _SRC

    def test_angular_border_radius_in_css(self):
        """Bloomberg style uses angular borders (radius 2px), not rounded."""
        assert "border-radius: 2px" in _SRC or "border-radius:2px" in _SRC


# ---------------------------------------------------------------------------
# T-D1-005: REGIME_COLORS values (from design_system)
# ---------------------------------------------------------------------------

class TestRegimeColorsValues:
    def test_all_five_regimes_present(self):
        from core.design_system import REGIME_COLORS
        expected = {"Low Vol", "Medium Vol", "High Vol", "Extreme Vol", "Uncertain"}
        assert expected.issubset(set(REGIME_COLORS.keys()))

    def test_low_vol_is_green(self):
        from core.design_system import REGIME_COLORS
        assert REGIME_COLORS["Low Vol"] == "#10b981"

    def test_medium_vol_is_blue(self):
        from core.design_system import REGIME_COLORS
        assert REGIME_COLORS["Medium Vol"] == "#3b82f6"

    def test_high_vol_is_amber(self):
        from core.design_system import REGIME_COLORS
        assert REGIME_COLORS["High Vol"] == "#f59e0b"

    def test_extreme_vol_is_red(self):
        from core.design_system import REGIME_COLORS
        assert REGIME_COLORS["Extreme Vol"] == "#ef4444"

    def test_uncertain_is_slate(self):
        from core.design_system import REGIME_COLORS
        assert REGIME_COLORS["Uncertain"] == "#64748b"
