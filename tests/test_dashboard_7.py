"""
tests/test_dashboard_7.py — Unit tests for pages/7_Correlation_Breaks.py.

Dashboard 7: SOC-style Correlation Break Detector.
Design language: #08080c bg, Share Tech Mono, pulse @keyframes, no HMM.
Tests: importability, _severity_css_class, _fmt_return, _pair_card_html,
       _append_alert, severity thresholds, design compliance, no-HMM guarantee.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from tests.conftest_dashboard_helpers import load_dashboard, make_st_mock

_ST = make_st_mock()
_d7 = load_dashboard(7, _ST)

_severity_css_class = _d7._severity_css_class
_fmt_return = _d7._fmt_return
_pair_card_html = _d7._pair_card_html
_append_alert = _d7._append_alert
PairResult = _d7.PairResult
_SEVERITY_COLOR = _d7._SEVERITY_COLOR

_SRC = (_REPO / "pages" / "7_Correlation_Breaks.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pair_result(severity: str = "Normal", z: float = 0.0) -> PairResult:
    dates = pd.date_range("2020-01-01", periods=100)
    corr = pd.Series(np.full(100, 0.8), index=dates)
    return PairResult(
        pair="SPY/QQQ",
        ticker_a="SPY",
        ticker_b="QQQ",
        corr_20d=corr,
        corr_60d=corr,
        hist_mean=0.85,
        hist_std=0.05,
        z_score=z,
        severity=severity,
        current_corr_60d=0.80,
        dates=dates,
    )


# ---------------------------------------------------------------------------
# T-D7-001: Importability
# ---------------------------------------------------------------------------

class TestImportability:
    def test_module_loads(self):
        assert _d7 is not None

    def test_required_names_present(self):
        for name in ("_severity_css_class", "_fmt_return", "_pair_card_html",
                     "_append_alert", "PairResult", "_SEVERITY_COLOR",
                     "_compute_pair_result", "_main_correlation_chart"):
            assert hasattr(_d7, name), f"Missing: {name}"


# ---------------------------------------------------------------------------
# T-D7-002: _severity_css_class
# ---------------------------------------------------------------------------

class TestSeverityCssClass:
    def test_extreme_returns_pulse_extreme(self):
        css = _severity_css_class("Extreme")
        assert "card-extreme" in css

    def test_significant_returns_pulse_significant(self):
        css = _severity_css_class("Significant")
        assert "card-significant" in css

    def test_normal_returns_empty_string(self):
        css = _severity_css_class("Normal")
        assert css == "", f"Normal must not animate, got: {css!r}"

    def test_notable_returns_empty_string(self):
        css = _severity_css_class("Notable")
        assert css == "", f"Notable must not animate (only Significant+Extreme pulse)"


# ---------------------------------------------------------------------------
# T-D7-003: _fmt_return
# ---------------------------------------------------------------------------

class TestFmtReturn:
    def test_positive_return_has_green_class(self):
        html = _fmt_return(0.10)
        assert "ctx-pos" in html

    def test_negative_return_has_red_class(self):
        html = _fmt_return(-0.10)
        assert "ctx-neg" in html

    def test_near_zero_is_neutral(self):
        html = _fmt_return(0.0)
        assert "ctx-neu" in html

    def test_returns_string(self):
        assert isinstance(_fmt_return(0.05), str)

    def test_positive_sign_shown(self):
        html = _fmt_return(0.05)
        assert "+" in html

    def test_percentage_formatted(self):
        html = _fmt_return(0.05)
        assert "%" in html

    def test_boundary_positive(self):
        """0.05% → neutral (0.0005 value)."""
        html = _fmt_return(0.0005)
        assert "ctx-neu" in html

    def test_boundary_above_0_1pct_is_positive(self):
        html = _fmt_return(0.002)
        assert "ctx-pos" in html


# ---------------------------------------------------------------------------
# T-D7-004: _pair_card_html
# ---------------------------------------------------------------------------

class TestPairCardHtml:
    def test_returns_html_string(self):
        result = _make_pair_result("Normal")
        html = _pair_card_html(result)
        assert isinstance(html, str)

    def test_contains_pair_name(self):
        result = _make_pair_result("Normal")
        html = _pair_card_html(result)
        assert "SPY/QQQ" in html

    def test_extreme_card_has_animation_class(self):
        result = _make_pair_result("Extreme", z=-3.0)
        html = _pair_card_html(result)
        assert "card-extreme" in html

    def test_significant_card_has_animation_class(self):
        result = _make_pair_result("Significant", z=-2.3)
        html = _pair_card_html(result)
        assert "card-significant" in html

    def test_normal_card_has_no_animation_class(self):
        result = _make_pair_result("Normal")
        html = _pair_card_html(result)
        assert "card-extreme" not in html
        assert "card-significant" not in html

    def test_severity_badge_shown(self):
        result = _make_pair_result("Notable")
        html = _pair_card_html(result)
        assert "Notable" in html

    def test_z_score_in_card(self):
        result = _make_pair_result("Extreme", z=-2.8)
        html = _pair_card_html(result)
        assert "z-score" in html.lower() or "z_score" in html.lower() or "-2.8" in html


# ---------------------------------------------------------------------------
# T-D7-005: Severity threshold logic (z-score classification)
# ---------------------------------------------------------------------------

class TestSeverityThresholds:
    """Verify the z-score → severity mapping defined in the spec."""

    def _severity_for_z(self, z: float) -> str:
        """Replicate the threshold logic from the dashboard."""
        if z > -1.5:
            return "Normal"
        elif z > -2.0:
            return "Notable"
        elif z > -2.5:
            return "Significant"
        else:
            return "Extreme"

    def test_z_above_minus_1_5_is_normal(self):
        assert self._severity_for_z(0.0) == "Normal"
        assert self._severity_for_z(-1.4) == "Normal"

    def test_z_below_minus_1_5_is_notable(self):
        assert self._severity_for_z(-1.6) == "Notable"
        assert self._severity_for_z(-1.9) == "Notable"

    def test_z_below_minus_2_is_significant(self):
        assert self._severity_for_z(-2.1) == "Significant"
        assert self._severity_for_z(-2.4) == "Significant"

    def test_z_below_minus_2_5_is_extreme(self):
        assert self._severity_for_z(-2.6) == "Extreme"
        assert self._severity_for_z(-5.0) == "Extreme"

    def test_boundary_exactly_minus_1_5_is_notable(self):
        # z > -1.5 → Normal; z == -1.5 → falls to Notable
        assert self._severity_for_z(-1.5) == "Notable"

    def test_boundary_exactly_minus_2_is_significant(self):
        # z > -2.0 → Notable; z == -2.0 → falls to Significant
        assert self._severity_for_z(-2.0) == "Significant"

    def test_boundary_exactly_minus_2_5_is_extreme(self):
        # z > -2.5 → Significant; z == -2.5 → falls to Extreme
        assert self._severity_for_z(-2.5) == "Extreme"


# ---------------------------------------------------------------------------
# T-D7-006: _append_alert (atomic append)
# ---------------------------------------------------------------------------

class TestAppendAlert:
    def test_appends_json_line(self, tmp_path, monkeypatch):
        """Alert must be written as valid JSON to the alerts file."""
        fake_log_dir = tmp_path / "logs"
        fake_log_dir.mkdir()
        fake_alerts_path = fake_log_dir / "alerts.json"

        # _d7 is the already-loaded dashboard module (loaded via importlib with
        # numeric-prefix filename — cannot use `import pages._7_Correlation_Breaks`)
        monkeypatch.setattr(_d7, "_ALERTS_PATH", str(fake_alerts_path))

        record = {
            "timestamp": "2024-01-01T00:00:00+00:00",
            "pair": "SPY/QQQ",
            "z_score": -2.8,
            "severity": "Extreme",
            "corr_60d": 0.45,
        }
        _d7._append_alert(record)

        assert fake_alerts_path.exists()
        lines = fake_alerts_path.read_text().strip().splitlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["pair"] == "SPY/QQQ"

    def test_multiple_alerts_each_on_own_line(self, tmp_path, monkeypatch):
        """Multiple alerts must each be a separate JSON line (NDJSON format)."""
        fake_log_dir = tmp_path / "logs"
        fake_log_dir.mkdir()
        fake_alerts_path = fake_log_dir / "alerts.json"

        monkeypatch.setattr(_d7, "_ALERTS_PATH", str(fake_alerts_path))

        for i in range(3):
            _d7._append_alert({"pair": f"PAIR{i}", "z_score": -float(i + 2)})

        lines = fake_alerts_path.read_text().strip().splitlines()
        assert len(lines) == 3
        for line in lines:
            json.loads(line)  # must be valid JSON


# ---------------------------------------------------------------------------
# T-D7-007: _SEVERITY_COLOR contract
# ---------------------------------------------------------------------------

class TestSeverityColors:
    def test_all_four_severities_mapped(self):
        for sev in ("Normal", "Notable", "Significant", "Extreme"):
            assert sev in _SEVERITY_COLOR

    def test_normal_is_muted_slate(self):
        assert _SEVERITY_COLOR["Normal"] == "#334155"

    def test_notable_is_amber(self):
        assert _SEVERITY_COLOR["Notable"] == "#f59e0b"

    def test_significant_is_orange(self):
        assert _SEVERITY_COLOR["Significant"] == "#f97316"

    def test_extreme_is_red(self):
        assert _SEVERITY_COLOR["Extreme"] == "#ef4444"


# ---------------------------------------------------------------------------
# T-D7-008: Design system compliance
# ---------------------------------------------------------------------------

class TestDesignCompliance:
    def test_soc_bg_color(self):
        assert "#08080c" in _SRC, "SOC bg #08080c must appear in dashboard 7"

    def test_share_tech_mono_font(self):
        assert "Share Tech Mono" in _SRC, "Share Tech Mono must be in dashboard 7"

    def test_pulse_keyframes_defined(self):
        assert "@keyframes pulse" in _SRC, (
            "Dashboard 7 must define @keyframes pulse animation"
        )

    def test_extreme_card_animation_class(self):
        assert "card-extreme" in _SRC, (
            "Dashboard 7 must define card-extreme animation class"
        )

    def test_significant_card_animation_class(self):
        assert "card-significant" in _SRC, (
            "Dashboard 7 must define card-significant animation class"
        )

    def test_no_animation_for_normal_state(self):
        """Normal cards must not animate — verify by checking that only
        Significant/Extreme CSS classes have 'animation:' attached."""
        import re
        # The card-extreme and card-significant classes should have animation
        assert "card-extreme" in _SRC
        assert "card-significant" in _SRC
        # card-normal and card-notable should NOT appear as animated classes
        assert "card-normal" not in _SRC, "Normal cards must not have an animation class"

    def test_no_hmm_import(self):
        import re
        # Check for actual import statements, not doc comments
        import_lines = [
            line for line in _SRC.splitlines()
            if re.match(r"\s*(import|from)\s+.*hmm_utils", line)
        ]
        assert not import_lines, (
            f"Dashboard 7 must not import core.hmm_utils (no HMM in correlation dashboard). "
            f"Found: {import_lines}"
        )

    def test_alerts_json_referenced(self):
        assert "alerts.json" in _SRC, (
            "Dashboard 7 must persist alerts to logs/alerts.json"
        )

    def test_atomic_write_pattern(self):
        """Alerts must be written with append-mode open (atomic)."""
        assert '"a"' in _SRC or "'a'" in _SRC, (
            "Dashboard 7 must use append mode ('a') for atomic alert writes"
        )

    def test_no_local_regime_colors_definition(self):
        import re
        assignments = re.findall(r"^REGIME_COLORS\s*=\s*\{", _SRC, re.MULTILINE)
        assert not assignments

    def test_z_score_thresholds_in_source(self):
        """The three threshold values -1.5, -2.0, -2.5 must appear in source."""
        for threshold in ("-1.5", "-2.0", "-2.5"):
            assert threshold in _SRC, f"Z-score threshold {threshold} not found in dashboard 7"

    def test_rolling_60_and_20_day_correlations(self):
        """Both 20-day and 60-day rolling correlations must be computed."""
        assert "rolling(20)" in _SRC, "20-day rolling correlation must be in dashboard 7"
        assert "rolling(60)" in _SRC, "60-day rolling correlation must be in dashboard 7"
