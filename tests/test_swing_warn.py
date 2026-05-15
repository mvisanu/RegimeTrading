"""Tests for swing/warn.py — regime-aware advisory."""
import json
import pytest
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@pytest.fixture
def mock_regime_low(monkeypatch):
    """Patch HMM + yfinance to return Low Vol regime."""
    import swing.warn as w

    @dataclass
    class FakeResult:
        labels: list
        stable_labels: list
        confidence: object

    monkeypatch.setattr(w, "_get_current_regime", lambda: ("Low Vol", 0.85))


@pytest.fixture
def mock_regime_high(monkeypatch):
    import swing.warn as w
    monkeypatch.setattr(w, "_get_current_regime", lambda: ("High Vol", 0.78))


@pytest.fixture
def stats_with_data(tmp_path, monkeypatch):
    """Write pattern_stats.json with known values and patch load()."""
    import swing.warn as w
    import swing.stats as s

    stats = {
        "gap-up": {
            "Low Vol":  {"n": 10, "avg_tp_pct": 55, "avg_pnl_pct": 3.2, "win_rate": 0.70},
            "High Vol": {"n":  5, "avg_tp_pct": 10, "avg_pnl_pct": -2.1, "win_rate": 0.20},
        },
        "downtrend-break": {
            "Low Vol":  {"n":  2, "avg_tp_pct": 40, "avg_pnl_pct": 1.0, "win_rate": 0.50},
        },
        "_meta": {"total_outcomes": 17},
    }
    monkeypatch.setattr(s, "load", lambda *a, **kw: stats)
    return stats


def test_clear_result_when_win_rate_above_threshold(
    mock_regime_low, stats_with_data
):
    from swing.warn import check
    result = check("AAPL", "gap-up", 0.7)
    assert result.should_warn is False
    assert result.regime == "Low Vol"
    assert result.win_rate == 0.70
    assert result.n == 10


def test_warn_when_win_rate_below_threshold(
    mock_regime_high, stats_with_data
):
    from swing.warn import check
    result = check("SNAP", "gap-up", 0.6)
    assert result.should_warn is True
    assert result.win_rate == 0.20
    assert "CAUTION" in result.message


def test_no_warn_when_insufficient_data(
    mock_regime_low, stats_with_data
):
    from swing.warn import check
    result = check("WOLF", "downtrend-break", 0.76)
    assert result.should_warn is False
    assert result.win_rate is None
    assert "Insufficient" in result.message
    assert result.n == 2


def test_no_warn_when_pattern_not_in_stats(
    mock_regime_low, stats_with_data
):
    from swing.warn import check
    result = check("XYZ", "oversold-bounce", 0.6)
    assert result.should_warn is False
    assert result.n == 0
    assert "Insufficient" in result.message


def test_result_fields_populated(mock_regime_low, stats_with_data):
    from swing.warn import check
    result = check("AAPL", "gap-up", 0.7)
    assert result.symbol == "AAPL"
    assert result.pattern == "gap-up"
    assert result.regime_confidence == 0.85
    assert result.avg_tp_pct == 55
