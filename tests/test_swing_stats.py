"""Tests for swing/stats.py — pure aggregation logic."""
import json
import os
import pytest
from pathlib import Path


@pytest.fixture
def tmp_outcomes(tmp_path):
    """Write outcomes.json to a temp dir and return the path."""
    records = [
        {"symbol": "AAPL", "pattern": "gap-up", "regime_at_add": "Low Vol",
         "tp_pct_complete": 100, "pnl_pct": 5.2, "outcome": "full_win"},
        {"symbol": "MSFT", "pattern": "gap-up", "regime_at_add": "Low Vol",
         "tp_pct_complete": 33, "pnl_pct": 2.1, "outcome": "partial_win"},
        {"symbol": "PLTR", "pattern": "gap-up", "regime_at_add": "Low Vol",
         "tp_pct_complete": 0, "pnl_pct": -3.0, "outcome": "loss"},
        {"symbol": "SNAP", "pattern": "gap-up", "regime_at_add": "High Vol",
         "tp_pct_complete": 0, "pnl_pct": -4.0, "outcome": "loss"},
        {"symbol": "NKE",  "pattern": "gap-up", "regime_at_add": "High Vol",
         "tp_pct_complete": 0, "pnl_pct": -2.5, "outcome": "loss"},
        # 5 downtrend-break in Low Vol with poor win rate
        *[{"symbol": f"X{i}", "pattern": "downtrend-break", "regime_at_add": "Low Vol",
           "tp_pct_complete": 0, "pnl_pct": -1.0, "outcome": "loss"} for i in range(5)],
    ]
    p = tmp_path / "outcomes.json"
    p.write_text(json.dumps(records))
    return p


@pytest.fixture
def tmp_stats(tmp_path):
    return tmp_path / "pattern_stats.json"


@pytest.fixture
def tmp_rules(tmp_path, monkeypatch):
    rules_path = tmp_path / "rules.md"
    import swing.stats as s
    monkeypatch.setattr(s, "_RULES_PATH", rules_path)
    return rules_path


def test_rebuild_creates_stats_file(tmp_outcomes, tmp_stats, tmp_rules):
    from swing.stats import rebuild
    rebuild(tmp_outcomes, tmp_stats)
    assert tmp_stats.exists()


def test_rebuild_groups_by_pattern_and_regime(tmp_outcomes, tmp_stats, tmp_rules):
    from swing.stats import rebuild, load
    rebuild(tmp_outcomes, tmp_stats)
    stats = load(tmp_stats)
    assert "gap-up" in stats
    assert "Low Vol" in stats["gap-up"]
    assert "High Vol" in stats["gap-up"]


def test_win_rate_counts_only_wins_and_partials(tmp_outcomes, tmp_stats, tmp_rules):
    from swing.stats import rebuild, load
    rebuild(tmp_outcomes, tmp_stats)
    stats = load(tmp_stats)
    # gap-up Low Vol: 2 wins (full+partial), 1 loss → win_rate = 2/3
    cell = stats["gap-up"]["Low Vol"]
    assert cell["n"] == 3
    assert abs(cell["win_rate"] - 2/3) < 0.01


def test_avg_tp_pct(tmp_outcomes, tmp_stats, tmp_rules):
    from swing.stats import rebuild, load
    rebuild(tmp_outcomes, tmp_stats)
    stats = load(tmp_stats)
    cell = stats["gap-up"]["Low Vol"]
    assert abs(cell["avg_tp_pct"] - (100 + 33 + 0) / 3) < 0.1


def test_below_threshold_appends_to_rules(tmp_outcomes, tmp_stats, tmp_rules):
    from swing.stats import rebuild
    rebuild(tmp_outcomes, tmp_stats)
    # downtrend-break Low Vol: 5 losses → win_rate=0.0, n=5 → should append
    assert tmp_rules.exists()
    content = tmp_rules.read_text()
    assert "downtrend-break" in content


def test_meta_includes_total_outcomes(tmp_outcomes, tmp_stats, tmp_rules):
    from swing.stats import rebuild, load
    rebuild(tmp_outcomes, tmp_stats)
    stats = load(tmp_stats)
    assert stats["_meta"]["total_outcomes"] == 10


def test_load_returns_empty_dict_when_missing(tmp_path):
    from swing.stats import load
    result = load(tmp_path / "nonexistent.json")
    assert result == {}


def test_rebuild_is_atomic(tmp_outcomes, tmp_stats, tmp_rules):
    from swing.stats import rebuild
    rebuild(tmp_outcomes, tmp_stats)
    tmp_file = tmp_stats.with_suffix(".tmp")
    assert not tmp_file.exists()
