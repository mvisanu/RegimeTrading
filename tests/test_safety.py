"""Tests for core.safety circuit breakers.

Pure function tests — no HMM model, no broker, no external state.
Each breaker is tested independently with values just above and just below
its trigger threshold.
"""

import time

import pytest

from core.safety import (
    _save_state,
    _STATE_FILE,
    check_all,
    check_concentration,
    check_daily_loss,
    check_max_drawdown,
    check_order_rate,
    check_weekly_loss,
    reset_state,
)


# ---------------------------------------------------------------------------
# Breaker 1: Daily loss limit (2%)
# ---------------------------------------------------------------------------

def test_daily_loss_trigger() -> None:
    """2.1% drop must trigger; 1.9% drop must not."""
    # 2.1% drop: (10000 - 9790) / 10000 = 0.021
    triggered, reason = check_daily_loss(equity_now=9790.0, equity_open=10000.0)
    assert triggered is True
    assert reason != ""

    # 1.9% drop: (10000 - 9810) / 10000 = 0.019
    triggered, reason = check_daily_loss(equity_now=9810.0, equity_open=10000.0)
    assert triggered is False
    assert reason == ""


def test_daily_loss_edge_zero_equity_open() -> None:
    """equity_open <= 0 must never trigger (guard against divide-by-zero)."""
    triggered, reason = check_daily_loss(equity_now=0.0, equity_open=0.0)
    assert triggered is False
    assert reason == ""


# ---------------------------------------------------------------------------
# Breaker 2: Weekly loss limit (5%)
# ---------------------------------------------------------------------------

def test_weekly_loss_trigger() -> None:
    """5.1% weekly drop must trigger; 4.9% must not."""
    # 5.1% drop: start=10000, end=9490 → (10000-9490)/10000 = 0.051
    triggered, reason = check_weekly_loss([10000.0, 9800.0, 9700.0, 9600.0, 9490.0])
    assert triggered is True
    assert reason != ""

    # 4.9% drop: start=10000, end=9510 → (10000-9510)/10000 = 0.049
    triggered, reason = check_weekly_loss([10000.0, 9900.0, 9800.0, 9700.0, 9510.0])
    assert triggered is False
    assert reason == ""


def test_weekly_loss_insufficient_history() -> None:
    """Single-element history must never trigger."""
    triggered, reason = check_weekly_loss([10000.0])
    assert triggered is False
    assert reason == ""


def test_weekly_loss_empty_history() -> None:
    """Empty history must never trigger."""
    triggered, reason = check_weekly_loss([])
    assert triggered is False
    assert reason == ""


# ---------------------------------------------------------------------------
# Breaker 3: Max drawdown limit (15%)
# ---------------------------------------------------------------------------

def test_max_drawdown_trigger() -> None:
    """15.1% drawdown from peak must trigger; 14.9% must not."""
    # 15.1% drawdown: (10000 - 8490) / 10000 = 0.151
    triggered, reason = check_max_drawdown(equity_now=8490.0, peak_equity=10000.0)
    assert triggered is True
    assert reason != ""

    # 14.9% drawdown: (10000 - 8510) / 10000 = 0.149
    triggered, reason = check_max_drawdown(equity_now=8510.0, peak_equity=10000.0)
    assert triggered is False
    assert reason == ""


def test_max_drawdown_zero_peak() -> None:
    """peak_equity <= 0 must never trigger."""
    triggered, reason = check_max_drawdown(equity_now=0.0, peak_equity=0.0)
    assert triggered is False
    assert reason == ""


# ---------------------------------------------------------------------------
# Breaker 4: Position concentration (25%)
# ---------------------------------------------------------------------------

def test_position_concentration_trigger() -> None:
    """25.1% concentration must trigger; 24.9% must not."""
    # 25.1%: 2510 / 10000 = 0.251
    triggered, reason = check_concentration(position_value=2510.0, portfolio_value=10000.0)
    assert triggered is True
    assert reason != ""

    # 24.9%: 2490 / 10000 = 0.249
    triggered, reason = check_concentration(position_value=2490.0, portfolio_value=10000.0)
    assert triggered is False
    assert reason == ""


def test_concentration_zero_portfolio() -> None:
    """portfolio_value <= 0 must never trigger."""
    triggered, reason = check_concentration(position_value=100.0, portfolio_value=0.0)
    assert triggered is False
    assert reason == ""


# ---------------------------------------------------------------------------
# Breaker 5: Order rate limit (20 orders / 60 seconds)
# ---------------------------------------------------------------------------

def test_order_rate_trigger() -> None:
    """21 timestamps in last 60s must trigger; 20 must not."""
    now = time.time()

    # 21 recent orders → triggered
    triggered, reason = check_order_rate([now] * 21)
    assert triggered is True
    assert reason != ""

    # 20 recent orders → not triggered (limit is > 20, so 20 is safe)
    triggered, reason = check_order_rate([now] * 20)
    assert triggered is False
    assert reason == ""


def test_order_rate_old_timestamps_ignored() -> None:
    """Timestamps older than 60s must not count toward the rate."""
    now = time.time()
    old_timestamps = [now - 120.0] * 30   # 30 orders, all >60s ago
    recent_timestamps = [now] * 5          # 5 current orders

    triggered, reason = check_order_rate(old_timestamps + recent_timestamps)
    assert triggered is False
    assert reason == ""


def test_order_rate_empty() -> None:
    """Empty timestamp list must not trigger."""
    triggered, reason = check_order_rate([])
    assert triggered is False
    assert reason == ""


# ---------------------------------------------------------------------------
# Aggregate: check_all
# ---------------------------------------------------------------------------

def test_check_all_returns_five_results() -> None:
    """check_all must return exactly 5 (triggered, reason) tuples."""
    now = time.time()
    results = check_all(
        equity_now=10000.0,
        equity_open=10000.0,
        equity_history=[10000.0, 10000.0],
        peak_equity=10000.0,
        position_value=1000.0,
        portfolio_value=10000.0,
        recent_order_timestamps=[now] * 5,
    )
    assert len(results) == 5
    for item in results:
        assert isinstance(item, tuple)
        assert len(item) == 2
        triggered, reason = item
        assert isinstance(triggered, bool)
        assert isinstance(reason, str)


def test_check_all_all_triggered() -> None:
    """check_all with extreme values must trigger all 5 breakers."""
    now = time.time()
    results = check_all(
        equity_now=5000.0,       # -50% daily loss
        equity_open=10000.0,
        equity_history=[10000.0, 5000.0],   # -50% weekly
        peak_equity=10000.0,     # -50% drawdown
        position_value=9000.0,   # 90% concentration
        portfolio_value=10000.0,
        recent_order_timestamps=[now] * 25,  # 25 orders
    )
    assert len(results) == 5
    for triggered, reason in results:
        assert triggered is True, f"Expected triggered but got: {reason!r}"
        assert reason != ""


def test_check_all_none_triggered() -> None:
    """check_all with healthy values must trigger no breakers."""
    now = time.time()
    results = check_all(
        equity_now=10000.0,
        equity_open=10000.0,
        equity_history=[10000.0, 10000.0],
        peak_equity=10000.0,
        position_value=1000.0,
        portfolio_value=10000.0,
        recent_order_timestamps=[now] * 5,
    )
    for triggered, reason in results:
        assert triggered is False
        assert reason == ""


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def test_reset_state() -> None:
    """reset_state() must complete without error and create the state file."""
    reset_state()
    assert _STATE_FILE.exists()


def test_atomic_save_creates_file(tmp_path, monkeypatch) -> None:
    """_save_state must create the state file atomically."""
    import core.safety as safety_module

    # Redirect the module-level paths to tmp_path
    fake_logs = tmp_path / "logs"
    fake_state = fake_logs / "safety_state.json"

    monkeypatch.setattr(safety_module, "_LOGS_DIR", fake_logs)
    monkeypatch.setattr(safety_module, "_STATE_FILE", fake_state)

    safety_module._save_state({"test_key": "test_value"})

    assert fake_state.exists()
    import json
    data = json.loads(fake_state.read_text())
    assert data["test_key"] == "test_value"


def test_load_state_missing_file(tmp_path, monkeypatch) -> None:
    """_load_state returns default state when file does not exist."""
    import core.safety as safety_module
    from core.safety import _DEFAULT_STATE

    fake_logs = tmp_path / "logs"
    fake_state = fake_logs / "safety_state.json"

    monkeypatch.setattr(safety_module, "_LOGS_DIR", fake_logs)
    monkeypatch.setattr(safety_module, "_STATE_FILE", fake_state)

    state = safety_module._load_state()
    assert state["daily_loss_halt"] == _DEFAULT_STATE["daily_loss_halt"]
    assert state["weekly_loss_halt_until"] == _DEFAULT_STATE["weekly_loss_halt_until"]
