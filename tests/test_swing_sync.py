"""Tests for swing/sync.py — Alpaca poller."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def watchlist(tmp_path):
    entries = [
        {"symbol": "PLTR", "pattern": "gap-up", "confidence": 0.6,
         "entry": 132.18, "stop": 128.76, "tp_ladder": [135.0, 140.0, 145.0],
         "added_ts": "2026-05-14T11:06:02-0400", "regime_at_add": "Low Vol",
         "status": "active", "tp_steps_hit": 0},
        {"symbol": "NKE", "pattern": "oversold-bounce", "confidence": 0.73,
         "entry": 42.33, "stop": 41.7, "tp_ladder": [],
         "added_ts": "2026-05-14T11:06:02-0400", "regime_at_add": "Low Vol",
         "status": "active", "tp_steps_hit": 0},
    ]
    p = tmp_path / "watchlist.json"
    p.write_text(json.dumps(entries))
    return p


@pytest.fixture
def outcomes_path(tmp_path):
    p = tmp_path / "outcomes.json"
    p.write_text("[]")
    return p


@pytest.fixture
def sync_state_path(tmp_path):
    return tmp_path / "sync_state.json"


@pytest.fixture
def sync_log_path(tmp_path):
    return tmp_path / "sync_log.md"


def _make_fake_order(symbol, filled_price, filled_at="2026-05-15T16:00:00Z"):
    order = MagicMock()
    order.symbol = symbol
    order.id = "abc12345"
    order.filled_avg_price = str(filled_price)
    order.filled_at = filled_at
    order.updated_at = filled_at
    return order


def test_sync_appends_outcome_for_matched_symbol(
    tmp_path, watchlist, outcomes_path, sync_state_path, sync_log_path, monkeypatch
):
    import swing.sync as s
    monkeypatch.setattr(s, "_WATCHLIST_PATH", watchlist)
    monkeypatch.setattr(s, "_OUTCOMES_PATH", outcomes_path)
    monkeypatch.setattr(s, "_SYNC_STATE_PATH", sync_state_path)
    monkeypatch.setattr(s, "_SYNC_LOG_PATH", sync_log_path)
    monkeypatch.setattr(s, "swing_stats", MagicMock())

    fake_order = _make_fake_order("PLTR", 137.5)

    with patch("swing.sync.TradingClient") as MockClient:
        instance = MockClient.return_value
        instance.get_orders.return_value = [fake_order]
        result = s.run()

    assert result.new_outcomes == 1
    assert "PLTR" in result.symbols_matched
    outcomes = json.loads(outcomes_path.read_text())
    assert len(outcomes) == 1
    assert outcomes[0]["symbol"] == "PLTR"


def test_sync_skips_unmatched_symbols(
    tmp_path, watchlist, outcomes_path, sync_state_path, sync_log_path, monkeypatch
):
    import swing.sync as s
    monkeypatch.setattr(s, "_WATCHLIST_PATH", watchlist)
    monkeypatch.setattr(s, "_OUTCOMES_PATH", outcomes_path)
    monkeypatch.setattr(s, "_SYNC_STATE_PATH", sync_state_path)
    monkeypatch.setattr(s, "_SYNC_LOG_PATH", sync_log_path)
    monkeypatch.setattr(s, "swing_stats", MagicMock())

    fake_order = _make_fake_order("AAPL", 180.0)

    with patch("swing.sync.TradingClient") as MockClient:
        instance = MockClient.return_value
        instance.get_orders.return_value = [fake_order]
        result = s.run()

    assert result.new_outcomes == 0


def test_tp_pct_complete_full_win(
    tmp_path, watchlist, outcomes_path, sync_state_path, sync_log_path, monkeypatch
):
    import swing.sync as s
    monkeypatch.setattr(s, "_WATCHLIST_PATH", watchlist)
    monkeypatch.setattr(s, "_OUTCOMES_PATH", outcomes_path)
    monkeypatch.setattr(s, "_SYNC_STATE_PATH", sync_state_path)
    monkeypatch.setattr(s, "_SYNC_LOG_PATH", sync_log_path)
    monkeypatch.setattr(s, "swing_stats", MagicMock())

    # PLTR close at 146 clears all 3 TP levels (135, 140, 145)
    fake_order = _make_fake_order("PLTR", 146.0)

    with patch("swing.sync.TradingClient") as MockClient:
        instance = MockClient.return_value
        instance.get_orders.return_value = [fake_order]
        s.run()

    outcomes = json.loads(outcomes_path.read_text())
    assert outcomes[0]["tp_pct_complete"] == 100
    assert outcomes[0]["outcome"] == "full_win"


def test_tp_pct_complete_loss(
    tmp_path, watchlist, outcomes_path, sync_state_path, sync_log_path, monkeypatch
):
    import swing.sync as s
    monkeypatch.setattr(s, "_WATCHLIST_PATH", watchlist)
    monkeypatch.setattr(s, "_OUTCOMES_PATH", outcomes_path)
    monkeypatch.setattr(s, "_SYNC_STATE_PATH", sync_state_path)
    monkeypatch.setattr(s, "_SYNC_LOG_PATH", sync_log_path)
    monkeypatch.setattr(s, "swing_stats", MagicMock())

    # PLTR close at 128 hits the stop (128.76)
    fake_order = _make_fake_order("PLTR", 128.0)

    with patch("swing.sync.TradingClient") as MockClient:
        instance = MockClient.return_value
        instance.get_orders.return_value = [fake_order]
        s.run()

    outcomes = json.loads(outcomes_path.read_text())
    assert outcomes[0]["tp_pct_complete"] == 0
    assert outcomes[0]["outcome"] == "loss"


def test_sync_is_idempotent(
    tmp_path, watchlist, outcomes_path, sync_state_path, sync_log_path, monkeypatch
):
    """Running sync twice with the same order should not create duplicate outcomes."""
    import swing.sync as s
    monkeypatch.setattr(s, "_WATCHLIST_PATH", watchlist)
    monkeypatch.setattr(s, "_OUTCOMES_PATH", outcomes_path)
    monkeypatch.setattr(s, "_SYNC_STATE_PATH", sync_state_path)
    monkeypatch.setattr(s, "_SYNC_LOG_PATH", sync_log_path)
    monkeypatch.setattr(s, "swing_stats", MagicMock())

    fake_order = _make_fake_order("PLTR", 137.5)

    with patch("swing.sync.TradingClient") as MockClient:
        instance = MockClient.return_value
        instance.get_orders.return_value = [fake_order]
        s.run()
        s.run()

    outcomes = json.loads(outcomes_path.read_text())
    assert len(outcomes) == 1


def test_alpaca_failure_returns_error(
    tmp_path, watchlist, outcomes_path, sync_state_path, sync_log_path, monkeypatch
):
    import swing.sync as s
    monkeypatch.setattr(s, "_WATCHLIST_PATH", watchlist)
    monkeypatch.setattr(s, "_OUTCOMES_PATH", outcomes_path)
    monkeypatch.setattr(s, "_SYNC_STATE_PATH", sync_state_path)
    monkeypatch.setattr(s, "_SYNC_LOG_PATH", sync_log_path)
    monkeypatch.setattr(s, "swing_stats", MagicMock())

    with patch("swing.sync.TradingClient") as MockClient:
        instance = MockClient.return_value
        instance.get_orders.side_effect = Exception("Network error")
        result = s.run()

    assert result.new_outcomes == 0
    assert len(result.errors) == 1
