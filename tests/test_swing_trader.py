"""Tests for swing/trader.py — auto-execution."""
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def watchlist_active(tmp_path):
    entries = [
        {"symbol": "PLTR", "pattern": "gap-up", "confidence": 0.6,
         "entry": 132.18, "stop": 128.76,
         "tp_ladder": [135.0, 140.0, 145.0],
         "regime_at_add": "Low Vol", "status": "active", "tp_steps_hit": 0},
    ]
    p = tmp_path / "watchlist.json"
    p.write_text(json.dumps(entries))
    return p


@pytest.fixture
def watchlist_watching(tmp_path):
    entries = [
        {"symbol": "SNAP", "pattern": "gap-up", "confidence": 0.6,
         "entry": 5.37, "stop": 5.19, "tp_ladder": [],
         "status": "watching", "tp_steps_hit": 0},
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


def _patch_trader(monkeypatch, tmp_path, watchlist_path, outcomes_path, sync_state_path):
    import swing.trader as t
    monkeypatch.setattr(t, "_WATCHLIST_PATH", watchlist_path)
    monkeypatch.setattr(t, "_OUTCOMES_PATH", outcomes_path)
    monkeypatch.setattr(t, "_SYNC_STATE_PATH", sync_state_path)


def test_check_stops_sells_when_price_at_stop(
    tmp_path, watchlist_active, outcomes_path, sync_state_path, monkeypatch
):
    import swing.trader as t
    _patch_trader(monkeypatch, tmp_path, watchlist_active, outcomes_path, sync_state_path)

    mock_broker = MagicMock()
    mock_broker.return_value.submit_order.return_value = {"id": "order1"}
    mock_client = MagicMock()
    mock_client.return_value.get_latest_trade.return_value.price = 128.0
    mock_client.return_value.get_open_position.return_value.qty = "10"

    with patch("swing.trader.TradingClient", mock_client), \
         patch("swing.trader.AlpacaBroker", mock_broker):
        actions = t.check_stops()

    assert len(actions) == 1
    assert actions[0].action == "sell_full"
    assert actions[0].symbol == "PLTR"


def test_check_stops_no_action_when_price_above_stop(
    tmp_path, watchlist_active, outcomes_path, sync_state_path, monkeypatch
):
    import swing.trader as t
    _patch_trader(monkeypatch, tmp_path, watchlist_active, outcomes_path, sync_state_path)

    mock_client = MagicMock()
    mock_client.return_value.get_latest_trade.return_value.price = 135.0
    mock_client.return_value.get_open_position.return_value.qty = "10"

    with patch("swing.trader.TradingClient", mock_client), \
         patch("swing.trader.AlpacaBroker", MagicMock()):
        actions = t.check_stops()

    assert len(actions) == 0


def test_check_tps_sells_partial_at_tp1(
    tmp_path, watchlist_active, outcomes_path, sync_state_path, monkeypatch
):
    import swing.trader as t
    _patch_trader(monkeypatch, tmp_path, watchlist_active, outcomes_path, sync_state_path)

    mock_broker = MagicMock()
    mock_broker.return_value.submit_order.return_value = {"id": "order2"}
    mock_client = MagicMock()
    mock_client.return_value.get_latest_trade.return_value.price = 136.0
    mock_client.return_value.get_open_position.return_value.qty = "9"

    with patch("swing.trader.TradingClient", mock_client), \
         patch("swing.trader.AlpacaBroker", mock_broker):
        actions = t.check_tps()

    assert len(actions) == 1
    assert actions[0].action == "sell_partial"
    wl = json.loads(watchlist_active.read_text())
    assert wl[0]["tp_steps_hit"] == 1


def test_check_regime_exits_all_on_extreme_vol(
    tmp_path, watchlist_active, outcomes_path, sync_state_path, monkeypatch
):
    import swing.trader as t
    _patch_trader(monkeypatch, tmp_path, watchlist_active, outcomes_path, sync_state_path)

    mock_broker = MagicMock()
    mock_broker.return_value.submit_order.return_value = {"id": "order3"}
    mock_client = MagicMock()
    mock_client.return_value.get_latest_trade.return_value.price = 130.0
    mock_client.return_value.get_open_position.return_value.qty = "5"

    monkeypatch.setattr(t, "_get_current_regime", lambda: ("Extreme Vol", 0.90))

    with patch("swing.trader.TradingClient", mock_client), \
         patch("swing.trader.AlpacaBroker", mock_broker):
        actions = t.check_regime()

    assert len(actions) == 1
    assert actions[0].action == "sell_full"
    wl = json.loads(watchlist_active.read_text())
    assert wl[0]["status"] == "closed"


def test_check_regime_no_action_in_low_vol(
    tmp_path, watchlist_active, outcomes_path, sync_state_path, monkeypatch
):
    import swing.trader as t
    _patch_trader(monkeypatch, tmp_path, watchlist_active, outcomes_path, sync_state_path)
    monkeypatch.setattr(t, "_get_current_regime", lambda: ("Low Vol", 0.85))

    with patch("swing.trader.TradingClient", MagicMock()), \
         patch("swing.trader.AlpacaBroker", MagicMock()):
        actions = t.check_regime()

    assert actions == []


def test_auto_buy_skips_when_daily_cap_reached(
    tmp_path, watchlist_watching, outcomes_path, sync_state_path, monkeypatch
):
    import swing.trader as t
    from datetime import date
    _patch_trader(monkeypatch, tmp_path, watchlist_watching, outcomes_path, sync_state_path)

    today = date.today().isoformat()
    sync_state_path.write_text(json.dumps({"buy_date": today, "daily_buy_count": 3}))

    actions = t.execute_auto_buy()
    assert len(actions) == 1
    assert actions[0].action == "skip"
    assert "daily_cap" in actions[0].reason


def test_auto_buy_skips_when_warn_fires(
    tmp_path, watchlist_watching, outcomes_path, sync_state_path, monkeypatch
):
    import swing.trader as t
    import swing.warn as w
    _patch_trader(monkeypatch, tmp_path, watchlist_watching, outcomes_path, sync_state_path)

    fake_warn = MagicMock()
    fake_warn.should_warn = True
    fake_warn.message = "CAUTION: poor edge"
    fake_warn.regime = "High Vol"
    monkeypatch.setattr(w, "check", lambda *a, **kw: fake_warn)

    with patch("swing.trader.AlpacaBroker", MagicMock()):
        actions = t.execute_auto_buy()

    assert actions[0].action == "skip"
    assert "warn" in actions[0].reason
