"""Tests for AlpacaBroker.get_portfolio_history()."""
import pytest
from unittest.mock import MagicMock


def _make_broker():
    """Return an AlpacaBroker with a fully mocked Alpaca client."""
    from core.broker import AlpacaBroker
    broker = AlpacaBroker.__new__(AlpacaBroker)
    broker._live = False
    broker._client = MagicMock()
    return broker


def test_get_portfolio_history_returns_timestamps_and_equity():
    broker = _make_broker()
    fake_history = MagicMock()
    fake_history.timestamp = [1700000000, 1700086400, 1700172800]
    fake_history.equity = [10000.0, 10200.0, 10150.0]
    broker._client.get_portfolio_history.return_value = fake_history

    result = broker.get_portfolio_history("1M", "1D")

    assert result["timestamps"] == [1700000000, 1700086400, 1700172800]
    assert result["equity"] == [10000.0, 10200.0, 10150.0]


def test_get_portfolio_history_preserves_none_equity():
    broker = _make_broker()
    fake_history = MagicMock()
    fake_history.timestamp = [1700000000, 1700086400]
    fake_history.equity = [10000.0, None]
    broker._client.get_portfolio_history.return_value = fake_history

    result = broker.get_portfolio_history("1W", "1H")

    assert result["equity"] == [10000.0, None]  # None preserved, caller filters


def test_get_portfolio_history_passes_period_and_timeframe():
    broker = _make_broker()
    fake_history = MagicMock()
    fake_history.timestamp = []
    fake_history.equity = []
    broker._client.get_portfolio_history.return_value = fake_history

    broker.get_portfolio_history("3M", "1D")

    broker._client.get_portfolio_history.assert_called_once_with(period="3M", timeframe="1D")


def test_get_portfolio_history_raises_runtime_error_on_api_failure():
    broker = _make_broker()
    broker._client.get_portfolio_history.side_effect = Exception("network error")

    with pytest.raises(RuntimeError, match="get_portfolio_history failed"):
        broker.get_portfolio_history("1M", "1D")
