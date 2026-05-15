"""Tests that broker.submit_order() uses real account equity for safety checks."""
import pytest
from unittest.mock import MagicMock, patch
from core.broker import AlpacaBroker


def _make_broker(equity: float = 100_000.0, portfolio_value: float = 100_000.0):
    """Return an AlpacaBroker with a mocked Alpaca client."""
    with patch("core.broker.TradingClient"):
        broker = AlpacaBroker.__new__(AlpacaBroker)
        broker._live = False

    account = MagicMock()
    account.equity = str(equity)
    account.portfolio_value = str(portfolio_value)
    account.last_equity = str(equity)

    client = MagicMock()
    client.get_account.return_value = account
    client.get_all_positions.return_value = []
    broker._client = client
    return broker


def test_daily_loss_breaker_fires_with_real_equity():
    """When account equity is down 3% from open, daily loss breaker must fire."""
    broker = _make_broker(equity=97_000.0, portfolio_value=97_000.0)
    with patch("core.safety.check_daily_loss", return_value=(True, "Daily loss 3.0% exceeds 2% limit")) as mock_check:
        with pytest.raises(RuntimeError, match="Daily loss"):
            broker.submit_order("AAPL", 1.0, "buy")
        mock_check.assert_called_once()
        called_equity_now = mock_check.call_args[1]["equity_now"] if mock_check.call_args[1] else mock_check.call_args[0][0]
        assert called_equity_now == pytest.approx(97_000.0), \
            f"Expected real equity 97000, got {called_equity_now} (placeholder not removed)"


def test_submit_order_invalid_side_raises():
    """Non-buy/sell side string must raise ValueError before touching Alpaca."""
    broker = _make_broker()
    with pytest.raises(ValueError, match="side"):
        broker.submit_order("AAPL", 10.0, "buyyy")


def test_submit_order_zero_qty_raises():
    """Zero qty must raise ValueError."""
    broker = _make_broker()
    with pytest.raises(ValueError, match="qty"):
        broker.submit_order("AAPL", 0.0, "buy")


def test_submit_order_negative_qty_raises():
    broker = _make_broker()
    with pytest.raises(ValueError, match="qty"):
        broker.submit_order("AAPL", -5.0, "buy")


def test_submit_order_invalid_symbol_raises():
    """Symbol longer than 5 chars must raise ValueError."""
    broker = _make_broker()
    with pytest.raises(ValueError, match="symbol"):
        broker.submit_order("TOOLONGSYM", 1.0, "buy")
