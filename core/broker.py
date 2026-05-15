"""Alpaca API wrapper. Defaults to paper trading. Live requires LIVE_TRADING=true + dashboard confirmation."""

import json
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from core import notify, safety

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

_TRADES_FILE = Path(__file__).parent.parent / "logs" / "trades.json"


# ---------------------------------------------------------------------------
# Logging helpers (module-level, used by AlpacaBroker methods)
# ---------------------------------------------------------------------------

def _build_log_record(symbol: str, qty: float, side: str, status: str, detail: str = "") -> dict:
    """Build a standard trade log record.

    Args:
        symbol: Ticker symbol (e.g. "AAPL").
        qty: Quantity of shares/units.
        side: "buy" or "sell".
        status: One of ACCEPTED, REJECTED_SAFETY, REJECTED_LIVE_UNCONFIRMED, ERROR.
        detail: Optional extra context (error message, triggered breaker info, etc.).

    Returns:
        Dict ready to be appended to the trade log.

    Example:
        >>> r = _build_log_record("AAPL", 10.0, "buy", "ACCEPTED")
        >>> r["symbol"]
        'AAPL'
        >>> r["status"]
        'ACCEPTED'
    """
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "qty": qty,
        "side": side,
        "status": status,
        "detail": detail,
    }


def _append_log(path: Path, record: dict) -> None:
    """Append a record to the JSON trade log atomically.

    Reads the existing list, appends the new record, writes to a .tmp file,
    then uses os.replace for an atomic swap. Never loses prior records even
    if the log is currently corrupt — falls back to an empty list.

    Args:
        path: Absolute path to the trades.json log file.
        record: Dict to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError):
            existing = []
    existing.append(record)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(existing, indent=2, default=str))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# AlpacaBroker
# ---------------------------------------------------------------------------

class AlpacaBroker:
    """Thin wrapper around alpaca-py TradingClient with safety gate and audit logging.

    Paper trading is the default. Live trading requires both ``LIVE_TRADING=true``
    in the environment **and** ``live_confirmed=True`` passed to every
    ``submit_order`` call, preventing accidental live execution.

    All order attempts — accepted or rejected — are appended to
    ``logs/trades.json`` atomically so the audit trail is always complete.
    """

    def __init__(self) -> None:
        """Load credentials from .env and initialise the TradingClient.

        Raises:
            EnvironmentError: If ALPACA_KEY_ID or ALPACA_SECRET are not set.
        """
        load_dotenv()

        key = os.environ.get("ALPACA_KEY_ID")
        secret = os.environ.get("ALPACA_SECRET")
        if not key or not secret:
            raise EnvironmentError("ALPACA_KEY_ID and ALPACA_SECRET must be set in .env")

        live = os.environ.get("LIVE_TRADING", "false").lower() == "true"
        self._live = live
        # paper=True → paper trading endpoint; paper=False → live endpoint
        self._client = TradingClient(api_key=key, secret_key=secret, paper=not live)

    # ------------------------------------------------------------------
    # Read-only account methods
    # ------------------------------------------------------------------

    def get_account(self) -> dict:
        """Return Alpaca account info as a plain dict.

        Works whether the market is open or closed.

        Returns:
            Dict of account fields (buying_power, equity, cash, etc.).

        Raises:
            RuntimeError: On API errors.
        """
        try:
            account = self._client.get_account()
            return account.__dict__
        except Exception as exc:
            raise RuntimeError(f"get_account failed: {exc}") from exc

    def get_positions(self) -> list[dict]:
        """Return all open positions as a list of plain dicts.

        Works whether the market is open or closed.

        Returns:
            List of position dicts (symbol, qty, market_value, etc.).

        Raises:
            RuntimeError: On API errors.
        """
        try:
            positions = self._client.get_all_positions()
            return [p.__dict__ for p in positions]
        except Exception as exc:
            raise RuntimeError(f"get_positions failed: {exc}") from exc

    def get_clock(self) -> dict:
        """Return market clock info.

        Works whether the market is open or closed.

        Returns:
            Dict with keys: is_open, next_open, next_close.

        Raises:
            RuntimeError: On API errors.
        """
        try:
            clock = self._client.get_clock()
            return {
                "is_open": clock.is_open,
                "next_open": clock.next_open,
                "next_close": clock.next_close,
            }
        except Exception as exc:
            raise RuntimeError(f"get_clock failed: {exc}") from exc

    def get_portfolio_history(self, period: str = "1M", timeframe: str = "1D") -> dict:
        """Return portfolio equity history for charting.

        Args:
            period: Alpaca period string — "1D", "1W", "1M", "3M", "1A".
            timeframe: Alpaca timeframe string — "5Min", "1H", "1D".

        Returns:
            Dict with keys ``timestamps`` (list[int] of Unix epoch seconds) and
            ``equity`` (list[float|None]).

        Raises:
            RuntimeError: On API errors.
        """
        try:
            history = self._client.get_portfolio_history(period=period, timeframe=timeframe)
            return {
                "timestamps": [int(ts) for ts in history.timestamp],
                "equity": [float(v) if v is not None else None for v in history.equity],
            }
        except Exception as exc:
            raise RuntimeError(f"get_portfolio_history failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        live_confirmed: bool = False,
        price: float | None = None,
        tp_ladder: list[float] | None = None,
        stop: float | None = None,
    ) -> dict:
        """Submit an order after passing safety checks and live-trading guard.

        Safety breakers are checked before every order using live account data
        fetched from Alpaca. If any breaker fires, the order is rejected and
        logged; no request is sent to Alpaca.

        For live trading (``LIVE_TRADING=true``), ``live_confirmed=True`` must
        be supplied explicitly — this prevents accidental live execution from
        automated code paths that forget the flag.

        All attempts (accepted or rejected) are appended to ``logs/trades.json``
        atomically.

        Args:
            symbol: Ticker symbol, 1–5 alpha characters (e.g. "AAPL").
            qty: Number of shares/units. Must be > 0.
            side: "buy" or "sell" (case-insensitive).
            order_type: Currently only "market" is implemented.
            time_in_force: Alpaca TIF string ("day", "gtc", etc.).
            live_confirmed: Must be True when ``LIVE_TRADING=true`` in env.

        Returns:
            Order dict from Alpaca on success.

        Raises:
            ValueError: If symbol, qty, or side fail validation.
            RuntimeError: If a safety breaker fires, if live_confirmed is
                missing for live trading, or if the Alpaca API call fails.
        """
        # ------------------------------------------------------------------
        # Guard 0: Input validation
        # ------------------------------------------------------------------
        if not symbol or not symbol.isalpha() or len(symbol) > 5:
            raise ValueError(f"Invalid symbol {symbol!r}: must be 1-5 alpha chars")
        if not isinstance(qty, (int, float)) or qty <= 0:
            raise ValueError(f"Invalid qty {qty!r}: must be a positive number")
        if side.lower() not in {"buy", "sell"}:
            raise ValueError(f"Invalid side {side!r}: must be 'buy' or 'sell'")

        # ------------------------------------------------------------------
        # Guard 1: Safety circuit breakers — use real account data
        # ------------------------------------------------------------------
        try:
            account = self._client.get_account()
            equity_now = float(account.equity)
            portfolio_value = float(account.portfolio_value)
            equity_open = float(account.last_equity)
            positions = self._client.get_all_positions()
            pos_value = sum(
                float(p.market_value) for p in positions
                if p.symbol == symbol
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch account data for safety check: {exc}") from exc

        try:
            state = safety._load_state()
            peak_equity = float(state.get("peak_equity") or equity_now)
            recent_timestamps = [float(t) for t in state.get("order_timestamps", [])]
        except Exception:
            peak_equity = equity_now
            recent_timestamps = []

        try:
            history = self._client.get_portfolio_history(period="1W", timeframe="1D")
            equity_history = [float(v) for v in history.equity if v is not None] or [equity_open, equity_now]
        except Exception:
            equity_history = [equity_open, equity_now]

        breakers = safety.check_all(
            equity_now=equity_now,
            equity_open=equity_open,
            equity_history=equity_history,
            peak_equity=peak_equity,
            position_value=pos_value,
            portfolio_value=portfolio_value,
            recent_order_timestamps=recent_timestamps,
        )
        triggered = [(b, r) for b, r in breakers if b]
        if triggered:
            record = _build_log_record(symbol, qty, side, "REJECTED_SAFETY", str(triggered))
            _append_log(_TRADES_FILE, record)
            notify.trade(symbol, qty, side, "REJECTED_SAFETY", str(triggered[0][1]), live=self._live)
            raise RuntimeError(f"Order rejected by safety check: {triggered[0][1]}")

        # ------------------------------------------------------------------
        # Guard 2: Live trading confirmation
        # ------------------------------------------------------------------
        if self._live and not live_confirmed:
            record = _build_log_record(
                symbol, qty, side,
                "REJECTED_LIVE_UNCONFIRMED",
                "live_confirmed=False while LIVE_TRADING=true",
            )
            _append_log(_TRADES_FILE, record)
            raise RuntimeError("Live trading requires live_confirmed=True in submit_order()")

        # ------------------------------------------------------------------
        # Build and submit the order
        # ------------------------------------------------------------------
        try:
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            tif_map = {
                "day": TimeInForce.DAY,
                "gtc": TimeInForce.GTC,
                "opg": TimeInForce.OPG,
                "cls": TimeInForce.CLS,
                "ioc": TimeInForce.IOC,
                "fok": TimeInForce.FOK,
            }
            tif = tif_map.get(time_in_force.lower(), TimeInForce.DAY)

            # Only market orders implemented; extend here for limit/stop
            order_request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=order_side,
                time_in_force=tif,
            )

            order = self._client.submit_order(order_data=order_request)
            order_dict = order.__dict__

            record = _build_log_record(
                symbol, qty, side, "ACCEPTED",
                str(order_dict.get("id", "")),
            )
            _append_log(_TRADES_FILE, record)
            notify.trade(
                symbol, qty, side, "ACCEPTED",
                live=self._live,
                price=price,
                tp_ladder=tp_ladder,
                stop=stop,
            )
            return order_dict

        except Exception as exc:
            record = _build_log_record(symbol, qty, side, "ERROR", str(exc))
            _append_log(_TRADES_FILE, record)
            raise RuntimeError(f"submit_order failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def cancel_all(self) -> None:
        """Cancel all open orders.

        Raises:
            RuntimeError: On API errors.
        """
        try:
            self._client.cancel_orders()
        except Exception as exc:
            raise RuntimeError(f"cancel_all failed: {exc}") from exc
