"""Circuit breaker safety system. Independent of the HMM regime model."""

import json
import os
import time
from pathlib import Path

_LOGS_DIR = Path(__file__).parent.parent / "logs"
_STATE_FILE = _LOGS_DIR / "safety_state.json"

# Default state structure
_DEFAULT_STATE = {
    "daily_loss_halt": False,
    "weekly_loss_halt_until": 0.0,   # unix timestamp, 0 = not halted
    "max_drawdown_halt": False,
    "peak_equity": None,             # float, tracked over time
    "order_timestamps": [],          # list of float unix timestamps
}


def _load_state() -> dict:
    """Load state from JSON file, return _DEFAULT_STATE copy if missing or corrupt."""
    if not _STATE_FILE.exists():
        return dict(_DEFAULT_STATE)
    try:
        data = json.loads(_STATE_FILE.read_text())
        # Fill in any missing keys from defaults
        state = dict(_DEFAULT_STATE)
        state.update(data)
        return state
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT_STATE)


def _save_state(state: dict) -> None:
    """Write state atomically: write to .tmp then os.replace."""
    _LOGS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, _STATE_FILE)


def reset_state() -> None:
    """Reset to _DEFAULT_STATE (for testing)."""
    _save_state(dict(_DEFAULT_STATE))


# ---------------------------------------------------------------------------
# Breaker 1: Daily loss limit (2%)
# ---------------------------------------------------------------------------

def check_daily_loss(equity_now: float, equity_open: float) -> tuple[bool, str]:
    """Triggered if equity dropped > 2% from today's open.

    Args:
        equity_now: Current portfolio equity value.
        equity_open: Portfolio equity at today's market open.

    Returns:
        (triggered, reason) where reason is non-empty only when triggered.

    Example:
        >>> check_daily_loss(9790.0, 10000.0)  # 2.1% drop
        (True, 'Daily loss 2.1% exceeds 2% limit')
        >>> check_daily_loss(9810.0, 10000.0)  # 1.9% drop
        (False, '')
    """
    if equity_open <= 0:
        return False, ""
    pct_drop = (equity_open - equity_now) / equity_open
    triggered = pct_drop > 0.02
    reason = f"Daily loss {pct_drop:.1%} exceeds 2% limit" if triggered else ""
    return triggered, reason


# ---------------------------------------------------------------------------
# Breaker 2: Weekly loss limit (5%)
# ---------------------------------------------------------------------------

def check_weekly_loss(equity_history: list[float]) -> tuple[bool, str]:
    """Triggered if equity dropped > 5% over the last 5 trading days.

    Args:
        equity_history: Ordered list of equity values (oldest first). Typically
            the last 5 daily closes plus today's value.

    Returns:
        (triggered, reason) where reason is non-empty only when triggered.

    Example:
        >>> check_weekly_loss([10000.0, 9900.0, 9800.0, 9700.0, 9490.0])  # >5% drop
        (True, 'Weekly loss 5.1% exceeds 5% limit')
        >>> check_weekly_loss([10000.0, 9900.0, 9800.0, 9700.0, 9510.0])  # <5% drop
        (False, '')
    """
    if len(equity_history) < 2:
        return False, ""
    start = equity_history[0]
    end = equity_history[-1]
    if start <= 0:
        return False, ""
    pct_drop = (start - end) / start
    triggered = pct_drop > 0.05
    reason = f"Weekly loss {pct_drop:.1%} exceeds 5% limit" if triggered else ""
    return triggered, reason


# ---------------------------------------------------------------------------
# Breaker 3: Max drawdown limit (15%)
# ---------------------------------------------------------------------------

def check_max_drawdown(equity_now: float, peak_equity: float) -> tuple[bool, str]:
    """Triggered if equity dropped > 15% from peak. Requires manual reset.

    Args:
        equity_now: Current portfolio equity value.
        peak_equity: All-time (or session) peak equity value.

    Returns:
        (triggered, reason) where reason is non-empty only when triggered.

    Example:
        >>> check_max_drawdown(8490.0, 10000.0)  # 15.1% drawdown
        (True, 'Max drawdown 15.1% exceeds 15% limit — manual reset required')
        >>> check_max_drawdown(8510.0, 10000.0)  # 14.9% drawdown
        (False, '')
    """
    if peak_equity <= 0:
        return False, ""
    drawdown = (peak_equity - equity_now) / peak_equity
    triggered = drawdown > 0.15
    reason = (
        f"Max drawdown {drawdown:.1%} exceeds 15% limit — manual reset required"
        if triggered else ""
    )
    return triggered, reason


# ---------------------------------------------------------------------------
# Breaker 4: Position concentration (25%)
# ---------------------------------------------------------------------------

def check_concentration(position_value: float, portfolio_value: float) -> tuple[bool, str]:
    """Triggered if a single position exceeds 25% of portfolio.

    Args:
        position_value: Market value of the single position being checked.
        portfolio_value: Total portfolio market value.

    Returns:
        (triggered, reason) where reason is non-empty only when triggered.

    Example:
        >>> check_concentration(2510.0, 10000.0)  # 25.1%
        (True, 'Position 25.1% of portfolio exceeds 25% limit')
        >>> check_concentration(2490.0, 10000.0)  # 24.9%
        (False, '')
    """
    if portfolio_value <= 0:
        return False, ""
    concentration = position_value / portfolio_value
    triggered = concentration > 0.25
    reason = (
        f"Position {concentration:.1%} of portfolio exceeds 25% limit"
        if triggered else ""
    )
    return triggered, reason


# ---------------------------------------------------------------------------
# Breaker 5: Order rate limit (20 orders / 60 seconds)
# ---------------------------------------------------------------------------

def check_order_rate(recent_order_timestamps: list[float]) -> tuple[bool, str]:
    """Triggered if > 20 orders submitted in the last 60 seconds.

    Args:
        recent_order_timestamps: List of unix timestamps for recent order submissions.
            Timestamps older than 60 seconds are ignored.

    Returns:
        (triggered, reason) where reason is non-empty only when triggered.

    Example:
        >>> import time
        >>> timestamps = [time.time()] * 21
        >>> check_order_rate(timestamps)
        (True, '21 orders in last 60s exceeds limit of 20')
        >>> check_order_rate([time.time()] * 20)
        (False, '')
    """
    now = time.time()
    recent = [t for t in recent_order_timestamps if now - t <= 60.0]
    triggered = len(recent) > 20
    reason = (
        f"{len(recent)} orders in last 60s exceeds limit of 20"
        if triggered else ""
    )
    return triggered, reason


# ---------------------------------------------------------------------------
# Aggregate checker
# ---------------------------------------------------------------------------

def check_all(
    equity_now: float,
    equity_open: float,
    equity_history: list[float],
    peak_equity: float,
    position_value: float,
    portfolio_value: float,
    recent_order_timestamps: list[float],
) -> list[tuple[bool, str]]:
    """Run all 5 breakers. Returns list of (triggered, reason) for each.

    Order: daily_loss, weekly_loss, max_drawdown, concentration, order_rate.

    Args:
        equity_now: Current portfolio equity.
        equity_open: Equity at today's market open.
        equity_history: Ordered list of equity values for weekly check.
        peak_equity: All-time peak equity for drawdown check.
        position_value: Value of the single position being checked.
        portfolio_value: Total portfolio value for concentration check.
        recent_order_timestamps: Unix timestamps of recent order submissions.

    Returns:
        List of 5 (triggered, reason) tuples in breaker order.
    """
    return [
        check_daily_loss(equity_now, equity_open),
        check_weekly_loss(equity_history),
        check_max_drawdown(equity_now, peak_equity),
        check_concentration(position_value, portfolio_value),
        check_order_rate(recent_order_timestamps),
    ]


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def status() -> dict:
    """Returns current breaker state dict for dashboard rendering.

    Returns:
        Dict with keys: daily_loss_halt, weekly_loss_halt_active,
        max_drawdown_halt, peak_equity.
    """
    state = _load_state()
    return {
        "daily_loss_halt": state["daily_loss_halt"],
        "weekly_loss_halt_active": state["weekly_loss_halt_until"] > time.time(),
        "max_drawdown_halt": state["max_drawdown_halt"],
        "peak_equity": state["peak_equity"],
    }
