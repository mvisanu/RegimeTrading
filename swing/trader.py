"""Auto-execution: stops, TP ladder, regime exits, auto-buy."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from alpaca.trading.client import TradingClient
from dotenv import load_dotenv

from core.broker import AlpacaBroker
from swing import warn as swing_warn

load_dotenv()

_REPO_ROOT = Path(__file__).parent.parent
_WATCHLIST_PATH = _REPO_ROOT / "swing" / "watchlist.json"
_OUTCOMES_PATH = _REPO_ROOT / "swing" / "outcomes.json"
_SYNC_STATE_PATH = _REPO_ROOT / "swing" / "improvement" / "sync_state.json"
_TP_PCT_TABLE = [0, 33, 66, 100]
_EXIT_REGIMES = {"Extreme Vol", "Uncertain"}
_MAX_DAILY_BUYS = 3


@dataclass
class TradeAction:
    symbol: str
    action: str
    reason: str
    order_id: str | None = None
    error: str | None = None


def check_stops() -> list[TradeAction]:
    """Market-sell active positions where live price <= stop."""
    watchlist = _load_watchlist()
    active = [e for e in watchlist if e.get("status") == "active"]
    if not active:
        return []

    client = _make_client()
    actions: list[TradeAction] = []
    changed = False

    for entry in active:
        symbol = entry["symbol"]
        stop = entry.get("stop", 0)
        try:
            live_price = float(client.get_latest_trade(symbol).price)
        except Exception as exc:
            actions.append(TradeAction(symbol, "skip", "price_fetch_failed", error=str(exc)))
            continue

        if live_price > 0 and live_price <= stop:
            try:
                qty = _get_position_qty(client, symbol)
                broker = AlpacaBroker()
                order = broker.submit_order(symbol=symbol, qty=qty, side="sell", live_confirmed=True)
                _append_outcome(entry, live_price, "stop_auto", "loss", 0)
                entry["status"] = "closed"
                changed = True
                actions.append(TradeAction(symbol, "sell_full", f"stop_hit@{live_price}", order_id=str(order.get("id", ""))))
            except Exception as exc:
                actions.append(TradeAction(symbol, "skip", "order_failed", error=str(exc)))

    if changed:
        _write_watchlist(watchlist)
    return actions


def check_tps() -> list[TradeAction]:
    """Sell 33% of position at each TP ladder level hit."""
    watchlist = _load_watchlist()
    active = [e for e in watchlist if e.get("status") == "active"]
    if not active:
        return []

    client = _make_client()
    actions: list[TradeAction] = []
    changed = False

    for entry in active:
        symbol = entry["symbol"]
        tp_ladder: list[float] = entry.get("tp_ladder") or []
        if not tp_ladder:
            continue

        steps_hit: int = entry.get("tp_steps_hit", 0)
        if steps_hit >= len(tp_ladder):
            continue

        try:
            live_price = float(client.get_latest_trade(symbol).price)
        except Exception as exc:
            actions.append(TradeAction(symbol, "skip", "price_fetch_failed", error=str(exc)))
            continue

        next_tp = tp_ladder[steps_hit]
        if live_price < next_tp:
            continue

        total_qty = _get_position_qty(client, symbol)
        is_last_tp = steps_hit == len(tp_ladder) - 1
        sell_qty = total_qty if is_last_tp else max(1.0, round(total_qty / 3))
        triggered_by = f"tp{steps_hit + 1}_auto"

        try:
            broker = AlpacaBroker()
            order = broker.submit_order(symbol=symbol, qty=sell_qty, side="sell", live_confirmed=True)
            entry["tp_steps_hit"] = steps_hit + 1
            new_steps = entry["tp_steps_hit"]

            if is_last_tp:
                _append_outcome(entry, live_price, triggered_by, "full_win", new_steps)
                entry["status"] = "closed"
            else:
                _append_outcome(entry, live_price, triggered_by, "partial_win", new_steps)

            changed = True
            actions.append(TradeAction(symbol, "sell_partial", f"tp{new_steps}@{live_price}", order_id=str(order.get("id", ""))))
        except Exception as exc:
            actions.append(TradeAction(symbol, "skip", "order_failed", error=str(exc)))

    if changed:
        _write_watchlist(watchlist)
    return actions


def check_regime() -> list[TradeAction]:
    """Exit all active positions when HMM detects Extreme Vol or Uncertain."""
    regime, _ = _get_current_regime()
    if regime not in _EXIT_REGIMES:
        return []

    watchlist = _load_watchlist()
    active = [e for e in watchlist if e.get("status") == "active"]
    if not active:
        return []

    client = _make_client()
    actions: list[TradeAction] = []
    changed = False

    for entry in active:
        symbol = entry["symbol"]
        try:
            live_price = float(client.get_latest_trade(symbol).price)
            qty = _get_position_qty(client, symbol)
            broker = AlpacaBroker()
            order = broker.submit_order(symbol=symbol, qty=qty, side="sell", live_confirmed=True)
            steps_hit = entry.get("tp_steps_hit", 0)
            _append_outcome(entry, live_price, "regime_exit", "loss", steps_hit)
            entry["status"] = "closed"
            changed = True
            actions.append(TradeAction(symbol, "sell_full", f"regime_exit:{regime}", order_id=str(order.get("id", ""))))
        except Exception as exc:
            actions.append(TradeAction(symbol, "skip", "order_failed", error=str(exc)))

    if changed:
        _write_watchlist(watchlist)
    return actions


def execute_auto_buy() -> list[TradeAction]:
    """Buy all 'watching' entries whose pattern clears the win-rate threshold."""
    state = _load_sync_state()
    today = date.today().isoformat()
    daily_count = state.get("daily_buy_count", 0) if state.get("buy_date") == today else 0

    if daily_count >= _MAX_DAILY_BUYS:
        return [TradeAction("*", "skip", f"daily_cap_reached ({_MAX_DAILY_BUYS})")]

    watchlist = _load_watchlist()
    watching = [e for e in watchlist if e.get("status", "watching") == "watching"]
    if not watching:
        return []

    actions: list[TradeAction] = []
    changed = False

    for entry in watching:
        if daily_count >= _MAX_DAILY_BUYS:
            break

        symbol = entry["symbol"]
        pattern = entry.get("pattern", "unknown")
        confidence = entry.get("confidence", 0.5)

        warn_result = swing_warn.check(symbol, pattern, confidence)
        entry["regime_at_add"] = warn_result.regime

        if warn_result.should_warn:
            actions.append(TradeAction(symbol, "skip", f"warn:{warn_result.message}"))
            continue

        try:
            broker = AlpacaBroker()
            order = broker.submit_order(
                symbol=symbol,
                qty=1,
                side="buy",
                live_confirmed=True,
                price=entry.get("entry") or None,
                tp_ladder=entry.get("tp_ladder") or None,
                stop=entry.get("stop") or None,
            )
            entry["status"] = "active"
            daily_count += 1
            changed = True
            actions.append(TradeAction(symbol, "buy", f"auto_buy in {warn_result.regime}", order_id=str(order.get("id", ""))))
        except Exception as exc:
            actions.append(TradeAction(symbol, "skip", "order_failed", error=str(exc)))

    if changed:
        _write_watchlist(watchlist)

    state.update({"buy_date": today, "daily_buy_count": daily_count})
    _save_sync_state(state)
    return actions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_current_regime() -> tuple[str, float]:
    from datetime import timedelta
    from core.data import load_ohlcv
    from core.hmm_utils import fit_and_filter
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=90)).isoformat()
    ohlcv = load_ohlcv("SPY", start, end)
    result = fit_and_filter(ohlcv)
    return result.stable_labels[-1], float(result.confidence[-1])


def _make_client() -> TradingClient:
    key = os.environ.get("ALPACA_KEY_ID", "")
    secret = os.environ.get("ALPACA_SECRET", "")
    paper = os.environ.get("LIVE_TRADING", "false").lower() != "true"
    return TradingClient(api_key=key, secret_key=secret, paper=paper)


def _get_position_qty(client: TradingClient, symbol: str) -> float:
    try:
        return float(client.get_open_position(symbol).qty)
    except Exception:
        return 1.0


def _load_watchlist() -> list[dict]:
    if not _WATCHLIST_PATH.exists():
        return []
    return json.loads(_WATCHLIST_PATH.read_text())


def _write_watchlist(watchlist: list[dict]) -> None:
    tmp = _WATCHLIST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(watchlist, indent=2))
    os.replace(tmp, _WATCHLIST_PATH)


def _load_sync_state() -> dict:
    if not _SYNC_STATE_PATH.exists():
        return {}
    try:
        return json.loads(_SYNC_STATE_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_sync_state(state: dict) -> None:
    _SYNC_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _SYNC_STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, _SYNC_STATE_PATH)


def _load_outcomes() -> list[dict]:
    if not _OUTCOMES_PATH.exists():
        return []
    try:
        return json.loads(_OUTCOMES_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _write_outcomes(outcomes: list[dict]) -> None:
    tmp = _OUTCOMES_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(outcomes, indent=2))
    os.replace(tmp, _OUTCOMES_PATH)


def _append_outcome(entry: dict, close_price: float, triggered_by: str, outcome: str, tp_steps_hit: int) -> None:
    records = _load_outcomes()
    tp_ladder = entry.get("tp_ladder") or []
    entry_price = entry.get("entry", close_price)
    tp_steps_total = len(tp_ladder) if tp_ladder else 1
    tp_pct_complete = _TP_PCT_TABLE[min(tp_steps_hit, 3)]
    pnl_pct = round((close_price - entry_price) / entry_price * 100, 2) if entry_price else 0.0

    records.append({
        "id": f"OUT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{entry['symbol']}",
        "symbol": entry["symbol"],
        "pattern": entry.get("pattern", "unknown"),
        "confidence": entry.get("confidence", 0.0),
        "entry": entry_price,
        "stop": entry.get("stop", 0),
        "tp_ladder": tp_ladder,
        "added_ts": entry.get("added_ts", ""),
        "regime_at_add": entry.get("regime_at_add", "Unknown"),
        "close_ts": datetime.now(timezone.utc).isoformat(),
        "close_price": close_price,
        "tp_steps_hit": tp_steps_hit,
        "tp_steps_total": tp_steps_total,
        "tp_pct_complete": tp_pct_complete,
        "pnl_pct": pnl_pct,
        "outcome": outcome,
        "triggered_by": triggered_by,
    })
    _write_outcomes(records)
