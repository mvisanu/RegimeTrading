"""Alpaca poller: closed orders → outcomes.json."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from dotenv import load_dotenv

from swing import stats as swing_stats

load_dotenv()

_REPO_ROOT = Path(__file__).parent.parent
_WATCHLIST_PATH = _REPO_ROOT / "swing" / "watchlist.json"
_OUTCOMES_PATH = _REPO_ROOT / "swing" / "outcomes.json"
_IMPROVEMENT_DIR = _REPO_ROOT / "swing" / "improvement"
_SYNC_STATE_PATH = _IMPROVEMENT_DIR / "sync_state.json"
_SYNC_LOG_PATH = _IMPROVEMENT_DIR / "sync_log.md"
_TP_PCT_TABLE = [0, 33, 66, 100]


@dataclass
class SyncResult:
    new_outcomes: int
    symbols_matched: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def run() -> SyncResult:
    """Fetch closed Alpaca orders since last sync; append to outcomes.json."""
    _IMPROVEMENT_DIR.mkdir(parents=True, exist_ok=True)

    state = _load_sync_state()
    last_ts = state.get("last_sync_ts")
    after_dt = (
        datetime.fromisoformat(last_ts)
        if last_ts
        else datetime.now(timezone.utc) - timedelta(days=30)
    )

    client = _make_client()
    watchlist = _load_watchlist()
    symbol_map = {e["symbol"]: e for e in watchlist}
    existing_outcomes = _load_outcomes()
    existing_ids = {o["id"] for o in existing_outcomes}

    errors: list[str] = []
    new_records: list[dict] = []

    try:
        request = GetOrdersRequest(status=QueryOrderStatus.CLOSED, after=after_dt)
        orders = client.get_orders(filter=request)
    except Exception as exc:
        errors.append(f"Alpaca fetch failed: {exc}")
        return SyncResult(0, [], errors)

    for order in orders:
        symbol = order.symbol
        if symbol not in symbol_map:
            continue

        record_id = f"OUT-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{str(order.id)[:8]}"
        if record_id in existing_ids:
            continue

        entry = symbol_map[symbol]
        try:
            record = _build_outcome(record_id, order, entry)
            new_records.append(record)
        except Exception as exc:
            errors.append(f"Failed for {symbol}: {exc}")

    if new_records:
        _write_outcomes(existing_outcomes + new_records)
        _update_watchlist_closed(watchlist, new_records)
        swing_stats.rebuild()

    now_iso = datetime.now(timezone.utc).isoformat()
    state["last_sync_ts"] = now_iso
    _save_sync_state(state)
    _append_sync_log(now_iso, len(new_records), [r["symbol"] for r in new_records], errors)

    return SyncResult(len(new_records), [r["symbol"] for r in new_records], errors)


def _build_outcome(record_id: str, order, entry: dict) -> dict:
    close_price = float(order.filled_avg_price or 0)
    tp_ladder: list[float] = entry.get("tp_ladder") or []
    entry_price: float = entry.get("entry", close_price)

    tp_steps_hit = sum(1 for tp in tp_ladder if close_price >= tp)
    tp_steps_total = len(tp_ladder) if tp_ladder else 1
    tp_pct_complete = _TP_PCT_TABLE[min(tp_steps_hit, 3)]

    pnl_pct = round((close_price - entry_price) / entry_price * 100, 2) if entry_price else 0.0

    if tp_pct_complete == 100:
        outcome = "full_win"
    elif tp_pct_complete > 0:
        outcome = "partial_win"
    elif close_price > entry_price:
        outcome = "breakeven"
    else:
        outcome = "loss"

    return {
        "id": record_id,
        "symbol": order.symbol,
        "pattern": entry.get("pattern", "unknown"),
        "confidence": entry.get("confidence", 0.0),
        "entry": entry_price,
        "stop": entry.get("stop", 0),
        "tp_ladder": tp_ladder,
        "added_ts": entry.get("added_ts", ""),
        "regime_at_add": entry.get("regime_at_add", "Unknown"),
        "close_ts": str(order.filled_at or order.updated_at),
        "close_price": close_price,
        "tp_steps_hit": tp_steps_hit,
        "tp_steps_total": tp_steps_total,
        "tp_pct_complete": tp_pct_complete,
        "pnl_pct": pnl_pct,
        "outcome": outcome,
        "triggered_by": "manual",
    }


def _make_client() -> TradingClient:
    key = os.environ.get("ALPACA_KEY_ID", "")
    secret = os.environ.get("ALPACA_SECRET", "")
    paper = os.environ.get("LIVE_TRADING", "false").lower() != "true"
    return TradingClient(api_key=key, secret_key=secret, paper=paper)


def _load_watchlist() -> list[dict]:
    if not _WATCHLIST_PATH.exists():
        return []
    return json.loads(_WATCHLIST_PATH.read_text())


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


def _update_watchlist_closed(watchlist: list[dict], new_records: list[dict]) -> None:
    fully_closed = {r["symbol"] for r in new_records if r["tp_pct_complete"] == 100 or r["outcome"] == "loss"}
    changed = False
    for entry in watchlist:
        if entry["symbol"] in fully_closed and entry.get("status") == "active":
            entry["status"] = "closed"
            changed = True
    if changed:
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
    tmp = _SYNC_STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2))
    os.replace(tmp, _SYNC_STATE_PATH)


def _append_sync_log(ts: str, count: int, symbols: list[str], errors: list[str]) -> None:
    _SYNC_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    syms = ", ".join(symbols) if symbols else "none"
    errs = f" | errors: {len(errors)}" if errors else ""
    line = f"- {ts} | {count} new outcomes | {syms}{errs}\n"
    with open(_SYNC_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)
