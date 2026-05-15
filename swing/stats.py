"""Outcome aggregation: outcomes.json → pattern_stats.json."""
from __future__ import annotations

import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_OUTCOMES_PATH = _REPO_ROOT / "swing" / "outcomes.json"
_STATS_PATH = _REPO_ROOT / "swing" / "improvement" / "pattern_stats.json"
_RULES_PATH = _REPO_ROOT / "swing" / "improvement" / "rules.md"
_WARN_THRESHOLD = 0.40
_MIN_SAMPLES = 5


def rebuild(
    outcomes_path: Path = _OUTCOMES_PATH,
    stats_path: Path = _STATS_PATH,
) -> None:
    """Read outcomes.json, aggregate by (pattern, regime_at_add), write pattern_stats.json."""
    outcomes_path = Path(outcomes_path)
    stats_path = Path(stats_path)

    outcomes: list[dict] = []
    if outcomes_path.exists():
        try:
            outcomes = json.loads(outcomes_path.read_text())
        except (json.JSONDecodeError, OSError):
            outcomes = []

    existing = load(stats_path)

    groups: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for record in outcomes:
        groups[record.get("pattern", "unknown")][record.get("regime_at_add", "Unknown")].append(record)

    result: dict = {}
    for pattern, regime_map in groups.items():
        result[pattern] = {}
        for regime, records in regime_map.items():
            n = len(records)
            avg_tp_pct = sum(r.get("tp_pct_complete", 0) for r in records) / n
            avg_pnl_pct = sum(r.get("pnl_pct", 0.0) for r in records) / n
            wins = sum(1 for r in records if r.get("outcome") in {"full_win", "partial_win"})
            result[pattern][regime] = {
                "n": n,
                "avg_tp_pct": round(avg_tp_pct, 1),
                "avg_pnl_pct": round(avg_pnl_pct, 2),
                "win_rate": round(wins / n, 3),
            }

    result["_meta"] = {
        "last_rebuilt": datetime.now(timezone.utc).isoformat(),
        "total_outcomes": len(outcomes),
    }

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = stats_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    os.replace(tmp, stats_path)

    _check_thresholds(result, existing)


def load(stats_path: Path = _STATS_PATH) -> dict:
    """Return pattern_stats dict; empty dict if file missing or corrupt."""
    stats_path = Path(stats_path)
    if not stats_path.exists():
        return {}
    try:
        return json.loads(stats_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _check_thresholds(new_stats: dict, old_stats: dict) -> None:
    for pattern, regime_map in new_stats.items():
        if pattern == "_meta":
            continue
        for regime, cell in regime_map.items():
            if cell["n"] < _MIN_SAMPLES or cell["win_rate"] >= _WARN_THRESHOLD:
                continue
            old_win_rate = old_stats.get(pattern, {}).get(regime, {}).get("win_rate", 1.0)
            if old_win_rate >= _WARN_THRESHOLD:
                _append_rule(pattern, regime, cell)


def _append_rule(pattern: str, regime: str, cell: dict) -> None:
    _RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    line = (
        f"\n## {ts} — {pattern} in {regime}\n"
        f"- win_rate: {cell['win_rate']:.1%} (n={cell['n']})\n"
        f"- avg_tp_pct: {cell['avg_tp_pct']}%\n"
        f"- avg_pnl_pct: {cell['avg_pnl_pct']}%\n"
        f"- Status: PENDING REVIEW\n"
    )
    with open(_RULES_PATH, "a", encoding="utf-8") as f:
        f.write(line)
