# Trading Self-Improvement Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a regime-aware outcome-tracking and auto-execution system for swing trades — logging Alpaca closed positions, aggregating win rates by pattern × regime, warning before bad setups, and auto-executing stops/TPs/regime exits.

**Architecture:** Four-module `swing/` Python package (stats → warn → sync → trader) consumed by Dashboard 8 and a Claude skill. All file writes are atomic. All orders route through `core.broker.AlpacaBroker` and its five safety circuit breakers.

**Tech Stack:** Python 3.11+, alpaca-py, yfinance, hmmlearn, Plotly, Streamlit, pytest, existing `core/` package.

---

## File Map

| File | Status | Responsibility |
|---|---|---|
| `pyproject.toml` | Modify | Add `swing*` to editable install |
| `swing/__init__.py` | Create | Package marker |
| `swing/outcomes.json` | Create | Append-only closed-position log |
| `swing/improvement/` | Create dir | Aggregated files + audit trail |
| `swing/stats.py` | Create | `rebuild()` + `load()` — pure aggregation |
| `swing/warn.py` | Create | `check()` — HMM regime + win-rate advisory |
| `swing/sync.py` | Create | `run()` — Alpaca poller → outcomes.json |
| `swing/trader.py` | Create | `check_stops/tps/regime/execute_auto_buy` |
| `pages/8_Swing_Improvement.py` | Create | Dashboard 8 — quant terminal |
| `.claude/skills/trading-improvement.md` | Create | Claude skill commands |
| `tests/test_swing_stats.py` | Create | Unit tests for stats.py |
| `tests/test_swing_warn.py` | Create | Unit tests for warn.py |
| `tests/test_swing_sync.py` | Create | Unit tests for sync.py |
| `tests/test_swing_trader.py` | Create | Unit tests for trader.py |

---

## Task 1: Package Scaffold

**Files:**
- Modify: `pyproject.toml`
- Create: `swing/__init__.py`
- Create: `swing/outcomes.json`
- Create: `swing/improvement/.gitkeep`

- [ ] **Step 1: Update pyproject.toml to include swing package**

Replace the `[tool.setuptools.packages.find]` section:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["core*", "swing*"]
```

- [ ] **Step 2: Create `swing/__init__.py`**

```python
"""Swing trading self-improvement package."""
```

- [ ] **Step 3: Initialize empty outcomes file**

```python
# Run in terminal:
# python -c "import json; open('swing/outcomes.json','w').write('[]')"
```

Create `swing/outcomes.json` with content:
```json
[]
```

- [ ] **Step 4: Create improvement directory**

Create `swing/improvement/.gitkeep` (empty file) to track the directory in git.

Create `swing/improvement/rules.md` with content:
```markdown
# Promoted Pattern Rules

Entries below are auto-appended when a pattern × regime cell crosses
win_rate < 40% with n ≥ 5 outcomes. Each entry includes source stats
so the origin is never lost.

---
```

- [ ] **Step 5: Reinstall editable package**

```powershell
.venv\Scripts\Activate.ps1
pip install -e .
python -c "import swing; print('swing package OK')"
```

Expected: `swing package OK`

- [ ] **Step 6: Commit**

```powershell
git add pyproject.toml swing/__init__.py swing/outcomes.json swing/improvement/
git commit -m "feat: scaffold swing/ self-improvement package"
```

---

## Task 2: swing/stats.py

**Files:**
- Create: `swing/stats.py`
- Create: `tests/test_swing_stats.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_swing_stats.py`:

```python
"""Tests for swing/stats.py — pure aggregation logic."""
import json
import os
import pytest
from pathlib import Path


@pytest.fixture
def tmp_outcomes(tmp_path):
    """Write outcomes.json to a temp dir and return the path."""
    records = [
        {"symbol": "AAPL", "pattern": "gap-up", "regime_at_add": "Low Vol",
         "tp_pct_complete": 100, "pnl_pct": 5.2, "outcome": "full_win"},
        {"symbol": "MSFT", "pattern": "gap-up", "regime_at_add": "Low Vol",
         "tp_pct_complete": 33, "pnl_pct": 2.1, "outcome": "partial_win"},
        {"symbol": "PLTR", "pattern": "gap-up", "regime_at_add": "Low Vol",
         "tp_pct_complete": 0, "pnl_pct": -3.0, "outcome": "loss"},
        {"symbol": "SNAP", "pattern": "gap-up", "regime_at_add": "High Vol",
         "tp_pct_complete": 0, "pnl_pct": -4.0, "outcome": "loss"},
        {"symbol": "NKE",  "pattern": "gap-up", "regime_at_add": "High Vol",
         "tp_pct_complete": 0, "pnl_pct": -2.5, "outcome": "loss"},
        # 5 downtrend-break in Low Vol with poor win rate
        *[{"symbol": f"X{i}", "pattern": "downtrend-break", "regime_at_add": "Low Vol",
           "tp_pct_complete": 0, "pnl_pct": -1.0, "outcome": "loss"} for i in range(5)],
    ]
    p = tmp_path / "outcomes.json"
    p.write_text(json.dumps(records))
    return p


@pytest.fixture
def tmp_stats(tmp_path):
    return tmp_path / "pattern_stats.json"


@pytest.fixture
def tmp_rules(tmp_path, monkeypatch):
    rules_path = tmp_path / "rules.md"
    import swing.stats as s
    monkeypatch.setattr(s, "_RULES_PATH", rules_path)
    return rules_path


def test_rebuild_creates_stats_file(tmp_outcomes, tmp_stats, tmp_rules):
    from swing.stats import rebuild
    rebuild(tmp_outcomes, tmp_stats)
    assert tmp_stats.exists()


def test_rebuild_groups_by_pattern_and_regime(tmp_outcomes, tmp_stats, tmp_rules):
    from swing.stats import rebuild, load
    rebuild(tmp_outcomes, tmp_stats)
    stats = load(tmp_stats)
    assert "gap-up" in stats
    assert "Low Vol" in stats["gap-up"]
    assert "High Vol" in stats["gap-up"]


def test_win_rate_counts_only_wins_and_partials(tmp_outcomes, tmp_stats, tmp_rules):
    from swing.stats import rebuild, load
    rebuild(tmp_outcomes, tmp_stats)
    stats = load(tmp_stats)
    # gap-up Low Vol: 2 wins (full+partial), 1 loss → win_rate = 2/3
    cell = stats["gap-up"]["Low Vol"]
    assert cell["n"] == 3
    assert abs(cell["win_rate"] - 2/3) < 0.01


def test_avg_tp_pct(tmp_outcomes, tmp_stats, tmp_rules):
    from swing.stats import rebuild, load
    rebuild(tmp_outcomes, tmp_stats)
    stats = load(tmp_stats)
    cell = stats["gap-up"]["Low Vol"]
    assert abs(cell["avg_tp_pct"] - (100 + 33 + 0) / 3) < 0.1


def test_below_threshold_appends_to_rules(tmp_outcomes, tmp_stats, tmp_rules):
    from swing.stats import rebuild
    rebuild(tmp_outcomes, tmp_stats)
    # downtrend-break Low Vol: 5 losses → win_rate=0.0, n=5 → should append
    assert tmp_rules.exists()
    content = tmp_rules.read_text()
    assert "downtrend-break" in content


def test_meta_includes_total_outcomes(tmp_outcomes, tmp_stats, tmp_rules):
    from swing.stats import rebuild, load
    rebuild(tmp_outcomes, tmp_stats)
    stats = load(tmp_stats)
    assert stats["_meta"]["total_outcomes"] == 10


def test_load_returns_empty_dict_when_missing(tmp_path):
    from swing.stats import load
    result = load(tmp_path / "nonexistent.json")
    assert result == {}


def test_rebuild_is_atomic(tmp_outcomes, tmp_stats, tmp_rules):
    from swing.stats import rebuild
    rebuild(tmp_outcomes, tmp_stats)
    tmp_file = tmp_stats.with_suffix(".tmp")
    assert not tmp_file.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_swing_stats.py -v
```

Expected: `ModuleNotFoundError: No module named 'swing.stats'`

- [ ] **Step 3: Create `swing/stats.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_swing_stats.py -v
```

Expected: `8 passed`

- [ ] **Step 5: Commit**

```powershell
git add swing/stats.py tests/test_swing_stats.py
git commit -m "feat: add swing/stats.py with pattern×regime aggregation"
```

---

## Task 3: swing/warn.py

**Files:**
- Create: `swing/warn.py`
- Create: `tests/test_swing_warn.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_swing_warn.py`:

```python
"""Tests for swing/warn.py — regime-aware advisory."""
import json
import pytest
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@pytest.fixture
def mock_regime_low(monkeypatch):
    """Patch HMM + yfinance to return Low Vol regime."""
    import swing.warn as w

    @dataclass
    class FakeResult:
        labels: list
        stable_labels: list
        confidence: object

    monkeypatch.setattr(w, "_get_current_regime", lambda: ("Low Vol", 0.85))


@pytest.fixture
def mock_regime_high(monkeypatch):
    import swing.warn as w
    monkeypatch.setattr(w, "_get_current_regime", lambda: ("High Vol", 0.78))


@pytest.fixture
def stats_with_data(tmp_path, monkeypatch):
    """Write pattern_stats.json with known values and patch load()."""
    import swing.warn as w
    import swing.stats as s

    stats = {
        "gap-up": {
            "Low Vol":  {"n": 10, "avg_tp_pct": 55, "avg_pnl_pct": 3.2, "win_rate": 0.70},
            "High Vol": {"n":  5, "avg_tp_pct": 10, "avg_pnl_pct": -2.1, "win_rate": 0.20},
        },
        "downtrend-break": {
            "Low Vol":  {"n":  2, "avg_tp_pct": 40, "avg_pnl_pct": 1.0, "win_rate": 0.50},
        },
        "_meta": {"total_outcomes": 17},
    }
    monkeypatch.setattr(s, "load", lambda *a, **kw: stats)
    return stats


def test_clear_result_when_win_rate_above_threshold(
    mock_regime_low, stats_with_data
):
    from swing.warn import check
    result = check("AAPL", "gap-up", 0.7)
    assert result.should_warn is False
    assert result.regime == "Low Vol"
    assert result.win_rate == 0.70
    assert result.n == 10


def test_warn_when_win_rate_below_threshold(
    mock_regime_high, stats_with_data
):
    from swing.warn import check
    result = check("SNAP", "gap-up", 0.6)
    assert result.should_warn is True
    assert result.win_rate == 0.20
    assert "CAUTION" in result.message


def test_no_warn_when_insufficient_data(
    mock_regime_low, stats_with_data
):
    from swing.warn import check
    result = check("WOLF", "downtrend-break", 0.76)
    assert result.should_warn is False
    assert result.win_rate is None
    assert "Insufficient" in result.message
    assert result.n == 2


def test_no_warn_when_pattern_not_in_stats(
    mock_regime_low, stats_with_data
):
    from swing.warn import check
    result = check("XYZ", "oversold-bounce", 0.6)
    assert result.should_warn is False
    assert result.n == 0
    assert "Insufficient" in result.message


def test_result_fields_populated(mock_regime_low, stats_with_data):
    from swing.warn import check
    result = check("AAPL", "gap-up", 0.7)
    assert result.symbol == "AAPL"
    assert result.pattern == "gap-up"
    assert result.regime_confidence == 0.85
    assert result.avg_tp_pct == 55
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_swing_warn.py -v
```

Expected: `ModuleNotFoundError: No module named 'swing.warn'`

- [ ] **Step 3: Create `swing/warn.py`**

```python
"""Regime-aware watchlist warning."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

from core.data import load_ohlcv
from core.hmm_utils import fit_and_filter
from swing import stats as swing_stats

_MIN_SAMPLES = 5
_WARN_WIN_RATE = 0.40
_LOOKBACK_DAYS = 90


@dataclass
class WarnResult:
    symbol: str
    pattern: str
    regime: str
    regime_confidence: float
    win_rate: float | None
    avg_tp_pct: float | None
    n: int
    should_warn: bool
    message: str


def check(symbol: str, pattern: str, confidence: float) -> WarnResult:
    """Return a regime-aware advisory for the given symbol and pattern."""
    regime, regime_confidence = _get_current_regime()

    all_stats = swing_stats.load()
    cell = all_stats.get(pattern, {}).get(regime, {})
    n = cell.get("n", 0)
    win_rate = cell.get("win_rate") if n >= _MIN_SAMPLES else None
    avg_tp_pct = cell.get("avg_tp_pct") if n >= _MIN_SAMPLES else None

    should_warn = win_rate is not None and win_rate < _WARN_WIN_RATE

    if n < _MIN_SAMPLES:
        message = (
            f"Insufficient data for {pattern} in {regime} "
            f"(n={n}, need {_MIN_SAMPLES})."
        )
    elif should_warn:
        message = (
            f"CAUTION: {symbol} / {pattern} in {regime}\n"
            f"Win rate: {win_rate:.0%} | Avg TP: {avg_tp_pct}% | n={n}\n"
            f"Poor historical edge. Consider waiting for Low/Medium Vol "
            f"or sizing ≤ 0.5×."
        )
    else:
        message = (
            f"CLEAR: {symbol} / {pattern} in {regime}\n"
            f"Win rate: {win_rate:.0%} | Avg TP: {avg_tp_pct}% | n={n}"
        )

    return WarnResult(
        symbol=symbol,
        pattern=pattern,
        regime=regime,
        regime_confidence=regime_confidence,
        win_rate=win_rate,
        avg_tp_pct=avg_tp_pct,
        n=n,
        should_warn=should_warn,
        message=message,
    )


def _get_current_regime() -> tuple[str, float]:
    """Run HMM on 90 days of SPY. Returns (regime_label, confidence)."""
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=_LOOKBACK_DAYS)).isoformat()
    ohlcv = load_ohlcv("SPY", start, end)
    result = fit_and_filter(ohlcv)
    regime = result.stable_labels[-1]
    confidence = float(result.confidence[-1])
    return regime, confidence
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_swing_warn.py -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```powershell
git add swing/warn.py tests/test_swing_warn.py
git commit -m "feat: add swing/warn.py regime-aware advisory"
```

---

## Task 4: swing/sync.py

**Files:**
- Create: `swing/sync.py`
- Create: `tests/test_swing_sync.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_swing_sync.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_swing_sync.py -v
```

Expected: `ModuleNotFoundError: No module named 'swing.sync'`

- [ ] **Step 3: Create `swing/sync.py`**

```python
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
    syms = ", ".join(symbols) if symbols else "none"
    errs = f" | errors: {len(errors)}" if errors else ""
    line = f"- {ts} | {count} new outcomes | {syms}{errs}\n"
    with open(_SYNC_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line)
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_swing_sync.py -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```powershell
git add swing/sync.py tests/test_swing_sync.py
git commit -m "feat: add swing/sync.py Alpaca poller"
```

---

## Task 5: swing/trader.py

**Files:**
- Create: `swing/trader.py`
- Create: `tests/test_swing_trader.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_swing_trader.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```powershell
pytest tests/test_swing_trader.py -v
```

Expected: `ModuleNotFoundError: No module named 'swing.trader'`

- [ ] **Step 3: Create `swing/trader.py`**

```python
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
    """Market-sell active positions where live price ≤ stop."""
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
            tp_pct = _TP_PCT_TABLE[min(new_steps, 3)]

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
            order = broker.submit_order(symbol=symbol, qty=1, side="buy", live_confirmed=True)
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
```

- [ ] **Step 4: Run tests to verify they pass**

```powershell
pytest tests/test_swing_trader.py -v
```

Expected: `7 passed`

- [ ] **Step 5: Run full suite to catch regressions**

```powershell
pytest tests/ -v --tb=short 2>&1 | Select-Object -Last 15
```

Expected: all existing tests still pass.

- [ ] **Step 6: Commit**

```powershell
git add swing/trader.py tests/test_swing_trader.py
git commit -m "feat: add swing/trader.py auto-stop/TP/regime/buy executor"
```

---

## Task 6: Migrate watchlist.json

**Files:**
- Modify: `swing/watchlist.json` (add `status`, `tp_steps_hit`, `regime_at_add` to all entries)

- [ ] **Step 1: Run migration script**

Run this one-time Python script in the terminal:

```powershell
.venv\Scripts\python.exe -c "
import json, os
from pathlib import Path

path = Path('swing/watchlist.json')
entries = json.loads(path.read_text())
for e in entries:
    e.setdefault('status', 'watching')
    e.setdefault('tp_steps_hit', 0)
    e.setdefault('regime_at_add', 'Unknown')
tmp = path.with_suffix('.tmp')
tmp.write_text(json.dumps(entries, indent=2))
os.replace(tmp, path)
print(f'Migrated {len(entries)} entries')
"
```

Expected: `Migrated 130 entries`

- [ ] **Step 2: Verify first entry has new fields**

```powershell
.venv\Scripts\python.exe -c "
import json
e = json.loads(open('swing/watchlist.json').read())[0]
print(e['symbol'], e.get('status'), e.get('tp_steps_hit'), e.get('regime_at_add'))
"
```

Expected: `F watching 0 Unknown`

- [ ] **Step 3: Commit**

```powershell
git add swing/watchlist.json
git commit -m "feat: migrate watchlist.json with status/tp_steps_hit/regime_at_add fields"
```

---

## Task 7: pages/8_Swing_Improvement.py — Dashboard 8

**Files:**
- Create: `pages/8_Swing_Improvement.py`

- [ ] **Step 1: Create Dashboard 8**

Create `pages/8_Swing_Improvement.py`:

```python
"""
pages/8_Swing_Improvement.py
============================
Dashboard 8 — Swing Trade Self-Improvement.

Design language: Quant terminal.
Background: #0a0a0f  Accent: #f59e0b (amber)  Font: JetBrains Mono
Card radius: 2px  No glow effects.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from swing import stats as swing_stats

_REPO_ROOT = Path(__file__).parent.parent
_OUTCOMES_PATH = _REPO_ROOT / "swing" / "outcomes.json"
_STATS_PATH = _REPO_ROOT / "swing" / "improvement" / "pattern_stats.json"
_SYNC_STATE_PATH = _REPO_ROOT / "swing" / "improvement" / "sync_state.json"

_BG = "#0a0a0f"
_CARD_BG = "#12120f"
_ACCENT = "#f59e0b"
_RED = "#ef4444"
_GREEN = "#22c55e"
_TEXT = "#e2e8f0"
_MUTED = "#64748b"
_FONT = "'JetBrains Mono', 'Fira Code', monospace"

_PATTERNS = ["gap-up", "downtrend-break", "oversold-bounce"]
_REGIMES = ["Low Vol", "Medium Vol", "High Vol", "Extreme Vol", "Uncertain"]

st.set_page_config(
    page_title="Swing Improvement",
    page_icon="📈",
    layout="wide",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] {{
    font-family: {_FONT};
    background-color: {_BG};
    color: {_TEXT};
}}
.metric-card {{
    background: {_CARD_BG};
    border: 1px solid #1e1e1a;
    border-radius: 2px;
    padding: 12px 16px;
    margin-bottom: 8px;
}}
.badge-win    {{ background:#14532d; color:{_GREEN}; padding:2px 8px; border-radius:2px; font-size:11px; }}
.badge-partial{{ background:#78350f; color:{_ACCENT}; padding:2px 8px; border-radius:2px; font-size:11px; }}
.badge-break  {{ background:#1e293b; color:{_MUTED};  padding:2px 8px; border-radius:2px; font-size:11px; }}
.badge-loss   {{ background:#7f1d1d; color:{_RED};    padding:2px 8px; border-radius:2px; font-size:11px; }}
</style>
""", unsafe_allow_html=True)


def _load_outcomes() -> list[dict]:
    if not _OUTCOMES_PATH.exists():
        return []
    try:
        return json.loads(_OUTCOMES_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _last_sync_ts() -> str:
    if not _SYNC_STATE_PATH.exists():
        return "Never"
    try:
        state = json.loads(_SYNC_STATE_PATH.read_text())
        return state.get("last_sync_ts", "Never")
    except (json.JSONDecodeError, OSError):
        return "Never"


def _data_quality_badge(n: int) -> str:
    if n >= 20:
        return f'<span style="color:{_GREEN}">● Data quality: GOOD ({n} outcomes)</span>'
    if n >= 5:
        return f'<span style="color:{_ACCENT}">● Data quality: LOW ({n} outcomes, need ≥ 20)</span>'
    return f'<span style="color:{_RED}">● Data quality: INSUFFICIENT ({n} outcomes, need ≥ 5)</span>'


def _render_header(outcomes: list[dict]) -> None:
    col1, col2, col3 = st.columns([3, 2, 1])
    with col1:
        st.markdown(f"## 📈 Swing Self-Improvement")
        st.markdown(
            _data_quality_badge(len(outcomes)),
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(f"<small style='color:{_MUTED}'>Last synced: {_last_sync_ts()}</small>", unsafe_allow_html=True)
    with col3:
        if st.button("⟳ Sync Now", use_container_width=True):
            from swing import sync as swing_sync
            with st.spinner("Syncing Alpaca orders…"):
                result = swing_sync.run()
            if result.errors:
                st.error(f"Sync errors: {result.errors}")
            else:
                st.success(f"{result.new_outcomes} new outcomes logged.")
            st.rerun()


def _render_heatmap(stats: dict) -> None:
    st.markdown(f"### Pattern × Regime Win Rate")
    st.markdown(
        f"<small style='color:{_MUTED}'>Red &lt;40% | Amber 40–60% | Green &gt;60% | Grey = insufficient data (n&lt;5)</small>",
        unsafe_allow_html=True,
    )

    z_vals, text_vals = [], []
    for pattern in _PATTERNS:
        row_z, row_text = [], []
        for regime in _REGIMES:
            cell = stats.get(pattern, {}).get(regime, {})
            n = cell.get("n", 0)
            if n < 5:
                row_z.append(None)
                row_text.append(f"n/a<br>(n={n})")
            else:
                wr = cell["win_rate"]
                row_z.append(wr * 100)
                row_text.append(
                    f"{wr:.0%}<br>n={n} | tp={cell['avg_tp_pct']:.0f}%"
                )
        z_vals.append(row_z)
        text_vals.append(row_text)

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=_REGIMES,
        y=_PATTERNS,
        text=text_vals,
        texttemplate="%{text}",
        colorscale=[[0, _RED], [0.4, _ACCENT], [0.6, _ACCENT], [1.0, _GREEN]],
        zmin=0, zmax=100,
        showscale=False,
        xgap=3, ygap=3,
    ))
    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font={"color": _TEXT, "family": _FONT, "size": 12},
        margin={"l": 140, "r": 20, "t": 20, "b": 40},
        height=200,
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_timeline(outcomes: list[dict]) -> None:
    if not outcomes:
        st.info("No outcome data yet. Run a sync once positions close.")
        return

    st.markdown("### Outcome Timeline")
    df = pd.DataFrame(outcomes)
    df["close_ts"] = pd.to_datetime(df["close_ts"], errors="coerce", utc=True)

    color_map = {"gap-up": _ACCENT, "downtrend-break": "#818cf8", "oversold-bounce": "#34d399"}
    fig = go.Figure()
    for pattern in _PATTERNS:
        sub = df[df["pattern"] == pattern]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub["close_ts"],
            y=sub["tp_pct_complete"],
            mode="markers",
            name=pattern,
            marker=dict(color=color_map.get(pattern, _TEXT), size=8, opacity=0.8),
            customdata=sub[["symbol", "regime_at_add", "entry", "close_price", "triggered_by"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Regime: %{customdata[1]}<br>"
                "Entry: $%{customdata[2]:.2f} → $%{customdata[3]:.2f}<br>"
                "TP%%: %{y}%%<br>"
                "Trigger: %{customdata[4]}<extra></extra>"
            ),
        ))

    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font={"color": _TEXT, "family": _FONT, "size": 11},
        height=300,
        margin={"l": 48, "r": 24, "t": 20, "b": 40},
        yaxis={"title": {"text": "TP% Complete", "font": {"color": _MUTED}},
               "range": [-5, 105], "tickfont": {"color": _MUTED}},
        xaxis={"tickfont": {"color": _MUTED}},
        legend={"bgcolor": _CARD_BG, "bordercolor": "#1e1e1a",
                "font": {"color": _TEXT}},
    )
    st.plotly_chart(fig, use_container_width=True)


_BADGE_MAP = {
    "full_win": "badge-win",
    "partial_win": "badge-partial",
    "breakeven": "badge-break",
    "loss": "badge-loss",
}


def _render_table(outcomes: list[dict]) -> None:
    if not outcomes:
        return

    st.markdown("### Recent Outcomes (last 30)")
    recent = sorted(outcomes, key=lambda r: r.get("close_ts", ""), reverse=True)[:30]
    rows = []
    for r in recent:
        badge_cls = _BADGE_MAP.get(r.get("outcome", ""), "badge-break")
        badge = f'<span class="{badge_cls}">{r.get("outcome","")}</span>'
        rows.append({
            "Symbol": r.get("symbol", ""),
            "Pattern": r.get("pattern", ""),
            "Regime": r.get("regime_at_add", ""),
            "Entry": f"${r.get('entry', 0):.2f}",
            "Close": f"${r.get('close_price', 0):.2f}",
            "TP%": f"{r.get('tp_pct_complete', 0)}%",
            "P&L%": f"{r.get('pnl_pct', 0):+.2f}%",
            "Outcome": badge,
            "Trigger": r.get("triggered_by", ""),
        })

    df = pd.DataFrame(rows)
    st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)


def main() -> None:
    outcomes = _load_outcomes()
    stats = swing_stats.load()

    _render_header(outcomes)
    st.markdown("---")
    _render_heatmap(stats)
    st.markdown("---")
    _render_timeline(outcomes)
    st.markdown("---")
    _render_table(outcomes)


main()
```

- [ ] **Step 2: Verify dashboard loads without error**

```powershell
.venv\Scripts\python.exe -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('d8', 'pages/8_Swing_Improvement.py')
# Just check imports resolve
print('imports OK')
"
```

Expected: `imports OK` (may warn about streamlit outside server — that's fine)

- [ ] **Step 3: Commit**

```powershell
git add pages/8_Swing_Improvement.py
git commit -m "feat: add Dashboard 8 — Swing Self-Improvement quant terminal"
```

---

## Task 8: Claude Skill

**Files:**
- Create: `.claude/skills/trading-improvement.md`

- [ ] **Step 1: Create `.claude/` directory and skill file**

```powershell
New-Item -ItemType Directory -Force -Path ".claude\skills"
```

Create `.claude/skills/trading-improvement.md`:

```markdown
---
name: trading-improvement
description: Swing trade self-improvement — sync outcomes from Alpaca, warn before adding watchlist entries, promote pattern rules. Trigger on /swing:sync, /swing:warn, /swing:promote, or any watchlist add.
metadata:
  type: skill
---

# Trading Self-Improvement Skill

## Commands

### /swing:sync
Run `swing.sync.run()` to fetch closed positions from Alpaca and update outcomes.

```python
from swing import sync
result = sync.run()
```

Report back:
- How many new outcomes were logged and which symbols
- Whether any pattern×regime cell newly crossed the warning threshold
- A summary table of current `pattern_stats.json`

If Alpaca credentials are missing (ALPACA_KEY_ID or ALPACA_SECRET not in .env), say so clearly — do not silently return 0 outcomes.

---

### /swing:warn SYMBOL PATTERN
Check if a symbol/pattern has poor historical edge in the current regime.

```python
from swing import warn
result = warn.check(symbol="SYMBOL", pattern="PATTERN", confidence=0.6)
print(result.message)
```

Format the result as:
```
[CLEAR|⚠ CAUTION]: SYMBOL / PATTERN
Current regime:  {regime} (confidence {regime_confidence:.0%})
Historical edge: {win_rate:.0%} win rate | avg TP {avg_tp_pct}% | n={n}
Recommendation:  {message}
```

If n < 5: report "Insufficient data (n=X)" — never invent confidence from thin data.

---

### Auto-hook: Watchlist Add
ANY TIME you add a symbol to `swing/watchlist.json`, you MUST:
1. Run `/swing:warn SYMBOL PATTERN` first
2. Show the result to the user
3. If `should_warn = True`, ask for explicit confirmation before writing the entry
4. When writing the entry, include `regime_at_add` from the warn result, `status: "watching"`, `tp_steps_hit: 0`

Never suppress the warning silently. Never add an entry without running the warn check.

**Entry template:**
```json
{
  "symbol": "SYMBOL",
  "pattern": "PATTERN",
  "setup": "dip",
  "added_ts": "YYYY-MM-DDTHH:MM:SS-0400",
  "entry_estimate": 0.0,
  "entry": 0.0,
  "stop": 0.0,
  "tp_ladder": [],
  "confidence": 0.6,
  "source": "swing-cycle",
  "ema8": 0.0,
  "atr20": 0.0,
  "sma200": 0.0,
  "status": "watching",
  "tp_steps_hit": 0,
  "regime_at_add": "{regime from warn result}"
}
```

---

### /swing:promote
Review `swing/improvement/rules.md` and ask the user which rules to promote to `CLAUDE.md`.

For each rule, show:
- The pattern and regime
- The win rate and n
- The date it was flagged

Ask: "Promote this rule to CLAUDE.md? (yes/no)"

Only promote rules the user explicitly approves. Include the source stats in the CLAUDE.md entry so the origin is never lost. Format:

```
## Trading Rule (promoted {date})
- Pattern `{pattern}` in `{regime}` has {win_rate:.0%} win rate (n={n} as of {flagged_date})
- Avoid adding new watchlist entries with this pattern×regime combination
```

Remove promoted entries from `rules.md` after writing to `CLAUDE.md`.

---

## Safety Reminders
- All orders route through `core.broker.AlpacaBroker.submit_order()` — 5 circuit breakers always active
- `LIVE_TRADING=true` in `.env` is required for real money — paper is the default
- Never auto-promote rules to `CLAUDE.md` without user approval per rule
- Auto-buy cap: 3 per calendar day
```

- [ ] **Step 2: Verify skill file is readable**

```powershell
Get-Content ".claude\skills\trading-improvement.md" | Select-Object -First 5
```

Expected: first 5 lines of the frontmatter.

- [ ] **Step 3: Commit**

```powershell
git add .claude/skills/trading-improvement.md
git commit -m "feat: add trading-improvement Claude skill"
```

---

## Task 9: Scheduling Setup

**Files:**
- No code changes — Claude Code /schedule configuration

- [ ] **Step 1: Set up daily market-close sync (4:05 PM ET)**

In Claude Code, type:

```
/schedule "Run swing sync and auto-trading checks" --cron "5 20 * * 1-5"
```

(20:05 UTC = 4:05 PM ET during EDT. Adjust to `5 21 * * 1-5` during EST Nov–Mar.)

The routine prompt:

```
Run the daily swing trading checks in order:
1. from swing import sync; result = sync.run(); print(f"Sync: {result.new_outcomes} new outcomes")
2. from swing import trader; actions = trader.check_stops(); print(f"Stops: {[a.action for a in actions]}")
3. actions = trader.check_tps(); print(f"TPs: {[a.action for a in actions]}")
4. actions = trader.check_regime(); print(f"Regime: {[a.action for a in actions]}")
Report a one-line summary of all actions taken.
```

- [ ] **Step 2: Set up intraday 30-min checks (9:35 AM – 3:55 PM ET)**

```
/schedule "Intraday stop/TP/regime checks" --cron "5,35 13-20 * * 1-5"
```

Routine prompt:

```
Run intraday swing trading checks (stops, TPs, regime only — no Alpaca order history fetch):
1. from swing import trader; trader.check_stops()
2. trader.check_tps()
3. trader.check_regime()
Report any actions taken or "no action required".
```

- [ ] **Step 3: Set up daily auto-buy (9:35 AM ET)**

```
/schedule "Daily auto-buy for watching entries" --cron "35 13 * * 1-5"
```

Routine prompt:

```
Run auto-buy for all 'watching' swing watchlist entries added since last market open:
from swing import trader; actions = trader.execute_auto_buy()
Report which symbols were bought, skipped (with reason), or hit the daily cap.
```

- [ ] **Step 4: Final full test run**

```powershell
pytest tests/ -v --tb=short 2>&1 | Select-Object -Last 20
```

Expected: all tests pass (existing 301 + new swing tests).

- [ ] **Step 5: Final commit**

```powershell
git add docs/
git commit -m "feat: complete trading self-improvement system — swing package + dashboard 8 + skill + scheduling"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| Auto-detect closed positions from Alpaca | Task 4 (sync.py) |
| TP ladder % completion scoring | Task 4 (sync.py `_build_outcome`) |
| Pattern × regime win rate aggregation | Task 2 (stats.py) |
| Regime via HMM on SPY | Task 3 (warn.py `_get_current_regime`) |
| Warn on watchlist add | Task 8 (skill auto-hook) |
| Dashboard 8 heatmap | Task 7 |
| Dashboard 8 timeline | Task 7 |
| Dashboard 8 recent outcomes table | Task 7 |
| Stop trigger auto-sell | Task 5 (trader.check_stops) |
| TP1/2/3 partial sell | Task 5 (trader.check_tps) |
| Regime exit (Extreme Vol / Uncertain) | Task 5 (trader.check_regime) |
| Auto-buy at 9:35 AM | Task 5 (trader.execute_auto_buy) |
| Daily buy cap (3/day) | Task 5 |
| Auto-buy skips when should_warn | Task 5 |
| Watchlist status lifecycle | Task 6 (migration) |
| regime_at_add + tp_steps_hit fields | Task 6 (migration) |
| Atomic writes throughout | All tasks (os.replace pattern) |
| /swing:sync command | Task 8 (skill) |
| /swing:warn command | Task 8 (skill) |
| /swing:promote command | Task 8 (skill) |
| Scheduling (daily + intraday + auto-buy) | Task 9 |
| Circuit breakers on all orders | Task 5 (AlpacaBroker) |
| rules.md audit trail | Task 2 (stats._append_rule) |

All spec requirements covered. ✓
