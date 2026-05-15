# Swing Scanner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the critical safety bypass in `broker.py`, clean up the watchlist, build a regime-aware swing scanner in `swing/scanner.py`, and surface a per-symbol approve/buy UI on the `app.py` landing page.

**Architecture:** Pure-logic `swing/scanner.py` scores watchlist symbols using HMM regime × confidence × R/R × recency, sizes positions via ATR-based 1% risk, and returns ranked `ScanResult` objects. `app.py` calls the scanner and renders a per-row "Buy" button per candidate; each button calls `AlpacaBroker.submit_order()` with real account equity. `broker.py` is fixed to query live account data before every safety check.

**Tech Stack:** Python 3.11, hmmlearn, yfinance, alpaca-py, Streamlit, pandas, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `core/broker.py` | Modify | Fix safety bypass; add input validation |
| `swing/scanner.py` | Create | Pure scoring/sizing logic — no Streamlit |
| `app.py` | Modify | Scanner UI with per-symbol Buy buttons |
| `tests/test_broker_safety.py` | Create | Verify real equity is used in safety checks |
| `tests/test_scanner.py` | Create | Unit tests for scanner pure functions |

---

## Task 1: Fix the Critical Safety Bypass in `broker.py`

**Files:**
- Modify: `core/broker.py:216–224`
- Create: `tests/test_broker_safety.py`

The `submit_order()` method currently passes hardcoded `equity_now=100_000.0` to `safety.check_all()`, making all five circuit breakers permanently inactive. Fix: query real account data from Alpaca before every safety check.

- [ ] **Step 1: Write the failing test**

Create `tests/test_broker_safety.py`:

```python
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
    # Patch equity_open to simulate start-of-day value > current equity
    with patch("core.safety.check_daily_loss", return_value=(True, "Daily loss 3.0% exceeds 2% limit")) as mock_check:
        with pytest.raises(RuntimeError, match="Daily loss"):
            broker.submit_order("AAPL", 1.0, "buy")
        mock_check.assert_called_once()
        # equity_now must NOT be the hardcoded 100_000 placeholder
        called_equity_now = mock_check.call_args[1]["equity_now"] if mock_check.call_args[1] else mock_check.call_args[0][0]
        assert called_equity_now == pytest.approx(97_000.0), \
            f"Expected real equity 97000, got {called_equity_now} (placeholder not removed)"


def test_submit_order_invalid_side_raises():
    """Non-buy/sell side string must raise ValueError before touching Alpaca."""
    broker = _make_broker()
    with pytest.raises(ValueError, match="side"):
        broker.submit_order("AAPL", 10.0, "buyyy")


def test_submit_order_zero_qty_raises():
    """Zero or negative qty must raise ValueError."""
    broker = _make_broker()
    with pytest.raises(ValueError, match="qty"):
        broker.submit_order("AAPL", 0.0, "buy")


def test_submit_order_negative_qty_raises():
    broker = _make_broker()
    with pytest.raises(ValueError, match="qty"):
        broker.submit_order("AAPL", -5.0, "buy")


def test_submit_order_invalid_symbol_raises():
    """Symbol longer than 5 chars or containing digits must raise ValueError."""
    broker = _make_broker()
    with pytest.raises(ValueError, match="symbol"):
        broker.submit_order("TOOLONGSYM", 1.0, "buy")
```

- [ ] **Step 2: Run test to verify it fails**

```
C:\Python311\python.exe -m pytest tests/test_broker_safety.py -v
```

Expected: FAIL — `test_daily_loss_breaker_fires_with_real_equity` fails because placeholder `100_000.0` is still in `submit_order()`.

- [ ] **Step 3: Fix `core/broker.py` — remove placeholder equity, add input validation**

Replace lines 177–229 (the full `submit_order` method signature and Guard 1 block):

```python
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        time_in_force: str = "day",
        live_confirmed: bool = False,
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
            symbol: Ticker symbol, 1–5 uppercase alpha characters (e.g. "AAPL").
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
            equity_open = float(account.last_equity)  # previous close = open proxy
            positions = self._client.get_all_positions()
            pos_value = sum(
                float(p.market_value) for p in positions
                if p.symbol == symbol
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch account data for safety check: {exc}") from exc

        state = safety._load_state()
        peak_equity = float(state.get("peak_equity") or equity_now)
        equity_history = [equity_open, equity_now]
        recent_timestamps = [float(t) for t in state.get("order_timestamps", [])]

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
            raise RuntimeError(f"Order rejected by safety check: {triggered[0][1]}")
```

- [ ] **Step 4: Run tests to verify they pass**

```
C:\Python311\python.exe -m pytest tests/test_broker_safety.py -v
```

Expected: 5 PASSED

- [ ] **Step 5: Run full suite to check no regressions**

```
C:\Python311\python.exe -m pytest tests/ -v --tb=short
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add core/broker.py tests/test_broker_safety.py
git commit -m "fix: replace hardcoded safety placeholders with real account equity in broker.submit_order"
```

---

## Task 2: Build `swing/scanner.py` — Pure Scoring Logic

**Files:**
- Create: `swing/__init__.py` (empty, makes `swing` a package)
- Create: `swing/scanner.py`
- Create: `tests/test_scanner.py`

All functions here are pure — no Streamlit, no broker calls, no side effects.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scanner.py`:

```python
"""Unit tests for swing/scanner.py pure scoring and sizing logic."""
import datetime
import json
import math
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

# We import after the file exists; writing test first drives the interface.
from swing.scanner import (
    load_watchlist,
    score_entry,
    size_position,
    scan,
    ScanResult,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

TODAY_ORD = datetime.date.today().toordinal()

VALID_ENTRY = {
    "symbol": "F",
    "entry": 13.60,
    "stop": 11.80,
    "tp_ladder": [13.97, 14.49, 14.79],
    "confidence": 0.76,
    "atr20": 0.46,
    "added_ordinal": TODAY_ORD,
    "sma200": 12.59,
}


# ---------------------------------------------------------------------------
# load_watchlist
# ---------------------------------------------------------------------------

def test_load_watchlist_returns_list(tmp_path):
    wl = [VALID_ENTRY]
    p = tmp_path / "watchlist.json"
    p.write_text(json.dumps(wl))
    result = load_watchlist(str(p))
    assert isinstance(result, list)
    assert len(result) == 1


def test_load_watchlist_deduplicates_by_symbol(tmp_path):
    """When a symbol appears twice, only the higher-confidence entry is kept."""
    wl = [
        {**VALID_ENTRY, "confidence": 0.60},
        {**VALID_ENTRY, "confidence": 0.76},
    ]
    p = tmp_path / "watchlist.json"
    p.write_text(json.dumps(wl))
    result = load_watchlist(str(p))
    assert len(result) == 1
    assert result[0]["confidence"] == 0.76


def test_load_watchlist_fixes_anomalous_ordinals(tmp_path):
    """Entries with added_ordinal < 2020-01-01 are corrected to today."""
    wl = [{**VALID_ENTRY, "added_ordinal": 3}]  # clearly wrong
    p = tmp_path / "watchlist.json"
    p.write_text(json.dumps(wl))
    result = load_watchlist(str(p))
    assert result[0]["added_ordinal"] == TODAY_ORD


def test_load_watchlist_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_watchlist("/nonexistent/watchlist.json")


# ---------------------------------------------------------------------------
# score_entry
# ---------------------------------------------------------------------------

def test_score_low_vol_beats_medium_vol():
    """Same entry in Low Vol must score higher than Medium Vol."""
    low = score_entry(VALID_ENTRY, "Low Vol", 0.85)
    med = score_entry(VALID_ENTRY, "Medium Vol", 0.85)
    assert low > med


def test_score_high_vol_returns_zero():
    """High Vol is hard-gated — score must be 0.0."""
    assert score_entry(VALID_ENTRY, "High Vol", 0.90) == 0.0


def test_score_extreme_vol_returns_zero():
    assert score_entry(VALID_ENTRY, "Extreme Vol", 0.90) == 0.0


def test_score_empty_tp_ladder_penalised():
    """Entry with no TP ladder gets rr_score=0 and scores below one with TP."""
    no_tp = {**VALID_ENTRY, "tp_ladder": []}
    with_tp = VALID_ENTRY
    assert score_entry(no_tp, "Low Vol", 0.85) < score_entry(with_tp, "Low Vol", 0.85)


def test_rr_score_capped_at_3():
    """R/R > 3.0 is treated as 3.0 — does not inflate score beyond 1.0."""
    huge_tp = {**VALID_ENTRY, "tp_ladder": [100.0]}  # absurd TP → R/R >> 3
    score = score_entry(huge_tp, "Low Vol", 1.0)
    assert 0.0 <= score <= 1.0


def test_score_is_float_in_unit_range():
    s = score_entry(VALID_ENTRY, "Low Vol", 0.76)
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# size_position
# ---------------------------------------------------------------------------

def test_size_position_atr_formula():
    """shares = floor(equity * 0.01 / atr20), minimum 1."""
    shares, cost = size_position(entry=13.60, atr20=0.46, account_equity=10_000.0)
    expected_shares = math.floor(10_000.0 * 0.01 / 0.46)  # = 21
    assert shares == expected_shares
    assert cost == pytest.approx(shares * 13.60)


def test_size_position_minimum_one_share():
    """Even if ATR is very large, we always buy at least 1 share."""
    shares, _ = size_position(entry=5.0, atr20=10_000.0, account_equity=1_000.0)
    assert shares == 1


def test_size_position_zero_atr_returns_one_share():
    shares, _ = size_position(entry=10.0, atr20=0.0, account_equity=10_000.0)
    assert shares == 1


# ---------------------------------------------------------------------------
# scan (integration — mocks HMM)
# ---------------------------------------------------------------------------

def test_scan_returns_top_n(tmp_path):
    """scan() returns at most top_n results."""
    wl = []
    for i, sym in enumerate(["AA", "BB", "CC", "DD", "EE"]):
        wl.append({**VALID_ENTRY, "symbol": sym, "added_ordinal": TODAY_ORD - i})
    p = tmp_path / "watchlist.json"
    p.write_text(json.dumps(wl))

    mock_result = MagicMock()
    mock_result.stable_labels = ["Low Vol"] * 252
    mock_result.confidence = [0.85] * 252

    with patch("swing.scanner.load_ohlcv", return_value=MagicMock()), \
         patch("swing.scanner.fit_and_filter", return_value=mock_result):
        results = scan(str(p), account_equity=10_000.0, top_n=3)

    assert len(results) <= 3
    assert all(isinstance(r, ScanResult) for r in results)


def test_scan_excludes_high_vol(tmp_path):
    """Symbols whose HMM regime is High Vol must not appear in results."""
    wl = [{**VALID_ENTRY, "symbol": "ZZ"}]
    p = tmp_path / "watchlist.json"
    p.write_text(json.dumps(wl))

    mock_result = MagicMock()
    mock_result.stable_labels = ["High Vol"] * 252
    mock_result.confidence = [0.90] * 252

    with patch("swing.scanner.load_ohlcv", return_value=MagicMock()), \
         patch("swing.scanner.fit_and_filter", return_value=mock_result):
        results = scan(str(p), account_equity=10_000.0, top_n=10)

    assert len(results) == 0


def test_scan_skips_symbol_on_data_error(tmp_path):
    """If a symbol's data fetch throws, it is skipped; others still scored."""
    wl = [
        {**VALID_ENTRY, "symbol": "GOOD"},
        {**VALID_ENTRY, "symbol": "BAD"},
    ]
    p = tmp_path / "watchlist.json"
    p.write_text(json.dumps(wl))

    mock_result = MagicMock()
    mock_result.stable_labels = ["Low Vol"] * 252
    mock_result.confidence = [0.85] * 252

    def side_effect(ticker, *a, **kw):
        if ticker == "BAD":
            raise ValueError("No data")
        return MagicMock()

    with patch("swing.scanner.load_ohlcv", side_effect=side_effect), \
         patch("swing.scanner.fit_and_filter", return_value=mock_result):
        results = scan(str(p), account_equity=10_000.0, top_n=10)

    symbols = [r.symbol for r in results]
    assert "GOOD" in symbols
    assert "BAD" not in symbols


def test_scan_results_sorted_descending(tmp_path):
    """Results must be sorted by final_score descending."""
    wl = []
    # Give different recency so scores differ
    for i, sym in enumerate(["AA", "BB", "CC"]):
        wl.append({**VALID_ENTRY, "symbol": sym, "added_ordinal": TODAY_ORD - i * 2})
    p = tmp_path / "watchlist.json"
    p.write_text(json.dumps(wl))

    mock_result = MagicMock()
    mock_result.stable_labels = ["Low Vol"] * 252
    mock_result.confidence = [0.85] * 252

    with patch("swing.scanner.load_ohlcv", return_value=MagicMock()), \
         patch("swing.scanner.fit_and_filter", return_value=mock_result):
        results = scan(str(p), account_equity=10_000.0, top_n=10)

    scores = [r.final_score for r in results]
    assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 2: Run tests to verify they fail**

```
C:\Python311\python.exe -m pytest tests/test_scanner.py -v
```

Expected: ImportError — `swing.scanner` does not exist yet.

- [ ] **Step 3: Create `swing/__init__.py`**

```python
```
(Empty file — makes `swing` a package.)

- [ ] **Step 4: Create `swing/scanner.py`**

```python
"""Regime-aware swing trade scanner.

Pure functions only — no Streamlit, no broker calls, no side effects.
Callers (app.py, CLI scripts) handle I/O and order submission.
"""
from __future__ import annotations

import datetime
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

from core.data import load_ohlcv
from core.hmm_utils import fit_and_filter
from core.allocation import target_exposure

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MIN_VALID_ORDINAL = datetime.date(2020, 1, 1).toordinal()
_RISK_PER_TRADE = 0.01          # 1% of account equity per trade
_MAX_RR = 3.0                   # R/R cap for normalisation
_RECENCY_HALF_LIFE_DAYS = 3.0   # score halves every 3 trading days

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    """Scored and sized watchlist candidate."""
    symbol: str
    regime: str
    regime_confidence: float
    watchlist_confidence: float
    rr_score: float
    recency_score: float
    final_score: float
    shares: int
    estimated_cost: float
    entry: float
    stop: float
    tp1: float | None
    skipped: bool = False
    skip_reason: str = ""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_watchlist(path: str | Path) -> list[dict]:
    """Load watchlist JSON, deduplicate by symbol, fix anomalous ordinals.

    Args:
        path: Path to watchlist.json.

    Returns:
        Deduplicated list of symbol dicts with corrected added_ordinal.

    Raises:
        FileNotFoundError: If path does not exist.
        ValueError: If the file is not valid JSON or not a list.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Watchlist not found: {p}")

    raw = json.loads(p.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Watchlist must be a JSON array, got {type(raw)}")

    today_ord = datetime.date.today().toordinal()

    # Fix anomalous ordinals
    for entry in raw:
        if entry.get("added_ordinal", 0) < _MIN_VALID_ORDINAL:
            entry["added_ordinal"] = today_ord

    # Deduplicate: keep highest-confidence entry per symbol
    by_symbol: dict[str, dict] = {}
    for entry in raw:
        sym = entry.get("symbol", "")
        if not sym:
            continue
        existing = by_symbol.get(sym)
        if existing is None or entry.get("confidence", 0) > existing.get("confidence", 0):
            by_symbol[sym] = entry

    return list(by_symbol.values())


def score_entry(entry: dict, current_regime: str, regime_confidence: float) -> float:
    """Compute composite buy score in [0.0, 1.0].

    Returns 0.0 for any symbol in High Vol or Extreme Vol (hard gate).

    Scoring weights:
        - R/R to first TP:  30%
        - ATR stop quality: 20%
        - Watchlist conf:   20%
        - Recency:          15%
        - Volume ratio:     10%
        - SMA200 alignment:  5%

    All components scaled by regime exposure so regime acts as top-level multiplier.

    Args:
        entry: Watchlist dict for one symbol.
        current_regime: HMM stable_labels[-1] for this symbol.
        regime_confidence: HMM confidence[-1] for this symbol.

    Returns:
        Float in [0.0, 1.0].
    """
    # Hard gate
    exposure = target_exposure(current_regime, regime_confidence)
    if exposure < 0.35:
        return 0.0

    entry_price = float(entry.get("entry", 0))
    stop_price = float(entry.get("stop", 0))
    risk = entry_price - stop_price

    # R/R score
    tp_ladder = entry.get("tp_ladder") or []
    if tp_ladder and risk > 0:
        reward = float(tp_ladder[0]) - entry_price
        rr = max(0.0, reward / risk)
        rr_score = min(rr / _MAX_RR, 1.0)
    else:
        rr_score = 0.0

    # ATR stop quality (stops < 1× ATR are noise; 2× ATR = perfect)
    atr = float(entry.get("atr20", 0))
    if atr > 0 and risk > 0:
        atr_mult = risk / atr
        atr_score = min(atr_mult / 2.0, 1.0)
    else:
        atr_score = 0.0

    # Watchlist confidence (rescale 0.5–1.0 → 0.0–1.0)
    conf = float(entry.get("confidence", 0.5))
    conf_score = max(0.0, (conf - 0.5) / 0.5)

    # Recency (exponential decay, half-life = _RECENCY_HALF_LIFE_DAYS)
    today_ord = datetime.date.today().toordinal()
    age_days = max(0, today_ord - int(entry.get("added_ordinal", today_ord)))
    recency_score = math.exp(-age_days * math.log(2) / _RECENCY_HALF_LIFE_DAYS)

    # Volume ratio bonus (present only if enriched)
    vol_ratio = float(entry.get("volume_ratio", 1.0))
    vol_score = min(max(0.0, (vol_ratio - 1.0) / 2.0), 1.0) if vol_ratio > 1 else 0.0

    # SMA200 alignment bonus
    sma200 = float(entry.get("sma200", entry_price))
    trend_bonus = 1.0 if entry_price > sma200 else 0.0

    composite = (
        0.30 * rr_score
        + 0.20 * atr_score
        + 0.20 * conf_score
        + 0.15 * recency_score
        + 0.10 * vol_score
        + 0.05 * trend_bonus
    )

    return round(composite * exposure, 4)


def size_position(entry: float, atr20: float, account_equity: float) -> tuple[int, float]:
    """Compute ATR-based position size risking 1% of account equity per trade.

    shares = floor(account_equity × 0.01 / atr20), minimum 1.
    estimated_cost = shares × entry.

    Args:
        entry: Entry price per share.
        atr20: 20-day Average True Range.
        account_equity: Current total account equity in dollars.

    Returns:
        (shares: int, estimated_cost: float)
    """
    if atr20 <= 0:
        return 1, round(entry, 4)
    risk_dollars = account_equity * _RISK_PER_TRADE
    shares = max(1, math.floor(risk_dollars / atr20))
    cost = round(shares * entry, 4)
    return shares, cost


def scan(
    watchlist_path: str | Path,
    account_equity: float,
    top_n: int = 10,
    benchmark: str = "SPY",
    lookback_years: int = 2,
) -> list[ScanResult]:
    """Score all watchlist symbols and return top_n buy candidates.

    For each symbol:
      1. Fetches OHLCV (lookback_years of daily data).
      2. Runs HMM fit_and_filter to get current regime + confidence.
      3. Applies hard gate: drops High Vol / Extreme Vol.
      4. Computes composite score via score_entry().
      5. Sizes position via size_position().

    Symbols that fail data fetch or HMM fit are skipped with a warning.

    Args:
        watchlist_path: Path to watchlist.json.
        account_equity: Live account equity used for position sizing.
        top_n: Maximum number of results to return.
        benchmark: Ticker used only for regime detection of overall market (unused in current impl — per-symbol regime is used).
        lookback_years: Years of OHLCV history to fetch per symbol.

    Returns:
        List of ScanResult sorted by final_score descending, length <= top_n.
    """
    entries = load_watchlist(watchlist_path)
    end = datetime.date.today().isoformat()
    start = (datetime.date.today() - datetime.timedelta(days=365 * lookback_years)).isoformat()

    results: list[ScanResult] = []
    skipped: list[ScanResult] = []

    for entry in entries:
        sym = entry["symbol"]
        try:
            df = load_ohlcv(sym, start, end)
            df.columns = [c.lower() for c in df.columns]
            regime_result = fit_and_filter(df)
            current_regime = regime_result.stable_labels[-1]
            regime_conf = float(regime_result.confidence[-1])
        except Exception as exc:
            warnings.warn(f"Skipping {sym}: {exc}")
            skipped.append(ScanResult(
                symbol=sym, regime="", regime_confidence=0.0,
                watchlist_confidence=float(entry.get("confidence", 0)),
                rr_score=0.0, recency_score=0.0, final_score=0.0,
                shares=0, estimated_cost=0.0,
                entry=float(entry.get("entry", 0)),
                stop=float(entry.get("stop", 0)),
                tp1=None, skipped=True, skip_reason=str(exc),
            ))
            continue

        final_score = score_entry(entry, current_regime, regime_conf)

        tp_ladder = entry.get("tp_ladder") or []
        tp1 = float(tp_ladder[0]) if tp_ladder else None

        shares, cost = size_position(
            entry=float(entry["entry"]),
            atr20=float(entry.get("atr20", 1.0)),
            account_equity=account_equity,
        )

        # Compute sub-scores for display
        risk = float(entry["entry"]) - float(entry["stop"])
        rr = (tp1 - float(entry["entry"])) / risk if tp1 and risk > 0 else 0.0
        rr_score = min(rr / _MAX_RR, 1.0) if rr > 0 else 0.0

        today_ord = datetime.date.today().toordinal()
        age = max(0, today_ord - int(entry.get("added_ordinal", today_ord)))
        recency_score = round(math.exp(-age * math.log(2) / _RECENCY_HALF_LIFE_DAYS), 4)

        result = ScanResult(
            symbol=sym,
            regime=current_regime,
            regime_confidence=round(regime_conf, 3),
            watchlist_confidence=float(entry.get("confidence", 0)),
            rr_score=round(rr_score, 3),
            recency_score=recency_score,
            final_score=final_score,
            shares=shares,
            estimated_cost=cost,
            entry=float(entry["entry"]),
            stop=float(entry["stop"]),
            tp1=tp1,
        )

        if final_score > 0.0:
            results.append(result)
        else:
            skipped.append(result._replace(skipped=True, skip_reason=f"Regime gate: {current_regime}") if hasattr(result, '_replace') else result)

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results[:top_n]
```

- [ ] **Step 5: Run tests to verify they pass**

```
C:\Python311\python.exe -m pytest tests/test_scanner.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Run full suite**

```
C:\Python311\python.exe -m pytest tests/ -v --tb=short
```

Expected: all previously passing tests still pass.

- [ ] **Step 7: Commit**

```bash
git add swing/__init__.py swing/scanner.py tests/test_scanner.py
git commit -m "feat: add swing/scanner.py — regime-aware scoring and ATR position sizing"
```

---

## Task 3: Add Scanner UI to `app.py`

**Files:**
- Modify: `app.py`

Adds a **Swing Scanner** section to the landing page. "Run Scan" triggers `scanner.scan()`. Results appear as a table with a per-row "Buy {symbol}" button. Clicking a button calls `AlpacaBroker.submit_order()` for that symbol only. After a buy, the button becomes a disabled "Ordered ✓" label.

- [ ] **Step 1: Replace `app.py` with the full updated version**

```python
"""RegimeTrading — Streamlit multi-page hub with Swing Scanner."""
from __future__ import annotations

import math
import os
from pathlib import Path

import streamlit as st

from core.broker import AlpacaBroker
from core.design_system import REGIME_COLORS
from swing.scanner import scan, ScanResult

st.set_page_config(
    page_title="RegimeTrading",
    page_icon="📈",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "scan_results" not in st.session_state:
    st.session_state.scan_results: list[ScanResult] = []
if "scan_skipped" not in st.session_state:
    st.session_state.scan_skipped: list[ScanResult] = []
if "ordered" not in st.session_state:
    st.session_state.ordered: set[str] = set()
if "account_equity" not in st.session_state:
    st.session_state.account_equity: float = 0.0
if "broker_error" not in st.session_state:
    st.session_state.broker_error: str = ""

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("RegimeTrading")
st.subheader("Adaptive algorithmic trading driven by market-regime detection")
st.caption(
    "Use the sidebar to navigate between dashboards. "
    "Run the Swing Scanner below to find top buy candidates from your watchlist."
)

st.divider()

# ---------------------------------------------------------------------------
# Swing Scanner section
# ---------------------------------------------------------------------------
st.header("Swing Scanner")

WATCHLIST_PATH = Path(__file__).parent / "swing" / "watchlist.json"

# Fetch account equity once per session or on scan
def _fetch_equity() -> tuple[float, str]:
    """Return (equity, error_message). error_message is '' on success."""
    try:
        broker = AlpacaBroker()
        account = broker.get_account()
        return float(account.get("equity", 10_000.0)), ""
    except Exception as exc:
        return 10_000.0, str(exc)


col_btn, col_equity = st.columns([2, 3])
with col_btn:
    run_scan = st.button("🔍 Run Scan", type="primary", use_container_width=True)
with col_equity:
    if st.session_state.account_equity:
        st.metric("Account Equity", f"${st.session_state.account_equity:,.2f}")
    elif st.session_state.broker_error:
        st.warning(f"Broker unavailable — using $10,000 placeholder. {st.session_state.broker_error}")

# ---------------------------------------------------------------------------
# Run scan
# ---------------------------------------------------------------------------
if run_scan:
    equity, err = _fetch_equity()
    st.session_state.account_equity = equity
    st.session_state.broker_error = err
    st.session_state.ordered = set()  # reset per scan

    with st.spinner("Running regime detection across watchlist symbols…"):
        try:
            results = scan(
                watchlist_path=WATCHLIST_PATH,
                account_equity=equity,
                top_n=10,
            )
            st.session_state.scan_results = results
        except Exception as exc:
            st.error(f"Scanner failed: {exc}")
            st.session_state.scan_results = []

# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------
if st.session_state.scan_results:
    st.subheader("Top 10 Buy Candidates")

    results: list[ScanResult] = st.session_state.scan_results

    header_cols = st.columns([0.4, 1.2, 1.4, 0.8, 0.8, 1.0, 0.9, 0.9, 1.2])
    for col, label in zip(header_cols, ["#", "Symbol", "Regime", "Score", "Shares", "Est. Cost", "Stop", "TP1", "Action"]):
        col.markdown(f"**{label}**")

    st.markdown("<hr style='margin:4px 0 8px'>", unsafe_allow_html=True)

    for i, r in enumerate(results, start=1):
        regime_color = REGIME_COLORS.get(r.regime, "#94a3b8")
        cols = st.columns([0.4, 1.2, 1.4, 0.8, 0.8, 1.0, 0.9, 0.9, 1.2])
        cols[0].write(f"**{i}**")
        cols[1].markdown(f"<span style='color:{regime_color};font-weight:700'>{r.symbol}</span>", unsafe_allow_html=True)
        cols[2].markdown(f"<span style='color:{regime_color}'>{r.regime}</span>", unsafe_allow_html=True)
        cols[3].write(f"{r.final_score:.3f}")
        cols[4].write(str(r.shares))
        cols[5].write(f"${r.estimated_cost:,.2f}")
        cols[6].write(f"${r.stop:.2f}")
        cols[7].write(f"${r.tp1:.2f}" if r.tp1 else "—")

        with cols[8]:
            already_ordered = r.symbol in st.session_state.ordered
            if already_ordered:
                st.success("Ordered ✓", icon="✅")
            else:
                if st.button(f"Buy {r.symbol}", key=f"buy_{r.symbol}_{i}"):
                    try:
                        broker = AlpacaBroker()
                        broker.submit_order(
                            symbol=r.symbol,
                            qty=float(r.shares),
                            side="buy",
                        )
                        st.session_state.ordered.add(r.symbol)
                        st.success(f"Order placed: {r.shares} × {r.symbol} @ ~${r.entry:.2f}")
                        st.rerun()
                    except RuntimeError as exc:
                        st.error(f"Order rejected: {exc}")

    st.markdown("<hr style='margin:8px 0 4px'>", unsafe_allow_html=True)
    st.caption(
        f"Positions sized at 1% account risk per trade. "
        f"All orders route through 5 safety circuit breakers. "
        f"Paper trading active (LIVE_TRADING={os.getenv('LIVE_TRADING','false')})."
    )

elif not run_scan:
    st.info("Click **Run Scan** to score your watchlist symbols against the current market regime.")

# ---------------------------------------------------------------------------
# Dashboard navigation reminder
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    """
    **Dashboards** — use the sidebar to navigate:
    | # | Dashboard | Purpose |
    |---|-----------|---------|
    | 1 | Regime Detection | Live HMM regime timeline |
    | 2 | Monte Carlo | Future return probability fan |
    | 3 | Sensitivity | SMA parameter sweep |
    | 4 | Portfolio Risk | Stress tests & drawdown |
    | 5 | Multi-Asset Backtest | HMM strategy vs buy-and-hold |
    | 6 | Sentiment | VADER news sentiment |
    | 7 | Correlation Breaks | Z-score break alerts |
    """
)
```

- [ ] **Step 2: Restart Streamlit and verify the UI loads**

Kill any running Streamlit process, then:

```
C:\Users\Bruce\AppData\Roaming\Python\Python311\Scripts\streamlit.exe run app.py
```

Open `http://localhost:8501`. Verify:
- "Swing Scanner" section is visible on the landing page
- "Run Scan" button is present
- Clicking it shows a spinner, then the top-10 table
- Each row has a "Buy {symbol}" button
- Clicking a Buy button shows "Ordered ✓" after success

- [ ] **Step 3: Run full test suite**

```
C:\Python311\python.exe -m pytest tests/ -v --tb=short
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat: add Swing Scanner UI to landing page with per-symbol Buy buttons"
```

---

## Task 4: Push Everything to GitHub

- [ ] **Step 1: Verify clean state**

```
git status
git log --oneline -5
```

- [ ] **Step 2: Push branch**

```
git push origin phase-1-core-modules
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task that covers it |
|---|---|
| Fix critical safety bypass (broker.py placeholders) | Task 1 |
| Add input validation on symbol/qty/side | Task 1 |
| Load and deduplicate watchlist | Task 2 — `load_watchlist()` |
| Fix anomalous added_ordinal values | Task 2 — `load_watchlist()` |
| Combined score: regime × conf × R/R × recency | Task 2 — `score_entry()` |
| Hard gate: High Vol / Extreme Vol excluded | Task 2 — `score_entry()` returns 0.0 |
| ATR-based 1% risk position sizing | Task 2 — `size_position()` |
| Skip symbol on data/HMM error, continue | Task 2 — `scan()` try/except |
| Per-symbol approve Buy button in app.py | Task 3 |
| "Ordered ✓" state after buy | Task 3 — `st.session_state.ordered` |
| Paper trading default shown in UI | Task 3 — footer caption |
| All tests pass | Tasks 1, 2, 3 each run full suite |
| Push to GitHub | Task 4 |

**Placeholder scan:** No TBDs, no vague steps. Every step has exact code or exact commands.

**Type consistency:** `ScanResult` is defined in Task 2 and imported in Task 3. `scan()` returns `list[ScanResult]`. `load_watchlist()`, `score_entry()`, `size_position()` are defined in Task 2 and tested in Task 2. No cross-task name drift found.
