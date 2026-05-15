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
