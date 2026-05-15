"""Regime-aware swing trade scanner.

Pure functions only — no Streamlit, no broker calls, no side effects.
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

_MIN_VALID_ORDINAL = datetime.date(2020, 1, 1).toordinal()
_RISK_PER_TRADE = 0.01
_MAX_RR = 3.0
_RECENCY_HALF_LIFE_DAYS = 3.0


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


def load_watchlist(path: str | Path) -> list[dict]:
    """Load watchlist JSON, deduplicate by symbol, fix anomalous ordinals.

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
    for entry in raw:
        if entry.get("added_ordinal", 0) < _MIN_VALID_ORDINAL:
            entry["added_ordinal"] = today_ord

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

    Returns 0.0 for High Vol or Extreme Vol (hard gate).

    Weights: R/R 30%, ATR quality 20%, watchlist conf 20%, recency 15%, volume 10%, SMA200 5%.
    All multiplied by regime exposure.
    """
    if current_regime in ("High Vol", "Extreme Vol"):
        return 0.0

    exposure = target_exposure(current_regime, regime_confidence)
    if exposure < 0.35:
        return 0.0

    entry_price = float(entry.get("entry", 0))
    stop_price = float(entry.get("stop", 0))
    risk = entry_price - stop_price

    tp_ladder = entry.get("tp_ladder") or []
    if tp_ladder and risk > 0:
        reward = float(tp_ladder[0]) - entry_price
        rr = max(0.0, reward / risk)
        rr_score = min(rr / _MAX_RR, 1.0)
    else:
        rr_score = 0.0

    atr = float(entry.get("atr20", 0))
    if atr > 0 and risk > 0:
        atr_score = min((risk / atr) / 2.0, 1.0)
    else:
        atr_score = 0.0

    conf = float(entry.get("confidence", 0.5))
    conf_score = max(0.0, (conf - 0.5) / 0.5)

    today_ord = datetime.date.today().toordinal()
    age_days = max(0, today_ord - int(entry.get("added_ordinal", today_ord)))
    recency_score = math.exp(-age_days * math.log(2) / _RECENCY_HALF_LIFE_DAYS)

    vol_ratio = float(entry.get("volume_ratio", 1.0))
    vol_score = min(max(0.0, (vol_ratio - 1.0) / 2.0), 1.0) if vol_ratio > 1 else 0.0

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
    """ATR-based position size: risk 1% of account equity per trade.

    Returns (shares, estimated_cost). Minimum 1 share.
    """
    if atr20 <= 0:
        return 1, round(entry, 4)
    shares = max(1, math.floor(account_equity * _RISK_PER_TRADE / atr20))
    return shares, round(shares * entry, 4)


def scan(
    watchlist_path: str | Path,
    account_equity: float,
    top_n: int = 10,
    lookback_years: int = 2,
) -> list[ScanResult]:
    """Score all watchlist symbols and return top_n buy candidates.

    Symbols in High Vol / Extreme Vol are excluded (hard gate).
    Symbols that fail data fetch or HMM fit are skipped with a warning.

    Returns list sorted by final_score descending, length <= top_n.
    """
    entries = load_watchlist(watchlist_path)
    end = datetime.date.today().isoformat()
    start = (datetime.date.today() - datetime.timedelta(days=365 * lookback_years)).isoformat()

    results: list[ScanResult] = []

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
            continue

        final_score = score_entry(entry, current_regime, regime_conf)

        tp_ladder = entry.get("tp_ladder") or []
        tp1 = float(tp_ladder[0]) if tp_ladder else None
        shares, cost = size_position(
            entry=float(entry["entry"]),
            atr20=float(entry.get("atr20", 1.0)),
            account_equity=account_equity,
        )

        risk = float(entry["entry"]) - float(entry["stop"])
        rr = (tp1 - float(entry["entry"])) / risk if tp1 and risk > 0 else 0.0
        rr_score = round(min(rr / _MAX_RR, 1.0) if rr > 0 else 0.0, 3)

        today_ord = datetime.date.today().toordinal()
        age = max(0, today_ord - int(entry.get("added_ordinal", today_ord)))
        recency_score = round(math.exp(-age * math.log(2) / _RECENCY_HALF_LIFE_DAYS), 4)

        if final_score > 0.0:
            results.append(ScanResult(
                symbol=sym,
                regime=current_regime,
                regime_confidence=round(regime_conf, 3),
                watchlist_confidence=float(entry.get("confidence", 0)),
                rr_score=rr_score,
                recency_score=recency_score,
                final_score=final_score,
                shares=shares,
                estimated_cost=cost,
                entry=float(entry["entry"]),
                stop=float(entry["stop"]),
                tp1=tp1,
            ))

    results.sort(key=lambda r: r.final_score, reverse=True)
    return results[:top_n]
