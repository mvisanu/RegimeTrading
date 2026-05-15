"""Unit tests for swing/scanner.py pure scoring and sizing logic."""
import datetime
import json
import math
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from swing.scanner import (
    load_watchlist,
    score_entry,
    size_position,
    scan,
    ScanResult,
)

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
    wl = [{**VALID_ENTRY, "added_ordinal": 3}]
    p = tmp_path / "watchlist.json"
    p.write_text(json.dumps(wl))
    result = load_watchlist(str(p))
    assert result[0]["added_ordinal"] == TODAY_ORD


def test_load_watchlist_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_watchlist("/nonexistent/watchlist.json")


def test_score_low_vol_beats_medium_vol():
    low = score_entry(VALID_ENTRY, "Low Vol", 0.85)
    med = score_entry(VALID_ENTRY, "Medium Vol", 0.85)
    assert low > med


def test_score_high_vol_returns_zero():
    assert score_entry(VALID_ENTRY, "High Vol", 0.90) == 0.0


def test_score_extreme_vol_returns_zero():
    assert score_entry(VALID_ENTRY, "Extreme Vol", 0.90) == 0.0


def test_score_empty_tp_ladder_penalised():
    no_tp = {**VALID_ENTRY, "tp_ladder": []}
    with_tp = VALID_ENTRY
    assert score_entry(no_tp, "Low Vol", 0.85) < score_entry(with_tp, "Low Vol", 0.85)


def test_rr_score_capped_at_3():
    huge_tp = {**VALID_ENTRY, "tp_ladder": [100.0]}
    score = score_entry(huge_tp, "Low Vol", 1.0)
    assert 0.0 <= score <= 1.0


def test_score_is_float_in_unit_range():
    s = score_entry(VALID_ENTRY, "Low Vol", 0.76)
    assert isinstance(s, float)
    assert 0.0 <= s <= 1.0


def test_size_position_atr_formula():
    shares, cost = size_position(entry=13.60, atr20=0.46, account_equity=10_000.0)
    expected_shares = math.floor(10_000.0 * 0.01 / 0.46)
    assert shares == expected_shares
    assert cost == pytest.approx(shares * 13.60)


def test_size_position_minimum_one_share():
    shares, _ = size_position(entry=5.0, atr20=10_000.0, account_equity=1_000.0)
    assert shares == 1


def test_size_position_zero_atr_returns_one_share():
    shares, _ = size_position(entry=10.0, atr20=0.0, account_equity=10_000.0)
    assert shares == 1


def test_scan_returns_top_n(tmp_path):
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
    wl = []
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
