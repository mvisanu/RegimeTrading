"""
core/data.py
============
Thin yfinance wrapper for downloading OHLCV market data.

This module has no Streamlit dependency and no internal caching — callers are
responsible for wrapping these functions with their own cache layer (e.g.
``st.cache_data`` in a dashboard, ``functools.lru_cache`` in scripts, or no
cache at all in tests).

Usage example::

    from core.data import load_ohlcv, date_range_default

    start, end = date_range_default(years=3)
    df = load_ohlcv("SPY", start, end)
    print(df.tail())
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta


def load_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLCV data from Yahoo Finance.

    Caching is the caller's responsibility — this function performs a live
    network request every time it is called.

    Parameters
    ----------
    ticker:
        Yahoo Finance ticker symbol, e.g. ``"SPY"`` or ``"AAPL"``.
    start:
        Start date as an ISO 8601 string, e.g. ``"2022-01-01"`` (inclusive).
    end:
        End date as an ISO 8601 string, e.g. ``"2024-12-31"`` (exclusive per
        yfinance convention — the last trading day returned is the one before
        *end* when *end* is a non-trading day, otherwise *end* itself).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by date with columns:
        ``Open``, ``High``, ``Low``, ``Close``, ``Volume``.
        Column names follow the standard yfinance capitalisation.

    Raises
    ------
    ValueError
        If the download returns an empty DataFrame, which indicates an invalid
        ticker, a date range that contains no trading days, or a network error
        that silently returned no rows.

    Examples
    --------
    >>> df = load_ohlcv("SPY", "2023-01-01", "2023-12-31")
    >>> list(df.columns)
    ['Open', 'High', 'Low', 'Close', 'Volume']
    """
    raw: pd.DataFrame = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for ticker={ticker!r} "
            f"between {start!r} and {end!r}. "
            "Check that the ticker is valid and the date range contains trading days."
        )

    # yfinance may return a MultiIndex column when downloading a single ticker
    # in certain versions (e.g. columns like ("Close", "SPY")).  Flatten to
    # simple string column names so callers always see the same interface.
    # Level order has historically varied across yfinance versions, so detect
    # which level holds the price-field names rather than assuming level 0.
    if isinstance(raw.columns, pd.MultiIndex):
        price_fields = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
        level = next(
            (i for i, level_vals in enumerate(raw.columns.levels)
             if price_fields & set(level_vals)),
            0  # fallback to level 0 if detection fails
        )
        raw.columns = raw.columns.get_level_values(level)

    ohlcv_cols = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in ohlcv_cols if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Downloaded data for {ticker!r} is missing expected columns: {missing}. "
            f"Got columns: {list(raw.columns)}"
        )

    return raw[ohlcv_cols].copy()


def date_range_default(years: int = 3) -> tuple[str, str]:
    """Return a ``(start, end)`` date string pair covering the last *years* years.

    Both strings are ISO 8601 formatted (``"YYYY-MM-DD"``).  *end* is today;
    *start* is exactly *years* calendar years before today.  Dashboards use
    this to pre-populate their date range inputs.

    Parameters
    ----------
    years:
        Number of calendar years to look back from today.  Defaults to 3.

    Returns
    -------
    tuple[str, str]
        ``(start_str, end_str)`` where both values are ``"YYYY-MM-DD"`` strings.

    Examples
    --------
    >>> start, end = date_range_default(years=1)
    >>> end == str(date.today())
    True
    """
    today = date.today()
    start = today - relativedelta(years=years)
    return start.isoformat(), today.isoformat()


# ---------------------------------------------------------------------------
# Smoke test — run directly:  python core/data.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Smoke test: load_ohlcv + date_range_default")

    start, end = date_range_default(years=2)
    print(f"  date_range_default(2) -> start={start!r}, end={end!r}")

    ticker = "SPY"
    print(f"  Downloading {ticker} from {start} to {end} ...")
    df = load_ohlcv(ticker, start, end)
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  First row:\n{df.head(1)}")
    print(f"  Last row:\n{df.tail(1)}")
    print("  PASS")

    # Test error path
    print("\n  Testing ValueError on bad ticker ...")
    try:
        load_ohlcv("INVALID_TICKER_XYZ_999", start, end)
        print("  FAIL — expected ValueError was not raised")
    except ValueError as exc:
        print(f"  ValueError raised as expected: {exc}")
        print("  PASS")
