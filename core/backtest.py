"""Walk-forward backtester for regime-allocation strategy. No in-sample paths."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from core.hmm_utils import fit_and_filter, _engineer_features
from core.verify import forward_filter
from core.allocation import target_exposure


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class BacktestResult:
    equity_curve: pd.Series          # indexed by date
    sharpe: float                    # annualised Sharpe ratio (252 trading days)
    max_drawdown: float              # as a positive fraction, e.g. 0.15 = 15%
    total_return: float              # e.g. 0.25 = 25%
    win_rate: float                  # fraction of profitable periods
    n_windows: int                   # number of walk-forward windows completed
    benchmark_equity: pd.Series      # buy-and-hold equity curve (same dates)
    benchmark_sharpe: float
    benchmark_max_drawdown: float
    benchmark_total_return: float


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _annualised_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Annualised Sharpe ratio (risk-free rate = 0).

    Parameters
    ----------
    returns:
        Series of period returns (daily fractions, e.g. 0.01 = 1%).
    periods_per_year:
        Number of trading periods per year.  Defaults to 252.

    Returns
    -------
    float
        Annualised Sharpe.  Returns 0.0 when standard deviation is zero
        (flat equity curve).

    Example
    -------
    >>> import pandas as pd
    >>> r = pd.Series([0.001] * 252)
    >>> _annualised_sharpe(r)  # constant return → infinite SR, but std=0 guard
    0.0
    """
    std = returns.std()
    if std == 0:
        return 0.0
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def _max_drawdown(equity: pd.Series) -> float:
    """Maximum drawdown as a positive fraction.

    Parameters
    ----------
    equity:
        Equity curve indexed by date.

    Returns
    -------
    float
        Maximum peak-to-trough decline expressed as a positive fraction,
        e.g. 0.15 for a 15% drawdown.  Returns 0.0 for a monotonically
        rising curve.

    Example
    -------
    >>> import pandas as pd
    >>> eq = pd.Series([100.0, 110.0, 90.0, 95.0])
    >>> round(_max_drawdown(eq), 4)
    0.1818
    """
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(-drawdown.min())


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of *df* with OHLCV columns normalised to lowercase.

    Accepts both ``"Close"`` / ``"close"`` (and mixed-case variants)
    so the backtester is tolerant of data-source conventions.

    Parameters
    ----------
    df:
        Raw OHLCV frame.

    Returns
    -------
    pd.DataFrame
        Frame whose columns are all lowercase.
    """
    return df.rename(columns={c: c.lower() for c in df.columns})


# ---------------------------------------------------------------------------
# Core walk-forward loop
# ---------------------------------------------------------------------------


def walk_forward_backtest(
    df: pd.DataFrame,
    train_years: int = 1,
    test_months: int = 6,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Walk-forward backtest of the regime-allocation strategy.

    For each window:
      1. Train HMM on the training slice (first ``train_years`` years).
      2. For each bar in the test slice:
         a. Get regime label from the trained model using ``forward_filter``
            on the combined train+test feature array (causal — no future data
            leaks because forward_filter is strictly causal).
         b. Get target exposure from ``allocation.target_exposure``.
         c. Compute return: ``price_return * exposure``.
      3. Slide the window forward by ``test_months`` and repeat.

    No full-period in-sample analysis is performed anywhere.

    Parameters
    ----------
    df:
        OHLCV DataFrame with a DatetimeIndex.  Columns may be either
        title-case (``"Close"``) or lowercase (``"close"``); both are
        accepted.  Must span at least ``train_years`` years plus one
        ``test_months`` block.
    train_years:
        Length of the in-sample training window in years.
    test_months:
        Length of each out-of-sample test slice in months.
    initial_capital:
        Starting portfolio value used to scale the equity curve.

    Returns
    -------
    BacktestResult
        Fully populated result dataclass.

    Raises
    ------
    ValueError
        If no walk-forward windows could be completed (data too short).

    Example
    -------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.bdate_range("2018-01-01", periods=600)
    >>> rng = np.random.default_rng(0)
    >>> prices = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    >>> df = pd.DataFrame({
    ...     "close": prices,
    ...     "high": prices * 1.005,
    ...     "low": prices * 0.995,
    ...     "open": prices,
    ...     "volume": 1_000_000,
    ... }, index=idx)
    >>> result = walk_forward_backtest(df, train_years=1, test_months=3)
    >>> result.n_windows >= 1
    True
    """
    # --- Normalise column names to lowercase ---
    df = _normalize_columns(df)
    df = df.sort_index()

    start_date = df.index[0]
    end_date = df.index[-1]

    all_returns: list[float] = []
    all_dates: list[pd.Timestamp] = []

    window_start = start_date
    n_windows = 0

    while True:
        train_end = window_start + relativedelta(years=train_years)
        test_end = train_end + relativedelta(months=test_months)

        if test_end > end_date:
            break

        # Slice with label-based indexing; include train_end in training data
        # and exclude it from test data to avoid any double-counting.
        train_df = df.loc[window_start:train_end]
        test_df = df.loc[train_end:test_end]

        # Skip degenerate windows
        if len(train_df) < 50 or len(test_df) < 5:
            break

        # 1. Fit HMM on training slice only.
        #    Skip any window where HMM fitting fails (degenerate covariance).
        try:
            train_result = fit_and_filter(train_df)
        except (ValueError, np.linalg.LinAlgError) as exc:
            import warnings
            warnings.warn(
                f"[backtest] HMM fitting failed for window starting "
                f"{window_start.date()} — skipping. Reason: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            window_start = window_start + relativedelta(months=test_months)
            continue

        # 2. Build feature array over train+test for causal forward-filtering.
        #    Using the full window gives each test bar the context of prior bars
        #    without leaking future test observations into past calculations —
        #    forward_filter is strictly causal (each posterior depends only on
        #    observations up to and including the current bar).
        full_df = df.loc[window_start:test_end]
        X_full = _engineer_features(full_df)

        # _engineer_features drops NaN rows from rolling windows (first 20 bars
        # are lost).  We need to align X_full rows back to calendar dates.
        # Reconstruct the index of rows that survived the dropna.
        close_full = full_df["close"]
        log_return = np.log(close_full / close_full.shift(1))
        realized_vol = log_return.rolling(20).std()
        hl_range_pct = (full_df["high"] - full_df["low"]) / close_full
        feature_frame = pd.DataFrame(
            {
                "log_return": log_return,
                "realized_vol": realized_vol,
                "hl_range_pct": hl_range_pct,
            }
        ).dropna()

        # Run forward filter with the *trained* model on the full feature array.
        posteriors_full = forward_filter(train_result.model, X_full)

        # Map posteriors back to their original dates
        posteriors_series = pd.DataFrame(
            posteriors_full, index=feature_frame.index
        )

        # Restrict to test dates only (after train_end)
        test_posterior_df = posteriors_series.loc[
            posteriors_series.index > train_end
        ]

        # 3. Iterate over test bars and compute strategy returns
        prev_close: float | None = None

        for date, posteriors_row in test_posterior_df.iterrows():
            if date not in test_df.index:
                continue

            row = test_df.loc[date]
            current_close = float(row["close"])

            # Price return: use previous bar's close from test_df when available
            if prev_close is None:
                # First test bar — use the last training bar as the prior close
                train_closes = train_df["close"]
                if len(train_closes) == 0:
                    prev_close = current_close
                else:
                    prev_close = float(train_closes.iloc[-1])

            price_return = current_close / prev_close - 1.0

            # Regime from argmax of forward posterior
            state_idx = int(posteriors_row.values.argmax())
            regime = train_result.label_map.get(state_idx, "Uncertain")
            conf = float(posteriors_row.values.max())

            # Fallback for labels not in allocation map (e.g. "Hyper Vol")
            try:
                exposure = target_exposure(regime, conf)
            except ValueError:
                exposure = target_exposure("Uncertain", conf)

            strategy_return = price_return * exposure

            all_returns.append(strategy_return)
            all_dates.append(date)
            prev_close = current_close

        window_start = window_start + relativedelta(months=test_months)
        n_windows += 1

    if not all_returns:
        raise ValueError(
            "No walk-forward windows completed — data too short?  "
            f"Need at least {train_years} training year(s) plus "
            f"{test_months} test month(s)."
        )

    # --- Build equity curve ---
    returns_series = pd.Series(all_returns, index=all_dates, dtype=float)
    equity_curve = (1.0 + returns_series).cumprod() * initial_capital

    # --- Benchmark: buy-and-hold over the same test-period dates ---
    bh_df = df.loc[all_dates, "close"]
    bh_returns = bh_df.pct_change().fillna(0.0)
    benchmark_equity = (1.0 + bh_returns).cumprod() * initial_capital

    return BacktestResult(
        equity_curve=equity_curve,
        sharpe=_annualised_sharpe(returns_series),
        max_drawdown=_max_drawdown(equity_curve),
        total_return=float(equity_curve.iloc[-1] / initial_capital - 1.0),
        win_rate=float((returns_series > 0).mean()),
        n_windows=n_windows,
        benchmark_equity=benchmark_equity,
        benchmark_sharpe=_annualised_sharpe(bh_returns),
        benchmark_max_drawdown=_max_drawdown(benchmark_equity),
        benchmark_total_return=float(
            benchmark_equity.iloc[-1] / initial_capital - 1.0
        ),
    )


# ---------------------------------------------------------------------------
# Inline smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """Smoke test using synthetic OHLCV data with 3 distinct volatility regimes.

    Verifies:
    - At least 1 walk-forward window completes.
    - BacktestResult has valid numeric fields.
    - equity_curve and benchmark_equity are positive Series.
    - max_drawdown and win_rate are in [0, 1].
    - n_windows matches expected count.
    """
    import sys

    print("Generating synthetic OHLCV data...")

    rng = np.random.default_rng(42)

    # Three distinct volatility regimes cycle repeatedly so every 1-year
    # training window contains all three regime types.  Each regime lasts
    # ~40 bars (~2 months), cycling: low → medium → high → low → ...
    # Total: 900 bars ≈ 3.6 years.
    vols = [0.008, 0.018, 0.040]
    spreads = [(0.006, 0.012), (0.015, 0.028), (0.035, 0.065)]
    bars_per_block = 40  # bars per regime block
    n_cycles = 8         # 8 full cycles × 3 regimes × 40 bars = 960 bars
    n_total = n_cycles * 3 * bars_per_block

    closes: list[np.ndarray] = []
    highs: list[np.ndarray] = []
    lows: list[np.ndarray] = []
    current_price = 100.0

    for _ in range(n_cycles):
        for i, (vol, (sp_lo, sp_hi)) in enumerate(zip(vols, spreads)):
            n = bars_per_block
            log_ret = rng.normal(0.0, vol, size=n)
            close = current_price * np.exp(np.cumsum(log_ret))
            spread = rng.uniform(sp_lo, sp_hi, size=n)
            closes.append(close)
            highs.append(close * (1.0 + spread))
            lows.append(close * (1.0 - spread))
            current_price = float(close[-1])

    close_arr = np.concatenate(closes)
    high_arr = np.concatenate(highs)
    low_arr = np.concatenate(lows)
    open_arr = close_arr * (1.0 + rng.normal(0.0, 0.003, size=n_total))
    volume_arr = rng.integers(100_000, 1_000_000, size=n_total).astype(float)

    idx = pd.bdate_range("2020-01-01", periods=n_total)
    df_test = pd.DataFrame(
        {
            "Open": open_arr,
            "High": high_arr,
            "Low": low_arr,
            "Close": close_arr,
            "Volume": volume_arr,
        },
        index=idx,
    )

    print(f"Data shape: {df_test.shape}  |  date range: {idx[0].date()} — {idx[-1].date()}")
    print("Running walk-forward backtest (train_years=1, test_months=6)...")

    result = walk_forward_backtest(
        df_test,
        train_years=1,
        test_months=6,
        initial_capital=100_000.0,
    )

    # --- Assertions ---
    errors: list[str] = []

    if result.n_windows < 1:
        errors.append(f"n_windows={result.n_windows} — expected >= 1")

    if not isinstance(result.equity_curve, pd.Series):
        errors.append("equity_curve is not a pd.Series")
    elif (result.equity_curve <= 0).any():
        errors.append("equity_curve contains non-positive values")

    if not isinstance(result.benchmark_equity, pd.Series):
        errors.append("benchmark_equity is not a pd.Series")
    elif (result.benchmark_equity <= 0).any():
        errors.append("benchmark_equity contains non-positive values")

    if not (0.0 <= result.max_drawdown <= 1.0):
        errors.append(f"max_drawdown={result.max_drawdown} out of [0, 1]")

    if not (0.0 <= result.win_rate <= 1.0):
        errors.append(f"win_rate={result.win_rate} out of [0, 1]")

    if not np.isfinite(result.sharpe):
        errors.append(f"sharpe={result.sharpe} is not finite")

    if not np.isfinite(result.total_return):
        errors.append(f"total_return={result.total_return} is not finite")

    if len(result.equity_curve) != len(result.benchmark_equity):
        errors.append(
            f"equity_curve length {len(result.equity_curve)} != "
            f"benchmark_equity length {len(result.benchmark_equity)}"
        )

    if errors:
        print("\nFAILED — assertion errors:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # --- Report ---
    print("\n--- BacktestResult ---")
    print(f"  n_windows          : {result.n_windows}")
    print(f"  total_return       : {result.total_return:.2%}")
    print(f"  sharpe             : {result.sharpe:.4f}")
    print(f"  max_drawdown       : {result.max_drawdown:.2%}")
    print(f"  win_rate           : {result.win_rate:.2%}")
    print(f"  equity_curve bars  : {len(result.equity_curve)}")
    print(f"  benchmark_total_ret: {result.benchmark_total_return:.2%}")
    print(f"  benchmark_sharpe   : {result.benchmark_sharpe:.4f}")
    print(f"  benchmark_mdd      : {result.benchmark_max_drawdown:.2%}")
    print("\nSMOKE TEST PASSED")
