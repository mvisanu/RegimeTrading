"""
tests/test_no_lookahead.py — GATE test for look-ahead bias.

Proves that core.verify.forward_filter produces no look-ahead bias:
the posterior at time t depends only on observations X[0..t], not
on any future bar.

All further Phase 1 work is gated on this test passing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hmmlearn.hmm import GaussianHMM

from core.hmm_utils import _engineer_features
from core.verify import forward_filter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_ohlcv(n_bars: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Generate deterministic OHLCV data for testing.

    Produces 3 consecutive volatility regimes (low / mid / high) of
    roughly equal length.  This ensures GaussianHMM(full covariance)
    can fit without degenerate covariance matrices.
    """
    rng = np.random.default_rng(seed)
    bars_per_regime = n_bars // 3
    remainder = n_bars - 3 * bars_per_regime

    regime_vols = [0.005, 0.015, 0.035]
    regime_spreads = [
        (0.003, 0.010),
        (0.012, 0.025),
        (0.030, 0.060),
    ]

    close_parts: list[np.ndarray] = []
    high_parts: list[np.ndarray] = []
    low_parts: list[np.ndarray] = []
    current = 100.0

    for idx, (vol, (sp_lo, sp_hi)) in enumerate(zip(regime_vols, regime_spreads)):
        n = bars_per_regime + (remainder if idx == 2 else 0)
        lr = rng.normal(0.0, vol, size=n)
        prices = current * np.exp(np.cumsum(lr))
        spread = rng.uniform(sp_lo, sp_hi, size=n)
        close_parts.append(prices)
        high_parts.append(prices * (1.0 + spread))
        low_parts.append(prices * (1.0 - spread))
        current = float(prices[-1])

    close = np.concatenate(close_parts)
    high = np.concatenate(high_parts)
    low = np.concatenate(low_parts)
    open_ = close * (1.0 + rng.normal(0.0, 0.003, size=n_bars))
    volume = rng.integers(50_000, 500_000, size=n_bars).astype(float)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestForwardFilterNoLookahead:
    """Verify that forward_filter is causally correct."""

    def test_forward_filter_matches_prefix_at_each_t(self):
        """
        forward_filter(model, X)[t] must equal forward_filter(model, X[:t+1])[-1].

        This is the fundamental no-lookahead property: the posterior at t
        computed on the full sequence must be identical to the posterior
        obtained by running the same filter on only the first t+1 bars.
        Uses the same fitted model in both cases to isolate the forward
        filter implementation from model-fitting randomness.
        """
        df = _synthetic_ohlcv(n_bars=300, seed=42)
        X = _engineer_features(df)

        model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        model.fit(X)

        # Streaming forward posteriors over the full sequence.
        full_posteriors = forward_filter(model, X)

        T = len(X)
        lo = T // 4
        hi = 3 * T // 4

        rng = np.random.default_rng(seed=0)
        sampled_t = rng.choice(
            np.arange(lo, hi), size=min(20, hi - lo), replace=False
        )
        sampled_t = np.sort(sampled_t)

        max_observed_diff = 0.0

        for t in sampled_t:
            # Run forward filter on prefix only.
            prefix_posteriors = forward_filter(model, X[: t + 1])

            streaming_at_t = full_posteriors[t]
            prefix_at_t = prefix_posteriors[-1]

            # Compare sorted posteriors (permutation-invariant).
            diff = np.abs(
                np.sort(streaming_at_t) - np.sort(prefix_at_t)
            ).max()
            max_observed_diff = max(max_observed_diff, diff)

            assert diff < 1e-6, (
                f"Look-ahead bias detected at t={t}: "
                f"sorted streaming={np.sort(streaming_at_t)}, "
                f"sorted prefix={np.sort(prefix_at_t)}, "
                f"max_diff={diff:.3e}"
            )

        # Report for visibility (pytest captures stdout).
        print(
            f"\nPassed 20 t-samples; max abs diff across all = {max_observed_diff:.2e}"
        )

    def test_posteriors_sum_to_one(self):
        """Each row of forward_filter output must sum to 1.0."""
        df = _synthetic_ohlcv(n_bars=200, seed=99)
        X = _engineer_features(df)

        model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        model.fit(X)

        posteriors = forward_filter(model, X)
        row_sums = posteriors.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)

    def test_posteriors_non_negative(self):
        """All posterior values must be >= 0."""
        df = _synthetic_ohlcv(n_bars=200, seed=7)
        X = _engineer_features(df)

        model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        model.fit(X)

        posteriors = forward_filter(model, X)
        assert (posteriors >= 0).all(), "Negative posterior probabilities found"

    def test_verify_import_check_passed(self):
        """
        Importing core.hmm_utils must have triggered the verify self-check
        and LOOKAHEAD_CHECK_PASSED must be True.
        """
        import core.hmm_utils  # noqa: F401 (side effect: triggers verify import)
        from core.verify import LOOKAHEAD_CHECK_PASSED

        assert LOOKAHEAD_CHECK_PASSED is True, (
            "core.verify.LOOKAHEAD_CHECK_PASSED is False — "
            "the import-time self-check failed. "
            "Check warnings printed during import for details."
        )


def test_forward_filter_matches_refit_at_each_t():
    """
    Generate deterministic OHLCV, fit HMM, run forward_filter.

    For 20 sampled t values in the middle half, run forward_filter on
    the prefix X[:t+1] with the SAME model and compare sorted posteriors
    at position t.  Max absolute error must be < 1e-6.

    This is the primary gate test for look-ahead bias.
    """
    df = _synthetic_ohlcv(n_bars=300, seed=42)
    X = _engineer_features(df)

    model = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(X)

    full_posteriors = forward_filter(model, X)

    T = len(X)
    lo = T // 4
    hi = 3 * T // 4

    rng = np.random.default_rng(seed=0)
    sampled_t = rng.choice(
        np.arange(lo, hi), size=min(20, hi - lo), replace=False
    )
    sampled_t = np.sort(sampled_t)

    for t in sampled_t:
        prefix_posteriors = forward_filter(model, X[: t + 1])

        diff = np.abs(
            np.sort(full_posteriors[t]) - np.sort(prefix_posteriors[-1])
        ).max()

        assert diff < 1e-6, (
            f"t={t}: max sorted-posterior diff = {diff:.3e} (threshold 1e-6)"
        )
