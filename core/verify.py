"""
verify.py — Look-ahead bias verifier for HMM forward filter.

Proves that forward_filter(model, X)[t] depends only on observations
X[:t+1], not on any future observations. This is done by comparing the
streaming output at position t against running forward_filter on the
prefix X[:t+1] with the SAME model — the results must be identical.

Module-level code runs a self-check on synthetic data when imported.
"""

from __future__ import annotations

import warnings

import numpy as np
from hmmlearn import _hmmc
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

# Module-level flag set by _run_import_check(); dashboards can inspect this.
LOOKAHEAD_CHECK_PASSED: bool = False


def forward_filter(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """
    Forward-only filter: returns P(state_t | obs_1..t) for every t.

    Uses only the forward lattice (no backward pass), so each row at
    index t depends exclusively on observations X[0..t].

    Parameters
    ----------
    model : GaussianHMM
        A fitted GaussianHMM instance.
    X : np.ndarray, shape (T, n_features)
        Observation sequence.

    Returns
    -------
    posteriors : np.ndarray, shape (T, n_components)
        Row t is the posterior probability over states given observations
        up to and including time t.
    """
    log_frameprob = model._compute_log_likelihood(X)
    _, fwdlattice = _hmmc.forward_log(
        model.startprob_, model.transmat_, log_frameprob
    )
    # Normalise row-wise via log-sum-exp for numerical stability.
    log_norm = logsumexp(fwdlattice, axis=1, keepdims=True)
    posteriors = np.exp(fwdlattice - log_norm)
    return posteriors


def verify_no_lookahead(
    model: GaussianHMM,
    X: np.ndarray,
    n_samples: int = 20,
) -> bool:
    """
    Verify that forward_filter produces no look-ahead bias.

    Strategy: for n_samples values of t drawn from the middle half of
    [0, T), compare the streaming posterior at position t (computed on
    the full sequence with the same model) against the posterior
    produced by running forward_filter on the prefix X[:t+1].  Both
    use the SAME model parameters, so if the implementation is
    forward-only the outputs must be bit-for-bit identical (within
    floating-point tolerance 1e-6).

    The sorted-vector comparison is used so that state-label permutation
    differences (which cannot arise here since the model is the same, but
    are handled defensively) do not cause spurious failures.

    Parameters
    ----------
    model : GaussianHMM
        A fitted GaussianHMM instance.
    X : np.ndarray, shape (T, n_features)
        Observation sequence used for verification.
    n_samples : int
        Number of time indices to probe.

    Returns
    -------
    True if all probed positions pass; raises AssertionError otherwise.
    """
    T = len(X)
    if T < 8:
        raise ValueError(f"X too short for verification (T={T}, need >= 8)")

    # Compute streaming posteriors once over the full sequence.
    streaming_posteriors = forward_filter(model, X)

    # Sample indices from the middle half to avoid trivially short prefixes.
    lo = T // 4
    hi = 3 * T // 4
    rng = np.random.default_rng(seed=0)
    sampled_t = rng.choice(np.arange(lo, hi), size=min(n_samples, hi - lo), replace=False)
    sampled_t = np.sort(sampled_t)

    for t in sampled_t:
        # Forward filter on prefix only — proves row t is causal.
        prefix_posteriors = forward_filter(model, X[: t + 1])
        refit_at_t = prefix_posteriors[-1]
        streaming_at_t = streaming_posteriors[t]

        # Sort both vectors before comparing to be robust to state-index
        # permutations (not strictly necessary when model is the same, but
        # makes the check permutation-invariant and easier to reason about).
        diff = np.abs(np.sort(streaming_at_t) - np.sort(refit_at_t)).max()
        if diff >= 1e-6:
            raise AssertionError(
                f"Look-ahead bias detected at t={t}: "
                f"sorted streaming={np.sort(streaming_at_t)}, "
                f"sorted prefix={np.sort(refit_at_t)}, "
                f"max_diff={diff:.3e}"
            )

    return True


def _make_synthetic_ohlcv(n_bars: int = 300, seed: int = 42) -> np.ndarray:
    """
    Generate deterministic OHLCV-like data for self-check purposes.

    Produces 3 consecutive volatility regimes (low / mid / high) of
    equal length so that _engineer_features yields well-separated
    clusters, giving GaussianHMM a non-degenerate covariance matrix.

    Returns a numpy array with shape (n_bars, 5):
        [open, high, low, close, volume]
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
    current_price = 100.0

    for idx, (vol, (sp_lo, sp_hi)) in enumerate(zip(regime_vols, regime_spreads)):
        n = bars_per_regime + (remainder if idx == 2 else 0)
        log_ret = rng.normal(0.0, vol, size=n)
        close = current_price * np.exp(np.cumsum(log_ret))
        spread = rng.uniform(sp_lo, sp_hi, size=n)
        close_parts.append(close)
        high_parts.append(close * (1.0 + spread))
        low_parts.append(close * (1.0 - spread))
        current_price = float(close[-1])

    close = np.concatenate(close_parts)
    high = np.concatenate(high_parts)
    low = np.concatenate(low_parts)
    open_ = close * (1.0 + rng.normal(0.0, 0.003, size=n_bars))
    volume = rng.integers(100_000, 1_000_000, size=n_bars).astype(float)

    return np.column_stack([open_, high, low, close, volume])


def _run_import_check() -> None:
    """
    Self-check executed once when this module is imported.

    Fits a small HMM on synthetic data and calls verify_no_lookahead.
    Sets LOOKAHEAD_CHECK_PASSED accordingly.  Does NOT re-raise on
    failure so that dashboards that import hmm_utils remain loadable
    even if the check fails — they should surface LOOKAHEAD_CHECK_PASSED
    as an error badge instead.
    """
    global LOOKAHEAD_CHECK_PASSED  # noqa: PLW0603
    try:
        import pandas as pd

        ohlcv = _make_synthetic_ohlcv(n_bars=200, seed=42)
        df = pd.DataFrame(ohlcv, columns=["open", "high", "low", "close", "volume"])

        # Import feature engineering here to avoid circular import at module top.
        # hmm_utils imports verify, so we import hmm_utils helpers lazily.
        from core.hmm_utils import _engineer_features  # noqa: PLC0415

        X = _engineer_features(df)

        model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        model.fit(X)

        verify_no_lookahead(model, X, n_samples=20)
        LOOKAHEAD_CHECK_PASSED = True

    except AssertionError as exc:
        warnings.warn(
            f"[verify.py] Look-ahead bias check FAILED: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        LOOKAHEAD_CHECK_PASSED = False
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"[verify.py] Look-ahead bias check could not run: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        LOOKAHEAD_CHECK_PASSED = False


# Run the self-check on import.
_run_import_check()
