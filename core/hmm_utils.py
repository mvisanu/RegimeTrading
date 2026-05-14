"""
hmm_utils.py — HMM-based market regime detection.

Uses forward filtering only (no Viterbi, no backward pass) to ensure
zero look-ahead bias.  All functions that touch the HMM use
core.verify.forward_filter exclusively.

Public surface
--------------
fit_and_filter(df)      -> RegimeResult
apply_stability_filter  -> list[str]
RegimeResult            dataclass
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    """Container for HMM regime detection outputs."""

    model: GaussianHMM
    """Fitted GaussianHMM instance."""

    posteriors: np.ndarray
    """Forward-filter posteriors, shape (T, n_components)."""

    labels: list[str]
    """Raw regime label per bar (pre-stability filter)."""

    stable_labels: list[str]
    """Post-stability-filter labels."""

    confidence: np.ndarray
    """Max posterior probability per bar, shape (T,)."""

    n_regimes: int
    """Number of HMM states selected."""

    label_map: dict[int, str]
    """Mapping from HMM state index to human-readable regime name."""

    feature_array: np.ndarray
    """The X array passed to the model (for verify.py)."""


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

_REGIME_LABELS = ["Low Vol", "Medium Vol", "High Vol", "Extreme Vol",
                  "Hyper Vol", "Ultra Vol"]


def _engineer_features(df: pd.DataFrame) -> np.ndarray:
    """
    Compute log-return, realised volatility, and high-low range from OHLCV.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: close, high, low.

    Returns
    -------
    X : np.ndarray, shape (T, 3)
        Columns: [log_return, realized_vol, hl_range_pct].
        All NaN rows have been dropped.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]

    log_return = np.log(close / close.shift(1))
    realized_vol = log_return.rolling(20).std()
    hl_range_pct = (high - low) / close

    combined = pd.DataFrame(
        {
            "log_return": log_return,
            "realized_vol": realized_vol,
            "hl_range_pct": hl_range_pct,
        }
    ).dropna()

    return combined.to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# BIC model selection
# ---------------------------------------------------------------------------

def _n_params_full_cov(n: int, n_features: int = 3) -> int:
    """
    Number of free parameters in a full-covariance GaussianHMM.

    Transition matrix (rows sum to 1): n*(n-1)
    Means:                             n * n_features
    Full covariance (symmetric):       n * n_features*(n_features+1)//2
    """
    transition = n * (n - 1)
    means = n * n_features
    cov = n * (n_features * (n_features + 1) // 2)
    return transition + means + cov


def _select_n_components(
    X: np.ndarray,
    min_n: int = 3,
    max_n: int = 6,
) -> tuple[GaussianHMM, int]:
    """
    Select optimal number of HMM states via BIC.

    BIC = -2 * log_likelihood + n_params * log(T)

    Parameters
    ----------
    X : np.ndarray
        Observation array of shape (T, n_features).
    min_n, max_n : int
        Range of state counts to evaluate (inclusive).

    Returns
    -------
    (best_model, best_n)
    """
    T = len(X)
    best_bic = np.inf
    best_model: GaussianHMM | None = None
    best_n = min_n

    for n in range(min_n, max_n + 1):
        model = GaussianHMM(
            n_components=n,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        model.fit(X)
        log_likelihood = model.score(X)  # total log-likelihood (sum over T samples)
        n_params = _n_params_full_cov(n, n_features=X.shape[1])
        bic = -2.0 * log_likelihood + n_params * np.log(T)
        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_n = n

    assert best_model is not None
    return best_model, best_n


# ---------------------------------------------------------------------------
# Regime labeling
# ---------------------------------------------------------------------------

def _label_regimes(
    model: GaussianHMM,
    X: np.ndarray,
    n: int,
) -> dict[int, str]:
    """
    Assign human-readable volatility regime labels to HMM states.

    States are ranked by their mean realised volatility (column 1 of X)
    and labelled from "Low Vol" upward.

    Parameters
    ----------
    model : GaussianHMM
        Fitted model.
    X : np.ndarray
        Feature array; column 1 is realised volatility.
    n : int
        Number of states.

    Returns
    -------
    dict mapping state index -> regime label.
    """
    from core.verify import forward_filter  # noqa: PLC0415

    posteriors = forward_filter(model, X)
    assignments = np.argmax(posteriors, axis=1)  # shape (T,)

    # Mean realised vol per state.
    mean_vol: dict[int, float] = {}
    for state in range(n):
        mask = assignments == state
        if mask.any():
            mean_vol[state] = X[mask, 1].mean()
        else:
            # Unvisited state: fall back to model mean for feature 1.
            mean_vol[state] = float(model.means_[state, 1])

    # Sort states by ascending mean volatility.
    sorted_states = sorted(mean_vol.keys(), key=lambda s: mean_vol[s])
    label_map: dict[int, str] = {}
    for rank, state in enumerate(sorted_states):
        label = _REGIME_LABELS[rank] if rank < len(_REGIME_LABELS) else f"Vol {rank}"
        label_map[state] = label

    return label_map


# ---------------------------------------------------------------------------
# Stability filter
# ---------------------------------------------------------------------------

def apply_stability_filter(labels: list[str]) -> list[str]:
    """
    Apply a two-stage stability filter to raw regime labels.

    Stage 1 — Minimum run length:
        A regime is only "active" after 3 consecutive identical bars.
        Before 3 consecutive occurrences, carry forward the previous
        stable regime (or "Uncertain" at the start of the series).

    Stage 2 — Chop filter:
        If the regime changes more than 4 times in any rolling 20-bar
        window, override those bars to "Uncertain".

    Parameters
    ----------
    labels : list[str]
        Raw regime label per bar.

    Returns
    -------
    list[str]
        Filtered labels, same length as input.
    """
    T = len(labels)
    stable: list[str] = ["Uncertain"] * T

    # --- Stage 1: minimum run of 3 ---
    current_stable = "Uncertain"
    run_label = labels[0] if T > 0 else "Uncertain"
    run_len = 0

    for i, lbl in enumerate(labels):
        if lbl == run_label:
            run_len += 1
        else:
            run_label = lbl
            run_len = 1

        if run_len >= 3:
            current_stable = run_label

        stable[i] = current_stable

    # --- Stage 2: chop filter (rolling 20-bar window, max 4 transitions) ---
    WINDOW = 20
    MAX_TRANSITIONS = 4

    # Snapshot Stage 1 output so Stage 2 reads consistent values even as it
    # overwrites the live list.  Without this, earlier "Uncertain" writes
    # reduce apparent transition counts in later windows, letting trailing
    # bars escape the chop filter.
    stage1_snapshot = stable[:]

    for i in range(T):
        window_start = max(0, i - WINDOW + 1)
        window = stage1_snapshot[window_start : i + 1]  # read from snapshot
        transitions = sum(
            1 for j in range(1, len(window)) if window[j] != window[j - 1]
        )
        if transitions > MAX_TRANSITIONS:
            for j in range(window_start, i + 1):
                stable[j] = "Uncertain"  # write to live list

    return stable


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fit_and_filter(
    df: pd.DataFrame,
    n_override: int | None = None,
) -> RegimeResult:
    """
    Full pipeline: feature engineering → model selection → forward filter
    → regime labeling → stability filter.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV data; must contain columns close, high, low.
    n_override : int | None
        If provided, skip BIC selection and use this many states.

    Returns
    -------
    RegimeResult
    """
    from core.verify import forward_filter  # noqa: PLC0415

    X = _engineer_features(df)

    if n_override is not None:
        model = GaussianHMM(
            n_components=n_override,
            covariance_type="full",
            n_iter=200,
            random_state=42,
        )
        model.fit(X)
        n = n_override
    else:
        model, n = _select_n_components(X)

    posteriors = forward_filter(model, X)
    confidence = posteriors.max(axis=1)

    label_map = _label_regimes(model, X, n)
    assignments = np.argmax(posteriors, axis=1)
    labels = [label_map[int(s)] for s in assignments]
    stable_labels = apply_stability_filter(labels)

    return RegimeResult(
        model=model,
        posteriors=posteriors,
        labels=labels,
        stable_labels=stable_labels,
        confidence=confidence,
        n_regimes=n,
        label_map=label_map,
        feature_array=X,
    )


# ---------------------------------------------------------------------------
# Trigger look-ahead bias check on import (spec requirement)
# ---------------------------------------------------------------------------
from core import verify as _verify_module  # noqa: E402, F401
