"""Regime -> portfolio exposure mapping.

Maps volatility regime labels to a target portfolio exposure fraction,
blended with a neutral 0.50 baseline according to the classifier's
confidence score.  All functions are pure and side-effect-free.
"""

_EXPOSURE_MAP: dict[str, float] = {
    "Low Vol":     0.95,
    "Medium Vol":  0.80,
    "High Vol":    0.60,
    "Extreme Vol": 0.30,
    "Hyper Vol":   0.0,
    "Ultra Vol":   0.0,
    "Uncertain":   0.50,
}


def target_exposure(regime: str, confidence: float) -> float:
    """Return portfolio exposure in [0, 1] for a given regime and confidence.

    At ``confidence=1.0`` the return value equals the regime's base exposure.
    At ``confidence=0.0`` the return value is 0.50 (regime-neutral).
    Values between 0 and 1 are interpolated linearly:

        exposure = base * confidence + 0.50 * (1 - confidence)

    ``confidence`` is clamped to [0, 1] before computation so out-of-range
    inputs degrade gracefully rather than producing nonsensical exposures.

    Args:
        regime:     Volatility regime label.  Must be one of the keys in
                    ``_EXPOSURE_MAP``: ``"Low Vol"``, ``"Medium Vol"``,
                    ``"High Vol"``, ``"Extreme Vol"``, ``"Hyper Vol"``,
                    ``"Ultra Vol"``, or ``"Uncertain"``.
        confidence: Classifier confidence in [0, 1].  Values outside this
                    range are clamped silently.

    Returns:
        Portfolio exposure fraction in [0, 1].

    Raises:
        ValueError: If *regime* is not a recognised label.

    Example:
        >>> target_exposure("Low Vol", 1.0)
        0.95
        >>> target_exposure("Low Vol", 0.0)
        0.5
        >>> target_exposure("Low Vol", 0.5)
        0.725
    """
    if regime not in _EXPOSURE_MAP:
        raise ValueError(
            f"Unknown regime {regime!r}. "
            f"Valid regimes: {sorted(_EXPOSURE_MAP)}"
        )

    # Clamp confidence defensively so callers with slightly out-of-range
    # values (e.g. floating-point drift yielding 1.0000000001) are handled.
    confidence = max(0.0, min(1.0, confidence))

    base: float = _EXPOSURE_MAP[regime]
    return base * confidence + 0.50 * (1.0 - confidence)
