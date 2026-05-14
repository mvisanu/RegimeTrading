"""Tests for core.allocation.target_exposure."""

import pytest

from core.allocation import target_exposure


# ---------------------------------------------------------------------------
# Parametrized happy-path cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("regime,confidence,expected", [
    # Full-confidence → base exposure returned unchanged
    ("Low Vol",     1.0, 0.95),
    ("Medium Vol",  1.0, 0.80),
    ("High Vol",    1.0, 0.60),
    ("Extreme Vol", 1.0, 0.30),
    ("Uncertain",   1.0, 0.50),
    # Zero confidence → neutral 0.50 for any regime
    ("Low Vol",     0.0, 0.50),
    ("Extreme Vol", 0.0, 0.50),
    # Mid-confidence interpolation: 0.95*0.5 + 0.5*0.5 = 0.475 + 0.25 = 0.725
    ("Low Vol",     0.5, 0.725),
])
def test_target_exposure(regime: str, confidence: float, expected: float) -> None:
    assert abs(target_exposure(regime, confidence) - expected) < 1e-9


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unknown_regime_raises_value_error() -> None:
    """An unrecognised regime string must raise ValueError (not KeyError)."""
    with pytest.raises(ValueError, match="Unknown regime"):
        target_exposure("Unknown Regime", 1.0)


def test_unknown_regime_error_message_contains_regime_name() -> None:
    """The ValueError message should include the bad value for easy debugging."""
    with pytest.raises(ValueError, match="'Bogus'"):
        target_exposure("Bogus", 0.8)


# ---------------------------------------------------------------------------
# Confidence clamping
# ---------------------------------------------------------------------------

def test_confidence_above_one_clamped_to_one() -> None:
    """confidence > 1.0 should produce the same result as confidence = 1.0."""
    assert target_exposure("Low Vol", 1.5) == pytest.approx(
        target_exposure("Low Vol", 1.0), abs=1e-9
    )


def test_confidence_well_above_one_clamped() -> None:
    """Large confidence values should also clamp to 1.0."""
    assert target_exposure("High Vol", 999.0) == pytest.approx(
        target_exposure("High Vol", 1.0), abs=1e-9
    )


def test_confidence_below_zero_clamped_to_zero() -> None:
    """confidence < 0.0 should produce the same result as confidence = 0.0."""
    assert target_exposure("Extreme Vol", -0.5) == pytest.approx(
        target_exposure("Extreme Vol", 0.0), abs=1e-9
    )


def test_confidence_well_below_zero_clamped() -> None:
    """Large negative confidence values should also clamp to 0.0."""
    assert target_exposure("Medium Vol", -999.0) == pytest.approx(
        target_exposure("Medium Vol", 0.0), abs=1e-9
    )


# ---------------------------------------------------------------------------
# Return-value bounds
# ---------------------------------------------------------------------------

def test_exposure_always_in_unit_interval() -> None:
    """Exposure must stay in [0, 1] across all valid regimes and confidences."""
    regimes = ["Low Vol", "Medium Vol", "High Vol", "Extreme Vol", "Uncertain"]
    confidences = [0.0, 0.25, 0.5, 0.75, 1.0]
    for regime in regimes:
        for conf in confidences:
            result = target_exposure(regime, conf)
            assert 0.0 <= result <= 1.0, (
                f"Out-of-range exposure {result} for regime={regime!r}, "
                f"confidence={conf}"
            )
