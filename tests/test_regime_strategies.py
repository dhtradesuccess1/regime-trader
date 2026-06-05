"""Tests for ``core.regime_strategies``."""

from core.regime_strategies import (
    MIN_CONFIDENCE,
    REGIME_TARGET_WEIGHTS,
    generate_signal,
    get_target_weight,
)


def test_bull_full_bear_flat():
    assert get_target_weight("bull", confidence=0.9) == 1.0
    assert get_target_weight("bear", confidence=0.9) == 0.0
    assert get_target_weight("neutral", confidence=0.9) == 0.5


def test_unstable_regime_goes_flat():
    assert get_target_weight("bull", confidence=0.9, regime_stable=False) == 0.0


def test_low_confidence_goes_flat():
    assert get_target_weight("bull", confidence=MIN_CONFIDENCE - 0.01) == 0.0


def test_unknown_regime_flat():
    assert get_target_weight("???", confidence=0.9) == 0.0


def test_generate_signal_passthrough():
    pred = {"current_regime": "bull", "confidence": 0.8, "regime_stable": True}
    sig = generate_signal(pred)
    assert sig["regime"] == "bull"
    assert sig["confidence"] == 0.8
    assert sig["regime_stable"] is True
    assert sig["target_weight"] == REGIME_TARGET_WEIGHTS["bull"]
