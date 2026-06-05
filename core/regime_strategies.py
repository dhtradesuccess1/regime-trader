"""Per-regime trading strategy definitions.

Maps each detected market regime to a target portfolio weight and turns an
online HMM prediction into an allocation signal. This is the transparent
long/flat default the system trades with: risk-off (flat) in bearish/stressed
regimes, fully invested in bull, reduced in neutral and overheated regimes.

The weight is the *desired* exposure; the risk manager still sizes and
limit-checks it before any order is placed. Signals from an unstable regime, or
below a minimum confidence, collapse to flat (0.0) to avoid trading noise.
"""

# Regime label -> target portfolio weight in [0, 1]. Mirrors the backtester's
# default so live and backtested behavior agree.
REGIME_TARGET_WEIGHTS = {
    "capitulation": 0.0,
    "crash": 0.0,
    "bear": 0.0,
    "neutral": 0.5,
    "bull": 1.0,
    "euphoria": 0.5,
    "mania": 0.0,
}

# Below this confidence the signal is treated as untradeable (go flat).
MIN_CONFIDENCE = 0.40


def get_target_weight(
    regime: str, confidence: float | None = None, regime_stable: bool = True
) -> float:
    """Return the target portfolio weight for a regime.

    Returns 0.0 (flat) when the regime is unstable or confidence is below
    ``MIN_CONFIDENCE``; otherwise the mapped weight (unknown regimes -> 0.0).
    """
    if not regime_stable:
        return 0.0
    if confidence is not None and confidence < MIN_CONFIDENCE:
        return 0.0
    return REGIME_TARGET_WEIGHTS.get(regime, 0.0)


def generate_signal(prediction: dict) -> dict:
    """Turn an HMM ``predict_online`` result into an allocation signal.

    Expects keys ``current_regime``, ``confidence``, ``regime_stable`` and
    returns those plus the computed ``target_weight``.
    """
    regime = prediction["current_regime"]
    confidence = prediction.get("confidence")
    regime_stable = prediction.get("regime_stable", True)
    return {
        "regime": regime,
        "confidence": confidence,
        "regime_stable": regime_stable,
        "target_weight": get_target_weight(regime, confidence, regime_stable),
    }
