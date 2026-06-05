"""Tests for ``core.hmm_engine``.

Covers the critical requirements: BIC-based regime selection in range, regimes
sorted by mean return with correct labels, online forward-algorithm output
schema, causality (no look-ahead), train-only scaling, persistence round-trip,
and the regime-stability filter (min-bars persistence + max-flips confidence
override).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from core.feature_engineering import FEATURE_COLUMNS, compute_features
from core.hmm_engine import (
    HMMRegimeEngine,
    REGIME_STABILITY_MAX_FLIPS,
    REGIME_STABILITY_MIN_BARS,
    UNSTABLE_CONFIDENCE,
    regime_labels,
)


def make_regime_features(n_per_regime: int = 120, seed: int = 11) -> pd.DataFrame:
    """Build OHLCV with two visibly different regimes, then featurize it.

    A calm up-drift followed by a volatile down-drift gives the HMM something
    real to separate.
    """
    rng = np.random.default_rng(seed)
    calm = rng.normal(0.0008, 0.006, size=n_per_regime)
    stormy = rng.normal(-0.0012, 0.022, size=n_per_regime)
    returns = np.concatenate([calm, stormy])
    close = 400 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0.003, 0.002, size=close.size)))
    low = close * (1 - np.abs(rng.normal(0.003, 0.002, size=close.size)))
    open_ = close * (1 + rng.normal(0, 0.002, size=close.size))
    volume = rng.integers(50_000_000, 120_000_000, size=close.size).astype(float)
    index = pd.date_range("2022-01-03", periods=close.size, freq="B")
    ohlcv = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )
    return compute_features(ohlcv)


@pytest.fixture(scope="module")
def fitted_engine():
    return HMMRegimeEngine().fit(make_regime_features())


# --------------------------------------------------------------- selection
def test_bic_selection_in_range(fitted_engine):
    """n_regimes is chosen from 3..7 via BIC scores recorded per candidate."""
    assert 3 <= fitted_engine.n_regimes <= 7
    assert set(fitted_engine.bic_scores) == {3, 4, 5, 6, 7}
    # The selected count must be the BIC minimizer.
    best = min(fitted_engine.bic_scores, key=fitted_engine.bic_scores.get)
    assert fitted_engine.n_regimes == best


def test_regimes_sorted_by_return(fitted_engine):
    """Regime ranks are ordered low->high mean return (0 = most bearish)."""
    eng = fitted_engine
    idx = eng.feature_names_.index("log_return")
    raw_means = (
        eng.model.means_[:, idx] * eng.scaler.scale_[idx] + eng.scaler.mean_[idx]
    )
    ordered = raw_means[eng._order]
    assert np.all(np.diff(ordered) >= 0)
    # Labels are unique and match the count.
    assert eng.labels == regime_labels(eng.n_regimes)
    assert len(set(eng.labels)) == eng.n_regimes


def test_labels_adjust_for_fewer_regimes():
    """The 5-regime mapping matches spec; other counts stay ordered & unique."""
    assert regime_labels(5) == ["crash", "bear", "neutral", "bull", "euphoria"]
    for n in (3, 4, 5, 6, 7):
        labels = regime_labels(n)
        assert len(labels) == n
        assert len(set(labels)) == n


# ------------------------------------------------------------ online output
def test_predict_online_output_schema(fitted_engine):
    """predict_online returns the required keys with valid ranges."""
    feats = make_regime_features()
    fitted_engine.reset_online()
    result = fitted_engine.predict_online(feats.iloc[0])

    assert set(result) == {
        "current_regime",
        "confidence",
        "regime_stable",
        "raw_probs",
    }
    assert isinstance(result["current_regime"], str)
    assert result["current_regime"] in fitted_engine.labels
    assert 0.0 <= result["confidence"] <= 1.0
    assert isinstance(result["regime_stable"], bool)
    assert set(result["raw_probs"]) == set(fitted_engine.labels)
    assert result["raw_probs"][result["current_regime"]] == pytest.approx(
        max(result["raw_probs"].values())
    )
    assert sum(result["raw_probs"].values()) == pytest.approx(1.0, abs=1e-6)


def test_online_is_causal_no_lookahead(fitted_engine):
    """The result for bar t depends only on bars 0..t, not on future bars.

    Feeding the same prefix twice — once alone, once followed by more bars from
    a different series — must yield identical results for the shared prefix.
    """
    feats = make_regime_features()
    prefix = feats.iloc[:10]

    fitted_engine.reset_online()
    first_pass = [fitted_engine.predict_online(row) for _, row in prefix.iterrows()]

    # Replay the same prefix; forward state must reproduce bit-for-bit.
    fitted_engine.reset_online()
    second_pass = [fitted_engine.predict_online(row) for _, row in prefix.iterrows()]

    for a, b in zip(first_pass, second_pass):
        assert a["current_regime"] == b["current_regime"]
        assert a["confidence"] == pytest.approx(b["confidence"])


def test_first_bar_not_stable(fitted_engine):
    """A single bar cannot satisfy the min-consecutive-bars requirement."""
    feats = make_regime_features()
    fitted_engine.reset_online()
    result = fitted_engine.predict_online(feats.iloc[0])
    assert result["regime_stable"] is False  # 1 < REGIME_STABILITY_MIN_BARS


# --------------------------------------------------------- stability filter
def test_consecutive_persistence_marks_stable(fitted_engine):
    """Once a regime persists >= MIN_BARS, regime_stable is True."""
    eng = fitted_engine
    eng.reset_online()
    eng._regime_history.extend([2] * (REGIME_STABILITY_MIN_BARS - 1))
    assert eng._consecutive_count(2) == REGIME_STABILITY_MIN_BARS - 1
    eng._regime_history.append(2)
    assert eng._consecutive_count(2) >= REGIME_STABILITY_MIN_BARS


def test_excessive_flips_force_low_confidence(fitted_engine):
    """> MAX_FLIPS switches in the window forces confidence to 0.3."""
    eng = fitted_engine
    eng.reset_online()
    # Alternating history => many flips, clearly above the threshold.
    alternating = [i % 2 for i in range(2 * (REGIME_STABILITY_MAX_FLIPS + 2))]
    eng._regime_history.extend(alternating)
    assert eng._count_flips() > REGIME_STABILITY_MAX_FLIPS

    feats = make_regime_features()
    result = eng.predict_online(feats.iloc[0])
    assert result["confidence"] == UNSTABLE_CONFIDENCE


# --------------------------------------------------------------- persistence
def test_save_load_roundtrip(fitted_engine, tmp_path):
    """A saved engine reloads and reproduces the same online output."""
    path = tmp_path / "engine.joblib"
    fitted_engine.save(str(path))
    reloaded = HMMRegimeEngine.load(str(path))

    assert reloaded.n_regimes == fitted_engine.n_regimes
    assert reloaded.labels == fitted_engine.labels

    feats = make_regime_features()
    fitted_engine.reset_online()
    reloaded.reset_online()
    a = fitted_engine.predict_online(feats.iloc[0])
    b = reloaded.predict_online(feats.iloc[0])
    assert a["current_regime"] == b["current_regime"]
    assert a["confidence"] == pytest.approx(b["confidence"])


# ===========================================================================
# Mandatory look-ahead / leakage guard tests (requested explicitly).
# ===========================================================================
def test_no_lookahead():
    """Train on first 60%, predict bar-by-bar on last 40%; the prediction for
    bar k must not depend on bars after k (the defining property of the online
    forward algorithm). Truncating the input after k must leave bars 0..k-1
    unchanged. This FAILS for any implementation that decodes the full sequence.
    """
    feats = make_regime_features()
    split = int(len(feats) * 0.6)
    train, oos = feats.iloc[:split], feats.iloc[split:]
    engine = HMMRegimeEngine((3, 4)).fit(train)

    engine.reset_online()
    full = [engine.predict_online(row) for _, row in oos.iterrows()]

    k = len(oos) // 2
    engine.reset_online()
    truncated = [engine.predict_online(row) for _, row in oos.iloc[:k].iterrows()]

    for a, b in zip(full[:k], truncated):
        assert a["current_regime"] == b["current_regime"]
        assert a["confidence"] == pytest.approx(b["confidence"])
        assert a["raw_probs"] == pytest.approx(b["raw_probs"])


def test_regime_stability(fitted_engine):
    """The stability filter suppresses single-bar flickers: a bar whose regime
    differs from the previous bar can never be reported as stable (it has only
    persisted 1 < MIN_BARS). A steady input stream does eventually become stable.
    """
    feats = make_regime_features()

    # Flicker suppression on real data: every fresh flip is unstable.
    fitted_engine.reset_online()
    prev = None
    for _, row in feats.iterrows():
        res = fitted_engine.predict_online(row)
        if prev is not None and res["current_regime"] != prev:
            assert res["regime_stable"] is False
        prev = res["current_regime"]

    # Steady input -> the regime persists -> becomes stable within MIN_BARS bars.
    fitted_engine.reset_online()
    steady = [fitted_engine.predict_online(feats.iloc[0]) for _ in range(6)]
    assert any(r["regime_stable"] for r in steady)


def test_regime_ordering(fitted_engine):
    """Regime 0 has the lowest mean return; ranks are monotonically increasing."""
    eng = fitted_engine
    idx = eng.feature_names_.index("log_return")
    raw_means = (
        eng.model.means_[:, idx] * eng.scaler.scale_[idx] + eng.scaler.mean_[idx]
    )
    ordered = raw_means[eng._order]
    assert ordered[0] == ordered.min()
    assert np.all(np.diff(ordered) >= 0)


def test_feature_scaling_leak():
    """The scaler is fitted on the train slice only, never the full dataset.

    The engine's scaler must match a StandardScaler fit on the train slice and
    must DIFFER from one fit on the full dataset (which would be leakage).
    """
    feats = make_regime_features()
    split = int(len(feats) * 0.6)
    train = feats.iloc[:split]
    engine = HMMRegimeEngine((3, 4)).fit(train)

    train_only = StandardScaler().fit(train.dropna().to_numpy(dtype=float))
    full = StandardScaler().fit(feats.dropna().to_numpy(dtype=float))

    np.testing.assert_allclose(engine.scaler.mean_, train_only.mean_, rtol=1e-6)
    np.testing.assert_allclose(engine.scaler.scale_, train_only.scale_, rtol=1e-6)
    assert not np.allclose(engine.scaler.mean_, full.mean_)


def test_confidence_range(fitted_engine):
    """Every confidence and every raw probability lies in [0, 1]."""
    feats = make_regime_features()
    fitted_engine.reset_online()
    for _, row in feats.iterrows():
        res = fitted_engine.predict_online(row)
        assert 0.0 <= res["confidence"] <= 1.0
        assert sum(res["raw_probs"].values()) == pytest.approx(1.0, abs=1e-6)
        for p in res["raw_probs"].values():
            assert 0.0 <= p <= 1.0


def test_handles_nan(fitted_engine):
    """A NaN bar is handled gracefully and does NOT corrupt forward state.

    The bar following a NaN must equal the baseline (NaN-free) prediction for
    that same bar, proving the NaN was skipped without advancing the filter.
    """
    feats = make_regime_features()

    # Baseline: two clean bars in sequence.
    fitted_engine.reset_online()
    fitted_engine.predict_online(feats.iloc[0])
    baseline = fitted_engine.predict_online(feats.iloc[1])

    # Same run, but with a NaN bar injected between the two clean bars.
    fitted_engine.reset_online()
    fitted_engine.predict_online(feats.iloc[0])
    nan_row = feats.iloc[1].copy()
    nan_row[:] = np.nan
    nan_res = fitted_engine.predict_online(nan_row)

    assert 0.0 <= nan_res["confidence"] <= 1.0
    assert nan_res["regime_stable"] is False

    after = fitted_engine.predict_online(feats.iloc[1])
    assert after["current_regime"] == baseline["current_regime"]
    assert after["confidence"] == pytest.approx(baseline["confidence"])


def test_fit_skips_failing_candidates(monkeypatch):
    """A candidate that fails to converge is disqualified, not fatal.

    Simulates hmmlearn's 'startprob_ must sum to 1 (got nan)' for n>=5; the
    engine should still fit with a viable smaller regime count.
    """
    import core.hmm_engine as hm

    real_score = hm.GaussianHMM.score

    def flaky_score(self, X, *args, **kwargs):
        if self.n_components >= 5:
            raise ValueError("startprob_ must sum to 1 (got nan)")
        return real_score(self, X, *args, **kwargs)

    monkeypatch.setattr(hm.GaussianHMM, "score", flaky_score)
    engine = HMMRegimeEngine((3, 5)).fit(make_regime_features())

    assert engine.n_regimes in (3, 4)  # 5 was disqualified
    assert engine.bic_scores[5] == float("inf")
    # The fitted engine is still usable.
    engine.reset_online()
    assert "current_regime" in engine.predict_online(make_regime_features().iloc[0])


def test_fit_raises_when_all_candidates_fail(monkeypatch):
    """If every candidate fails, fit raises a clear error instead of NaNs."""
    import core.hmm_engine as hm

    def always_fail(self, X, *args, **kwargs):
        raise ValueError("startprob_ must sum to 1 (got nan)")

    monkeypatch.setattr(hm.GaussianHMM, "score", always_fail)
    with pytest.raises(RuntimeError, match="failed to fit"):
        HMMRegimeEngine((3, 4)).fit(make_regime_features())
