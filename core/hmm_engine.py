"""Hidden Markov Model engine for market-regime detection.

This is the regime-detection brain. It wraps ``hmmlearn``'s Gaussian HMM but
adds the operational discipline a live trading system needs:

* **No look-ahead.** Online inference uses the *forward algorithm* one bar at a
  time (:meth:`HMMRegimeEngine.predict_online`). We never call
  ``model.predict()`` / ``model.decode()`` on a full sequence at inference time,
  because Viterbi/posterior decoding over a whole sequence peeks at future bars.
* **Honest model selection.** The number of regimes is chosen by BIC computed on
  a held-out validation slice (the last 20% of the training window), not on the
  data the model was fit to.
* **Train-only scaling.** ``StandardScaler`` is fit on training data only and
  saved with the model artifact, so the test period never leaks into scaling.
* **Stable, interpretable regimes.** After training, regimes are sorted by mean
  return (regime 0 = lowest = "crash", regime N = highest), labelled, and a
  stability filter suppresses signals from noisy regime flips.

Covariance type
---------------
We use ``covariance_type="diag"``. With up to 7 states over 5 features and only
~250 training bars, full covariances (105 params at n=7) overfit badly; diagonal
covariances (35 params) are the standard, stable choice for regime HMMs.
"""

import logging
from collections import deque

import joblib
import numpy as np
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.preprocessing import StandardScaler

from core.feature_engineering import FEATURE_COLUMNS
from settings.config import (
    HMM_N_REGIMES_RANGE,
    REGIME_STABILITY_MAX_FLIPS,
    REGIME_STABILITY_MIN_BARS,
)

logger = logging.getLogger(__name__)

# Fraction of the training window held out (at the end) for BIC model selection.
VALIDATION_FRACTION = 0.20

# Number of recent bars over which regime flips are counted for stability.
STABILITY_WINDOW = 20

# Confidence forced when the regime is flipping too often to be trusted.
UNSTABLE_CONFIDENCE = 0.3

# HMM fitting hyper-parameters.
HMM_N_ITER = 100
HMM_TOL = 1e-3
HMM_COVARIANCE_TYPE = "diag"
RANDOM_STATE = 42

# Canonical regime label spectrum, ordered from lowest to highest mean return.
# The 5-regime case is the spec's reference mapping; other counts in the
# configured 3..7 search range reuse the same low->high spectrum with unique,
# ordered names.
_LABELS_BY_N = {
    3: ["bear", "neutral", "bull"],
    4: ["crash", "bear", "bull", "euphoria"],
    5: ["crash", "bear", "neutral", "bull", "euphoria"],
    6: ["crash", "bear", "neutral", "bull", "euphoria", "mania"],
    7: ["capitulation", "crash", "bear", "neutral", "bull", "euphoria", "mania"],
}


def regime_labels(n_regimes: int) -> list[str]:
    """Return ``n_regimes`` unique labels ordered from lowest to highest return.

    Regime 0 is always the most bearish ("crash"-like) and the last is the most
    bullish ("euphoria"-like). The 5-regime mapping matches the spec exactly;
    other counts adjust the label set while preserving the ordering.
    """
    if n_regimes in _LABELS_BY_N:
        return list(_LABELS_BY_N[n_regimes])
    raise ValueError(
        f"No label mapping for n_regimes={n_regimes}; "
        f"supported counts are {sorted(_LABELS_BY_N)}."
    )


def _n_free_parameters(n_states: int, n_features: int) -> int:
    """Number of free parameters in a diagonal-covariance Gaussian HMM."""
    start = n_states - 1
    trans = n_states * (n_states - 1)
    means = n_states * n_features
    covars = n_states * n_features  # diagonal
    return start + trans + means + covars


class HMMRegimeEngine:
    """Trainable, online-inferring HMM regime detector.

    Typical lifecycle::

        engine = HMMRegimeEngine()
        engine.fit(train_features)         # selects n_regimes, scales, sorts
        engine.reset_online()              # clear forward state for a fresh run
        for _, bar in oos_features.iterrows():
            result = engine.predict_online(bar)   # one bar at a time
    """

    def __init__(self, n_regimes_range: tuple[int, int] = HMM_N_REGIMES_RANGE):
        self.n_regimes_range = n_regimes_range

        # Set by fit().
        self.model: GaussianHMM | None = None
        self.scaler: StandardScaler | None = None
        self.n_regimes: int | None = None
        self.feature_names_: list[str] | None = None
        self.labels: list[str] | None = None
        # Maps sorted rank -> original hmmlearn state index (ascending return).
        self._order: np.ndarray | None = None
        self.bic_scores: dict[int, float] = {}
        self.training_date_range: tuple | None = None

        # Online forward-algorithm state.
        self._log_alpha: np.ndarray | None = None
        self._regime_history: deque[int] = deque(maxlen=STABILITY_WINDOW)

    # ------------------------------------------------------------------ fit
    def fit(self, features) -> "HMMRegimeEngine":
        """Train the engine on a window of feature rows.

        ``features`` is a DataFrame of the columns produced by
        :func:`core.feature_engineering.compute_features`. Selects the regime
        count by held-out BIC, fits a train-only scaler, fits the final model on
        the full window, and sorts/labels regimes by mean return.
        """
        clean = features.dropna()
        self.feature_names_ = list(clean.columns)
        X = clean.to_numpy(dtype=float)
        n_features = X.shape[1]

        if len(clean):
            self.training_date_range = (clean.index[0], clean.index[-1])
            logger.info(
                "Training HMM on %d bars from %s to %s",
                len(clean),
                self.training_date_range[0],
                self.training_date_range[1],
            )

        # ---- Model selection: BIC on a held-out validation slice ----------
        split = int(len(X) * (1 - VALIDATION_FRACTION))
        X_sub, X_val = X[:split], X[split:]

        lo, hi = self.n_regimes_range
        self.bic_scores = {}
        for n in range(lo, hi + 1):
            # Scale on the sub-train only, never on the validation slice.
            try:
                sub_scaler = StandardScaler().fit(X_sub)
                model = self._new_model(n)
                model.fit(sub_scaler.transform(X_sub))
                val_loglik = model.score(sub_scaler.transform(X_val))
            except (ValueError, FloatingPointError) as exc:
                # hmmlearn can fail to converge on a candidate (degenerate /
                # NaN parameters). Disqualify it rather than crashing the fit.
                logger.warning("hmm_candidate_failed: n_regimes=%d (%s)", n, exc)
                self.bic_scores[n] = float("inf")
                continue
            if not np.isfinite(val_loglik):
                logger.warning("hmm_candidate_nonfinite_loglik: n_regimes=%d", n)
                self.bic_scores[n] = float("inf")
                continue
            k = _n_free_parameters(n, n_features)
            self.bic_scores[n] = float(-2.0 * val_loglik + k * np.log(len(X_val)))

        # Candidates ordered best (lowest) BIC first; skip the disqualified ones.
        ranked = sorted(self.bic_scores, key=self.bic_scores.get)
        viable = [n for n in ranked if np.isfinite(self.bic_scores[n])]
        if not viable:
            raise RuntimeError(
                "HMM failed to fit any candidate regime count "
                f"in range {self.n_regimes_range}; data may be degenerate."
            )
        logger.info(
            "BIC per n_regimes: %s",
            {n: round(b, 1) for n, b in self.bic_scores.items() if np.isfinite(b)},
        )

        # ---- Final fit on the full training window ------------------------
        # Try candidates in BIC order; the best may still fail on the full
        # window, so fall through to the next-best until one fits.
        self.scaler = StandardScaler().fit(X)
        X_scaled = self.scaler.transform(X)
        for n in viable:
            try:
                model = self._new_model(n)
                model.fit(X_scaled)
                model.score(X_scaled)  # validate params are usable
            except (ValueError, FloatingPointError) as exc:
                logger.warning("hmm_final_fit_failed: n_regimes=%d (%s)", n, exc)
                continue
            self.n_regimes = n
            self.model = model
            break
        else:
            raise RuntimeError(
                "HMM failed to fit on the full training window for any viable "
                f"candidate in {self.n_regimes_range}."
            )
        logger.info("selected n_regimes=%d", self.n_regimes)

        self._sort_regimes_by_return()
        self.labels = regime_labels(self.n_regimes)
        self.reset_online()
        return self

    def _new_model(self, n_states: int) -> GaussianHMM:
        return GaussianHMM(
            n_components=n_states,
            covariance_type=HMM_COVARIANCE_TYPE,
            n_iter=HMM_N_ITER,
            tol=HMM_TOL,
            random_state=RANDOM_STATE,
        )

    def _sort_regimes_by_return(self) -> None:
        """Compute the rank->state ordering by ascending mean log return.

        Each state's modelled mean log return is read from the fitted means and
        inverse-transformed back to raw units, so the ordering is well defined
        even for states that never appear in a Viterbi decode.
        """
        log_ret_idx = self.feature_names_.index("log_return")
        # Inverse-transform the scaled state means for the log_return feature.
        scaled_means = self.model.means_[:, log_ret_idx]
        raw_means = (
            scaled_means * self.scaler.scale_[log_ret_idx]
            + self.scaler.mean_[log_ret_idx]
        )
        # order[rank] = original state index, ascending by mean return.
        self._order = np.argsort(raw_means)

    # -------------------------------------------------------------- predict
    def reset_online(self) -> None:
        """Clear the forward-algorithm state and regime history."""
        self._log_alpha = None
        self._regime_history = deque(maxlen=STABILITY_WINDOW)

    def predict_online(self, bar_features) -> dict:
        """Process a single bar with the forward algorithm and return regime info.

        Returns a dict with:

        * ``current_regime``: ``str`` -- the regime label.
        * ``confidence``: ``float`` in [0, 1] -- filtered probability of the most
          likely state (forced to 0.3 when the regime is flipping too often).
        * ``regime_stable``: ``bool`` -- True once the current regime has
          persisted for at least ``REGIME_STABILITY_MIN_BARS`` bars.
        * ``raw_probs``: ``dict[str, float]`` -- filtered probability per regime.
        """
        if self.model is None:
            raise RuntimeError("Engine is not fitted; call fit() first.")

        x_scaled = self._scale_bar(bar_features)

        # Graceful NaN handling: a bar with missing/non-finite features carries
        # no information, so we skip it WITHOUT advancing the forward state or
        # regime history (which would corrupt the filter). We return the carried
        # posterior (or a uniform prior if none yet) with confidence 0 and
        # regime_stable False, so downstream code can hold rather than trade.
        if not np.all(np.isfinite(x_scaled)):
            logger.warning("predict_online received non-finite features; bar skipped")
            if self._log_alpha is not None:
                post_sorted = np.exp(self._log_alpha)[self._order]
            else:
                post_sorted = np.full(self.n_regimes, 1.0 / self.n_regimes)
            label = self.labels[int(np.argmax(post_sorted))]
            raw_probs = {
                self.labels[r]: float(post_sorted[r]) for r in range(self.n_regimes)
            }
            return {
                "current_regime": label,
                "confidence": 0.0,
                "regime_stable": False,
                "raw_probs": raw_probs,
            }

        emission_logprob = self._emission_logprob(x_scaled)

        with np.errstate(divide="ignore"):
            if self._log_alpha is None:
                log_alpha = np.log(self.model.startprob_) + emission_logprob
            else:
                log_trans = np.log(self.model.transmat_)
                # forward step: alpha_t[j] = e_j(x_t) * sum_i alpha_{t-1}[i] T[i,j]
                log_alpha = emission_logprob + logsumexp(
                    self._log_alpha[:, None] + log_trans, axis=0
                )

        # Normalize (scaled forward algorithm) to keep values bounded; this does
        # not change the filtered posterior.
        log_alpha = log_alpha - logsumexp(log_alpha)
        self._log_alpha = log_alpha

        posterior = np.exp(log_alpha)
        # Reorder original states into return-sorted ranks.
        post_sorted = posterior[self._order]

        rank = int(np.argmax(post_sorted))
        confidence = float(post_sorted[rank])
        label = self.labels[rank]
        raw_probs = {
            self.labels[r]: float(post_sorted[r]) for r in range(self.n_regimes)
        }

        # Update history and apply the stability filter.
        self._regime_history.append(rank)
        regime_stable = self._consecutive_count(rank) >= REGIME_STABILITY_MIN_BARS
        if self._count_flips() > REGIME_STABILITY_MAX_FLIPS:
            confidence = UNSTABLE_CONFIDENCE

        return {
            "current_regime": label,
            "confidence": confidence,
            "regime_stable": regime_stable,
            "raw_probs": raw_probs,
        }

    # --------------------------------------------------------- stability ops
    def _consecutive_count(self, rank: int) -> int:
        """How many bars (including now) the current regime has persisted."""
        count = 0
        for r in reversed(self._regime_history):
            if r == rank:
                count += 1
            else:
                break
        return count

    def _count_flips(self) -> int:
        """Number of regime switches within the stability window."""
        hist = list(self._regime_history)
        return sum(1 for a, b in zip(hist, hist[1:]) if a != b)

    # ------------------------------------------------------------- helpers
    def _scale_bar(self, bar_features) -> np.ndarray:
        """Order a single bar's features and apply the saved scaler."""
        if hasattr(bar_features, "reindex"):  # pd.Series
            vec = bar_features.reindex(self.feature_names_).to_numpy(dtype=float)
        elif isinstance(bar_features, dict):
            vec = np.array(
                [bar_features[name] for name in self.feature_names_], dtype=float
            )
        else:
            vec = np.asarray(bar_features, dtype=float).ravel()
            if vec.shape[0] != len(self.feature_names_):
                raise ValueError(
                    f"Expected {len(self.feature_names_)} features, "
                    f"got {vec.shape[0]}."
                )
        return self.scaler.transform(vec.reshape(1, -1))[0]

    def _emission_logprob(self, x_scaled: np.ndarray) -> np.ndarray:
        """Per-state Gaussian emission log-probability for one scaled bar."""
        logp = np.empty(self.n_regimes)
        for s in range(self.n_regimes):
            cov = np.asarray(self.model.covars_[s])
            if cov.ndim == 1:
                cov = np.diag(cov)
            elif cov.ndim == 0:
                cov = np.eye(len(x_scaled)) * cov
            logp[s] = multivariate_normal.logpdf(
                x_scaled, mean=self.model.means_[s], cov=cov, allow_singular=True
            )
        return logp

    # ---------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        """Persist the model, scaler, ordering, and metadata to ``path``."""
        if self.model is None:
            raise RuntimeError("Nothing to save; engine is not fitted.")
        artifact = {
            "model": self.model,
            "scaler": self.scaler,
            "n_regimes": self.n_regimes,
            "feature_names": self.feature_names_,
            "labels": self.labels,
            "order": self._order,
            "bic_scores": self.bic_scores,
            "training_date_range": self.training_date_range,
        }
        joblib.dump(artifact, path)

    @classmethod
    def load(cls, path: str) -> "HMMRegimeEngine":
        """Load an engine previously written by :meth:`save`."""
        artifact = joblib.load(path)
        engine = cls()
        engine.model = artifact["model"]
        engine.scaler = artifact["scaler"]
        engine.n_regimes = artifact["n_regimes"]
        engine.feature_names_ = artifact["feature_names"]
        engine.labels = artifact["labels"]
        engine._order = artifact["order"]
        engine.bic_scores = artifact["bic_scores"]
        engine.training_date_range = artifact["training_date_range"]
        engine.reset_online()
        return engine
