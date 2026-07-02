"""Error-analysis machinery (Week 5): blatancy index, group profiles, calibration.

The core object is a **mechanical-extremity index** ("blatancy index" when read
on cheaters, "mechanical-skill index" when read on legit players): for each
instance, the average percentile rank of its value -- relative to the LEGIT
TRAINING distribution -- on the eight shot-centric features Week 2 identified
as the discriminative core, each taken in the *cheater-ward* direction.

    index ~ 0.5  : mechanically typical of the legit population
    index -> 1.0 : extreme aimbot-like mechanics on every axis
                   (cheater: blatant; legit: snappy, highly skilled aim)

Design properties, chosen deliberately:
  * anchored to the legit TRAIN distribution only (test never contributes
    statistics; test instances are ranked against train ECDFs);
  * defined by Week-2 EDA directions, NOT by the champion's score -- so
    "does detection rate rise with blatancy?" is a real question, not a
    tautology (the residual caveat: the champion consumes these features
    among its 34, so independence is partial, and the write-up says so);
  * one index serves both analyses: recall-vs-blatancy on cheaters, and
    FP-concentration-vs-skill on legit players. That symmetry IS the
    project's central tension: the more a legit player's mechanics resemble
    an aimbot's, the more the detector is structurally tempted to flag them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (feature, cheater-ward direction) -- signs from reports/eda_findings.md.
# +1: cheaters sit HIGHER than legit; -1: cheaters sit LOWER.
INDEX_FEATURES = [
    ("frac_shots_locked", +1),    # fire with crosshair already still
    ("frac_shots_medspeed", -1),  # fewer mid-speed correction shots
    ("zcr_dpitch", +1),           # vertical micro-oscillation
    ("speed_at_shot_mean", -1),   # slower aim at the trigger
    ("speed_at_shot_std", -1),    # less variable shot-time speed
    ("speed_pre_shot_mean", -1),  # less last-moment correction
    ("settle_ratio", -1),         # already settled into the shot
    ("speed_max", -1),            # smaller peak flicks
]


class ExtremityIndex:
    """Percentile-rank index anchored to the legit training distribution."""

    def __init__(self, features=None):
        self.features = features or INDEX_FEATURES
        self._sorted: dict[str, np.ndarray] = {}

    def fit(self, legit_train: pd.DataFrame):
        """legit_train: feature rows of LEGIT TRAINING instances only."""
        for name, _ in self.features:
            self._sorted[name] = np.sort(legit_train[name].to_numpy())
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Mean cheater-ward percentile rank in [0, 1] per row of X."""
        parts = []
        for name, sign in self.features:
            ref = self._sorted[name]
            # mid-rank ECDF: average of left/right ranks, robust to ties
            lo = np.searchsorted(ref, X[name].to_numpy(), side="left")
            hi = np.searchsorted(ref, X[name].to_numpy(), side="right")
            pct = (lo + hi) / (2.0 * len(ref))
            parts.append(pct if sign > 0 else 1.0 - pct)
        return np.mean(parts, axis=0)


def rate_by_bin(index: np.ndarray, flagged: np.ndarray, edges: np.ndarray,
                labels=None) -> pd.DataFrame:
    """Flag rate per index bin (recall if rows are cheaters, FPR if legit)."""
    bins = np.digitize(index, edges[1:-1])   # 0..len(edges)-2
    rows = []
    for b in range(len(edges) - 1):
        m = bins == b
        rows.append({
            "bin": labels[b] if labels else b,
            "lo": round(float(edges[b]), 4),
            "hi": round(float(edges[b + 1]), 4),
            "n": int(m.sum()),
            "n_flagged": int(flagged[m].sum()),
            "rate": round(float(flagged[m].mean()) if m.any() else np.nan, 4),
        })
    return pd.DataFrame(rows)


def profile_groups(X: pd.DataFrame, groups: dict[str, np.ndarray],
                   features: list[str]) -> pd.DataFrame:
    """Mean of each feature per named boolean mask -- FN/TP/FP profile table."""
    out = {}
    for gname, mask in groups.items():
        sub = X.loc[mask, features]
        out[f"{gname} (n={int(mask.sum())})"] = sub.mean()
    return pd.DataFrame(out).round(4)


# --- calibration --------------------------------------------------------------

class Calibrator:
    """Picklable margin -> P(cheater) mapper (Platt sigmoid or isotonic)."""

    def __init__(self, method: str = "platt"):
        if method not in ("platt", "isotonic"):
            raise ValueError(f"unknown method {method!r}")
        self.method = method
        self.model = None

    def fit(self, scores: np.ndarray, y: np.ndarray):
        """Fit on OUT-OF-FOLD scores only."""
        if self.method == "platt":
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(C=1e6, max_iter=5000)  # plain sigmoid
            self.model.fit(np.asarray(scores).reshape(-1, 1), y)
        else:
            from sklearn.isotonic import IsotonicRegression
            self.model = IsotonicRegression(out_of_bounds="clip",
                                            y_min=0.0, y_max=1.0)
            self.model.fit(scores, y)
        return self

    def __call__(self, s) -> np.ndarray:
        s = np.asarray(s, dtype=float)
        if self.method == "platt":
            return self.model.predict_proba(s.reshape(-1, 1))[:, 1]
        return self.model.predict(s)


def fit_calibrator(scores: np.ndarray, y: np.ndarray, method: str = "platt"):
    """Convenience: fitted Calibrator (picklable, callable)."""
    return Calibrator(method).fit(scores, y)


def cv_brier(scores: np.ndarray, y: np.ndarray, folds: np.ndarray,
             method: str) -> float:
    """Fold-honest Brier score: calibrator fit on 4 folds, scored on the 5th,
    using the SAME persisted folds as everything else."""
    from sklearn.metrics import brier_score_loss
    briers = []
    for f in sorted(np.unique(folds)):
        val = folds == f
        cal = fit_calibrator(scores[~val], y[~val], method)
        briers.append(brier_score_loss(y[val], cal(scores[val])))
    return float(np.mean(briers))
