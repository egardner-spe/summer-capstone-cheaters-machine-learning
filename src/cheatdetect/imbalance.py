"""Imbalance handling (Week 3): SMOTE + class weights, leakage-safe by design.

The dataset is ~5:1 legit:cheater. Two standard countermeasures, implemented as
ONE pipeline factory so Week 4 treats the strategy as a hyperparameter:

    none          scaler -> model                       (baseline)
    class_weight  scaler -> model(class_weight=balanced)
    smote         scaler -> SMOTE -> model
    smote_cw      scaler -> SMOTE -> model(class_weight=balanced)

Leakage safety: everything lives in an imblearn Pipeline, so inside any CV
loop the scaler is fit and SMOTE resamples ONLY on that fold's training
portion; validation folds and the frozen test set are never resampled and
never influence scaling. Synthetic samples can never cross into evaluation
data by construction.

Notes for Week 4 (recorded here so the write-up and code agree):
  * SMOTE at full parity makes class_weight="balanced" a near no-op (weights
    ~1 after resampling), so smote_cw mainly differs from smote when
    sampling_strategy < 1. Compare strategies empirically; don't stack blindly.
  * SMOTE runs after scaling because it interpolates between kNN neighbours --
    distances should be computed in the standardised space.
  * XGBoost has no class_weight; use scale_pos_weight(y_train) instead.
"""
from __future__ import annotations

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler

from . import config as C

STRATEGIES = ("none", "class_weight", "smote", "smote_cw")


def scale_pos_weight(y) -> float:
    """XGBoost's class-weight analogue: n_negative / n_positive."""
    y = np.asarray(y)
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def make_pipeline(estimator, strategy: str = "class_weight",
                  sampling_strategy: float | str = "auto",
                  seed: int = C.RANDOM_SEED) -> Pipeline:
    """scaler [-> SMOTE] -> estimator, as a single leakage-safe object.

    estimator is cloned; class_weight is set on the clone when the strategy
    asks for it (estimator must support it -- LogReg/SVM/RF do, XGBoost
    doesn't: pass scale_pos_weight yourself and use 'none'/'smote').
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, "
                         f"got {strategy!r}")
    est = clone(estimator)
    if strategy in ("class_weight", "smote_cw"):
        est.set_params(class_weight="balanced")

    steps = [("scale", StandardScaler())]
    if strategy in ("smote", "smote_cw"):
        steps.append(("smote", SMOTE(sampling_strategy=sampling_strategy,
                                     k_neighbors=C.SMOTE_K_NEIGHBORS,
                                     random_state=seed)))
    steps.append(("model", est))
    return Pipeline(steps)


def fold_resample_report(X, y, fold_ids, sampling_strategy="auto",
                         seed: int = C.RANDOM_SEED):
    """Per-fold class counts before/after in-fold SMOTE (QA for the write-up).

    Mimics exactly what the pipeline does inside CV: for each fold f, fit
    SMOTE on the OTHER folds (the training portion) and report counts. Also
    verifies synthetic samples stay within the real minority's feature range
    (SMOTE interpolates, so this must hold -- checked, not assumed).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    rows = []
    for f in sorted(np.unique(fold_ids)):
        tr = fold_ids != f
        Xs = StandardScaler().fit_transform(X[tr])
        sm = SMOTE(sampling_strategy=sampling_strategy,
                   k_neighbors=C.SMOTE_K_NEIGHBORS, random_state=seed)
        Xr, yr = sm.fit_resample(Xs, y[tr])

        n_min_real = int((y[tr] == 1).sum())
        synth = Xr[len(Xs):]                      # SMOTE appends synthetics
        real_min = Xs[y[tr] == 1]
        within = bool(len(synth) == 0 or
                      ((synth.min(0) >= real_min.min(0) - 1e-9).all() and
                       (synth.max(0) <= real_min.max(0) + 1e-9).all()))
        rows.append({
            "fold_held_out": int(f),
            "train_legit": int((y[tr] == 0).sum()),
            "train_cheater_real": n_min_real,
            "train_cheater_after_smote": int((yr == 1).sum()),
            "synthetic_added": int((yr == 1).sum()) - n_min_real,
            "synthetic_within_real_range": within,
        })
    return rows
