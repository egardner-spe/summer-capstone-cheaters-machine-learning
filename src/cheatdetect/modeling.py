"""Model zoo for Week 4: candidates x imbalance strategies, one build function.

Four candidate families, per the implementation plan:

    logreg  L2 logistic regression -- the linear baseline the Week-2/3 smoke
            tests already anchor (~0.71 ROC-AUC); anything below this is broken.
    rf      random forest -- nonlinear interactions, robust to feature scale.
    xgb     XGBoost -- gradient boosting, usually the strongest tabular model.
    svm     RBF SVM -- the plan's kernel method; scored via decision_function
            (probability=False: Platt scaling would 5x the fit cost for no
            benefit, since thresholds are chosen on score scale anyway).

Strategy wiring reuses cheatdetect.imbalance.make_pipeline. One special case:
XGBoost has no `class_weight` parameter, so `class_weight`/`smote_cw`
strategies are translated to `scale_pos_weight`, matching sklearn's
"balanced" semantics (weights computed from the data the model actually sees):

    xgb + class_weight -> scale_pos_weight = n_neg/n_pos of the training fold
                          (~5.0; constant across stratified folds)
    xgb + smote_cw     -> SMOTE first brings parity, so balanced weights are
                          ~1.0 => IDENTICAL to plain `smote`. This config is
                          excluded from the run matrix rather than duplicated;
                          the comparison table says why.

All estimators are deterministic given config.RANDOM_SEED.
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from . import config as C
from . import imbalance as I

MODEL_NAMES = ("logreg", "rf", "xgb", "svm")

# 5.004 on the full training set; stratification keeps folds within +/-0.01,
# so a fixed value is equivalent to per-fold computation (verified in QA).
XGB_SCALE_POS_WEIGHT = 5.004


def make_estimator(name: str, seed: int = C.RANDOM_SEED, **overrides):
    """Default-configured estimator; overrides are tuning-grid parameters."""
    if name == "logreg":
        est = LogisticRegression(max_iter=5000)
    elif name == "rf":
        est = RandomForestClassifier(
            n_estimators=400, n_jobs=-1, random_state=seed)
    elif name == "xgb":
        est = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, tree_method="hist",
            eval_metric="logloss", n_jobs=-1, random_state=seed)
    elif name == "svm":
        est = SVC(kernel="rbf", C=1.0, gamma="scale", cache_size=1000,
                  probability=False, random_state=seed)
    else:
        raise ValueError(f"unknown model {name!r}")
    if overrides:
        est.set_params(**overrides)
    return est


def build_pipeline(model: str, strategy: str, seed: int = C.RANDOM_SEED,
                   **overrides):
    """(model, strategy) -> leakage-safe pipeline, handling the xgb cases."""
    est = make_estimator(model, seed=seed, **overrides)
    if model == "xgb":
        if strategy == "class_weight":
            est.set_params(scale_pos_weight=XGB_SCALE_POS_WEIGHT)
            return I.make_pipeline(est, strategy="none", seed=seed)
        if strategy == "smote_cw":
            raise ValueError("xgb+smote_cw is identical to xgb+smote at "
                             "SMOTE parity (balanced weight ~1); excluded")
        return I.make_pipeline(est, strategy=strategy, seed=seed)
    return I.make_pipeline(est, strategy=strategy, seed=seed)


def run_matrix():
    """The stage-1 comparison grid: 15 configs (xgb+smote_cw excluded)."""
    out = []
    for m in MODEL_NAMES:
        for s in I.STRATEGIES:
            if m == "xgb" and s == "smote_cw":
                continue
            out.append((m, s))
    return out


# --- stage-2 tuning grids (deliberately light: compare-then-tune design) -----
TUNING_GRIDS = {
    "logreg": [{"C": c} for c in (0.1, 1.0, 10.0)],
    "rf": [{"max_depth": d, "min_samples_leaf": l}
           for d in (None, 12, 20) for l in (1, 3, 5)],
    "xgb": [{"max_depth": d, "learning_rate": lr, "n_estimators": n}
            for d in (3, 5, 7) for lr in (0.05, 0.08) for n in (400, 800)],
    "svm": [{"C": c} for c in (0.5, 2.0, 8.0)],
}
