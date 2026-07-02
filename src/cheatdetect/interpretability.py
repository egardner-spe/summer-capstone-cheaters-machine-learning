"""Interpretability machinery (Week 6): SHAP with a split-role design.

The champion is an RBF-SVM, which only supports the slow, approximate
KernelExplainer. The Week-4 runner-up RF (same 'none' strategy, PR-AUC 0.426
vs 0.435) supports fast, EXACT TreeExplainer. So the roles are split:

  * RF + TreeExplainer  -> the detailed global story (rankings, beeswarm,
    per-error-group attributions), computed exactly on the whole test set;
  * SVM + KernelExplainer -> a checkpointed sample, used to test whether the
    two models RANK features the same way. If they agree, the RF narrative
    speaks for the champion too -- and the agreement is itself evidence that
    the feature signal, not model idiosyncrasy, drives detection.

The RF here is refit WITHOUT the scaler (trees are scale-invariant, and
unscaled inputs make SHAP value/colour axes physically interpretable);
equivalence with the piped version is asserted, not assumed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def refit_rf_unscaled(X_train, y_train, X_test, y_test, params: dict,
                      seed: int, pipeline_auc: float):
    """Best-config RF fit on raw (unscaled) features; verified equivalent."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    rf = RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                random_state=seed, **params)
    rf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
    if abs(auc - pipeline_auc) > 0.005:
        raise AssertionError(
            f"unscaled RF (AUC {auc:.4f}) is not equivalent to the piped "
            f"version (AUC {pipeline_auc:.4f}) -- scaler mattered?!")
    return rf, auc


def tree_shap(rf, X: pd.DataFrame) -> np.ndarray:
    """Exact SHAP values for the positive class. -> (n, n_features)."""
    import shap
    sv = shap.TreeExplainer(rf).shap_values(X.to_numpy())
    if isinstance(sv, list):          # older API: [class0, class1]
        return np.asarray(sv[1])
    if sv.ndim == 3:                  # newer API: (n, features, classes)
        return sv[:, :, 1]
    return sv


def mean_abs_ranking(shap_values: np.ndarray,
                     feature_names: list[str]) -> pd.Series:
    return pd.Series(np.abs(shap_values).mean(axis=0),
                     index=feature_names).sort_values(ascending=False)


def group_attribution(shap_values: np.ndarray, feature_names: list[str],
                      groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """Mean SIGNED SHAP per feature per group (what pushed each group's
    score, and in which direction)."""
    out = {}
    for g, mask in groups.items():
        out[f"{g} (n={int(mask.sum())})"] = shap_values[mask].mean(axis=0)
    return pd.DataFrame(out, index=feature_names).round(5)


def rank_agreement(rank_a: pd.Series, rank_b: pd.Series, top_k: int = 8):
    """Spearman rho over all features + overlap of the top-k sets."""
    from scipy.stats import spearmanr
    common = rank_a.index
    rho, p = spearmanr(rank_a[common].to_numpy(), rank_b[common].to_numpy())
    top_a, top_b = set(rank_a.head(top_k).index), set(rank_b.head(top_k).index)
    return {"spearman_rho": round(float(rho), 4),
            "spearman_p": float(p),
            f"top{top_k}_overlap": len(top_a & top_b),
            f"top{top_k}_a": sorted(top_a), f"top{top_k}_b": sorted(top_b)}
