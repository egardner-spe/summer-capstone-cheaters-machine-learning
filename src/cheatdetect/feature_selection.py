"""Conservative, train-only feature pruning (Week 3).

Two unsupervised rules, both computed on TRAINING rows only so the test set
contributes nothing to any modelling decision (labels are never used at all,
but train-only keeps the protocol airtight and easy to defend):

  1. near-zero variance  -- var(train) < config.NZV_THRESHOLD. A feature that
     never moves cannot discriminate and destabilises scaling.
  2. near-perfect correlation -- greedy pass over pairs with
     |Pearson r| > config.CORR_THRESHOLD on train. From each redundant pair,
     drop the member with the larger mean |r| against all remaining features
     (the more "generic" one), so the survivor carries the most independent
     information. Ties break by feature order (deterministic).

This is deliberately NOT supervised selection (no AUC filtering, no model-based
importance) -- that would either need to live inside the CV loop or leak. The
Week-2 EDA showed the signal is spread across many weak features, so we only
remove what is provably redundant and let the Week-4 models weigh the rest.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def near_zero_variance(train: pd.DataFrame,
                       threshold: float = C.NZV_THRESHOLD) -> list[str]:
    v = train.var(axis=0)
    return v[v < threshold].index.tolist()


def correlation_prune(train: pd.DataFrame,
                      threshold: float = C.CORR_THRESHOLD):
    """Greedy redundancy removal. Returns (dropped, log_rows)."""
    corr = train.corr().abs()
    np.fill_diagonal(corr.values, 0.0)
    dropped: list[str] = []
    log: list[dict] = []

    while True:
        live = [c for c in corr.columns if c not in dropped]
        sub = corr.loc[live, live]
        r_max = sub.to_numpy().max() if len(live) > 1 else 0.0
        if r_max <= threshold:
            break
        i, j = np.unravel_index(np.argmax(sub.to_numpy()), sub.shape)
        a, b = sub.index[i], sub.columns[j]
        # drop the member more correlated with everything else on average
        loser = a if sub.loc[a].mean() >= sub.loc[b].mean() else b
        keeper = b if loser == a else a
        dropped.append(loser)
        log.append({"dropped": loser, "kept": keeper,
                    "abs_r": round(float(r_max), 4),
                    "reason": f"|r|>{threshold} with {keeper}"})
    return dropped, log


def finalize_features(features: pd.DataFrame, split: pd.DataFrame):
    """Apply both rules on train rows. Returns (final_names, prune_log_df)."""
    feat_cols = [c for c in features.columns if c != "label"]
    train_rows = split.loc[split["split"] == "train", "row"]
    train = features.loc[train_rows, feat_cols]

    log: list[dict] = []
    nzv = near_zero_variance(train)
    for f in nzv:
        log.append({"dropped": f, "kept": "", "abs_r": np.nan,
                    "reason": f"near-zero variance (<{C.NZV_THRESHOLD:g})"})

    corr_dropped, corr_log = correlation_prune(train.drop(columns=nzv))
    log.extend(corr_log)

    dropped = set(nzv) | set(corr_dropped)
    final = [f for f in feat_cols if f not in dropped]   # original order
    return final, pd.DataFrame(log)
