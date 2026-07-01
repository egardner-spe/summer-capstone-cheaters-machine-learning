"""Dedup-aware stratified train/test split with persisted CV folds (Week 3).

Design decisions (agreed 2026-07-01, rationale in reports/methodology.md):

  * 80/20 stratified holdout. The 20% test set is written once here and is
    UNTOUCHED until the final Week-4+ evaluation.
  * 5-fold stratified CV fold ids are assigned NOW, inside the training set,
    and persisted. Every Week-4 model (RF / XGBoost / SVM / baselines) is
    tuned and compared on the *same* folds -- differences between models can't
    be fold luck.
  * Dedup-awareness: the split consumes the keep flag from the Week-3 data-
    quality pass (scripts/04_data_quality.py). Because that policy removes
    every duplicate copy (one survivor per same-label group, none for
    mixed-label groups), the kept matrix contains no identical feature
    vectors at all -- verified by an assert -- so no group-aware splitter is
    needed and plain stratification is sufficient.
  * One seed (config.RANDOM_SEED) drives both the holdout and the folds.

The persisted artifact is one row per ORIGINAL instance (12,000), so any future
question ("where did row 7312 end up and why?") has a single answer file.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config as C


def make_split(instance_table: pd.DataFrame, features: pd.DataFrame,
               test_size: float = C.TEST_SIZE, n_folds: int = C.N_FOLDS,
               seed: int = C.RANDOM_SEED) -> pd.DataFrame:
    """Return the split table: row, label, keep, drop_reason, split, cv_fold.

    split   : 'train' / 'test' / 'dropped'
    cv_fold : 0..n_folds-1 for train rows, -1 otherwise
    """
    t = instance_table[["row", "label", "keep", "drop_reason"]].copy()

    kept = t[t["keep"]]
    feat_cols = [c for c in features.columns if c != "label"]
    if features.loc[kept["row"], feat_cols].duplicated().any():
        raise AssertionError("duplicate feature vectors survived the keep "
                             "policy -- dedup and split are out of sync")

    train_rows, test_rows = train_test_split(
        kept["row"].to_numpy(), test_size=test_size, random_state=seed,
        stratify=kept["label"].to_numpy())

    t["split"] = "dropped"
    t.loc[t["row"].isin(train_rows), "split"] = "train"
    t.loc[t["row"].isin(test_rows), "split"] = "test"

    # persisted CV folds inside train, stratified, same seed
    t["cv_fold"] = -1
    tr = t[t["split"] == "train"]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(
            skf.split(np.zeros(len(tr)), tr["label"].to_numpy())):
        t.loc[tr.index[val_idx], "cv_fold"] = fold
    return t


def load_split(split_path=None) -> pd.DataFrame:
    """Load the persisted split table (outputs/splits/splits.parquet)."""
    return pd.read_parquet(split_path or C.SPLIT_DIR / "splits.parquet")


def load_model_ready(features: pd.DataFrame | None = None,
                     feature_list: list[str] | None = None):
    """Convenience for Week 4+: (X_train, y_train, X_test, y_test, fold_ids).

    Joins features.parquet with the persisted split so no downstream script
    ever re-derives the split. fold_ids aligns with X_train rows.
    """
    import json
    if features is None:
        features = pd.read_parquet(C.FEAT_DIR / "features.parquet")
    if feature_list is None:
        p = C.FEAT_DIR / "final_features.json"
        feature_list = (json.loads(p.read_text()) if p.exists()
                        else [c for c in features.columns if c != "label"])
    s = load_split()
    tr, te = s[s["split"] == "train"], s[s["split"] == "test"]
    X_train = features.loc[tr["row"], feature_list].to_numpy(np.float64)
    X_test = features.loc[te["row"], feature_list].to_numpy(np.float64)
    return (X_train, tr["label"].to_numpy(), X_test, te["label"].to_numpy(),
            tr["cv_fold"].to_numpy())
