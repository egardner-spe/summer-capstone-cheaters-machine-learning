"""Week 3 QA: prove the imbalance pipeline is wired correctly. NOT Week-4 results.

Three checks, all on the persisted split/folds (scripts/05) and final feature
list (scripts/06):

  1. Per-fold resample audit -- in-fold SMOTE counts, and verification that
     synthetic minority samples stay within the real minority's feature range.
  2. Plumbing smoke test -- one plain LogReg through each strategy pipeline,
     evaluated on the persisted CV folds (PredefinedSplit). This verifies the
     pipeline runs end-to-end and that strategies behave sanely relative to
     each other. Absolute numbers are NOT results; RF/XGB/SVM are Week 4.
  3. Permuted-label control THROUGH the SMOTE pipeline -- must land ~0.50.
     This is the strongest wiring check: if SMOTE leaked resampled rows into
     validation folds, permuted labels would score above chance.

The frozen test set is not touched by this script.

Run:  PYTHONPATH=src python scripts/07_check_imbalance.py
Outputs:
    outputs/quality/imbalance_check.json   (all numbers used by the write-up)
"""
import sys
import json
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cheatdetect import config as C, imbalance as I, splitting as S  # noqa: E402

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import PredefinedSplit, cross_val_score  # noqa: E402


def main():
    X_train, y_train, X_test, y_test, folds = S.load_model_ready()
    print(f"train: {X_train.shape}  test: {X_test.shape} (untouched here)")
    print(f"features: {X_train.shape[1]} (final list)  folds: "
          f"{sorted(set(folds.tolist()))}")

    report = {"n_train": int(len(y_train)), "n_features": int(X_train.shape[1]),
              "scale_pos_weight_for_xgb": round(I.scale_pos_weight(y_train), 4)}

    # ---- 1. per-fold SMOTE audit -------------------------------------------
    print("\n[1] in-fold SMOTE resample audit")
    audit = I.fold_resample_report(X_train, y_train, folds)
    for r in audit:
        print(f"  fold {r['fold_held_out']} held out: "
              f"{r['train_cheater_real']} real cheaters -> "
              f"{r['train_cheater_after_smote']} after SMOTE "
              f"(+{r['synthetic_added']} synthetic, within real range: "
              f"{r['synthetic_within_real_range']})")
    assert all(r["synthetic_within_real_range"] for r in audit), \
        "synthetic samples escaped the real minority range"
    report["fold_audit"] = audit

    # ---- 2. strategy plumbing test (QA ONLY) --------------------------------
    print("\n[2] strategy plumbing test -- LogReg, persisted folds "
          "(QA ONLY, real modelling is Week 4)")
    cv = PredefinedSplit(folds)
    base = LogisticRegression(max_iter=5000)
    report["strategies"] = {}
    for strat in I.STRATEGIES:
        pipe = I.make_pipeline(base, strategy=strat)
        auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        ap = cross_val_score(pipe, X_train, y_train, cv=cv,
                             scoring="average_precision")
        print(f"  {strat:<12s} ROC-AUC {auc.mean():.3f} +/- {auc.std():.3f}   "
              f"PR-AUC {ap.mean():.3f}")
        report["strategies"][strat] = {
            "roc_auc_mean": round(float(auc.mean()), 4),
            "roc_auc_std": round(float(auc.std()), 4),
            "pr_auc_mean": round(float(ap.mean()), 4)}
    print(f"  (PR-AUC chance baseline = {y_train.mean():.3f})")
    report["pr_baseline"] = round(float(y_train.mean()), 4)

    # ---- 3. permuted-label control through SMOTE ----------------------------
    print("\n[3] permuted-label control through the SMOTE pipeline")
    rng = np.random.default_rng(C.RANDOM_SEED)
    y_perm = rng.permutation(y_train)
    pipe = I.make_pipeline(base, strategy="smote")
    auc_perm = cross_val_score(pipe, X_train, y_perm, cv=cv, scoring="roc_auc")
    print(f"  permuted ROC-AUC = {auc_perm.mean():.3f} "
          "(must be ~0.50 => resampling does not leak)")
    report["permuted_roc_auc"] = round(float(auc_perm.mean()), 4)
    assert abs(auc_perm.mean() - 0.5) < 0.03, "permuted control off chance!"

    out = C.QUALITY_DIR / "imbalance_check.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nSaved -> {out}")
    print("All wiring checks passed. Model training starts in Week 4.")


if __name__ == "__main__":
    main()
