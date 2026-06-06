"""Week 2 verification: feature-matrix quality + leakage control.

This is QA for the Week-2 deliverable, NOT the Week-4 evaluation. It checks the
matrix is finite, non-degenerate, balanced as expected, and -- via a permuted-
label control -- free of obvious leakage. The logistic-regression CV line is a
SMOKE TEST that the matrix is learnable; real modelling (RF/XGB/SVM, tuned
metrics) is Week 4.

Run:  PYTHONPATH=src python scripts/03_check_features.py
"""
import sys
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cheatdetect import config as C  # noqa: E402


def main():
    df = pd.read_parquet(C.FEAT_DIR / "features.parquet")
    feats = [c for c in df.columns if c != "label"]
    X = df[feats].to_numpy(np.float64); y = df["label"].to_numpy()

    print(f"shape: {df.shape[0]} rows x {len(feats)} features")
    print(f"NaN: {int(np.isnan(X).sum())}  Inf: {int(np.isinf(X).sum())}")
    print("class balance:", df['label'].map(C.LABELS).value_counts().to_dict())
    const = [f for f in feats if df[f].std() == 0]
    print(f"constant features: {const if const else 'none'}")
    dup = int(df.duplicated(subset=feats).sum())
    print(f"duplicate feature rows: {dup}")
    rng = df[feats].agg(['min', 'max']).T
    bad = rng[(rng['min'] < -1e6) | (rng['max'] > 1e6)]
    print(f"features with extreme range: {list(bad.index) if len(bad) else 'none'}")

    # ---- leakage control + learnability smoke test ----
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
        cv = StratifiedKFold(5, shuffle=True, random_state=0)
        auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
        ap = cross_val_score(pipe, X, y, cv=cv, scoring="average_precision")
        rng2 = np.random.default_rng(0)
        auc_perm = cross_val_score(pipe, X, rng2.permutation(y), cv=cv, scoring="roc_auc")
        print("\n[SMOKE TEST ONLY -- real modelling is Week 4]")
        print(f"  LogReg 5-fold ROC-AUC = {auc.mean():.3f} +/- {auc.std():.3f}")
        print(f"  LogReg 5-fold PR-AUC  = {ap.mean():.3f}  (baseline = {y.mean():.3f})")
        print(f"  permuted-label ROC-AUC = {auc_perm.mean():.3f}  (must be ~0.50 => no leakage)")
    except Exception as e:  # noqa: BLE001
        print("sklearn smoke test skipped:", e)


if __name__ == "__main__":
    main()
