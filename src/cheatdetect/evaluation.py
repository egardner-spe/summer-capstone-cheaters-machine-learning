"""Evaluation machinery (Week 4): out-of-fold scoring, metrics, op points.

Protocol invariants, enforced here so no script can get them wrong:

  * All model comparison happens on the PERSISTED folds (splits.parquet).
    Every config sees identical fold boundaries.
  * Scores are continuous (predict_proba[:,1] or decision_function) --
    thresholds are a separate, later decision.
  * Operating points are chosen on OUT-OF-FOLD training scores and frozen
    BEFORE the test set is touched:
        f1_max : threshold maximising F1 (balanced review point)
        fpr_1pct: highest-recall threshold with FPR <= 1% (ban-worthy point --
                  anticheat economics punish false accusations far more than
                  misses, so this is the deployment-relevant number)
  * matthews_corrcoef (MCC) and F1 are reported at BOTH points; ROC-AUC and
    PR-AUC are threshold-free headline metrics.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             matthews_corrcoef, precision_recall_curve,
                             roc_auc_score, roc_curve)

MAX_FPR = 0.01   # the strict operating point


def score_of(pipe, X):
    """Continuous score, higher = more cheater-like."""
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def fit_score_fold(build_fn, X, y, folds, f):
    """Fit a FRESH pipeline on all folds except f, score fold f.

    The atomic unit of work: scripts checkpoint at this granularity so long
    comparisons can resume across interrupted runs.
    """
    val = folds == f
    pipe = build_fn()
    t0 = time.perf_counter()
    pipe.fit(X[~val], y[~val])
    dt = time.perf_counter() - t0
    s = score_of(pipe, X[val])
    metrics = {"fold": int(f), "fit_seconds": round(dt, 2),
               "roc_auc": float(roc_auc_score(y[val], s)),
               "pr_auc": float(average_precision_score(y[val], s))}
    return s, metrics


def oof_scores(build_fn, X, y, folds):
    """Out-of-fold scores on the persisted folds.

    build_fn: zero-arg factory returning a FRESH pipeline (never reuse a
    fitted object across folds). Returns (scores, per_fold list of dicts).
    """
    scores = np.full(len(y), np.nan)
    per_fold = []
    for f in sorted(np.unique(folds)):
        s, m = fit_score_fold(build_fn, X, y, folds, f)
        scores[folds == f] = s
        per_fold.append(m)
    assert not np.isnan(scores).any(), "some rows never received an OOF score"
    return scores, per_fold


def run_checkpointed(units, ckpt_dir, budget_seconds: float = 30.0):
    """Run pending (uid, fn) units until the time budget is spent.

    fn() -> (scores_array, metrics_dict); each completed unit is persisted to
    ckpt_dir/<uid>.npz, so an interrupted comparison resumes for free. Returns
    (n_done, n_total). Callers re-invoke until n_done == n_total, then
    assemble via load_checkpoint().
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for uid, fn in units:
        path = ckpt_dir / f"{uid}.npz"
        if path.exists():
            continue
        if time.perf_counter() - t0 > budget_seconds:
            break
        scores, metrics = fn()
        tmp = path.with_suffix(".tmp.npz")   # atomic-ish: no partial .npz
        np.savez(tmp, scores=scores, **metrics)
        tmp.rename(path)
    done = sum(1 for uid, _ in units if (ckpt_dir / f"{uid}.npz").exists())
    return done, len(units)


def load_checkpoint(ckpt_dir, uid):
    """Read one unit back: (scores, metrics dict)."""
    with np.load(Path(ckpt_dir) / f"{uid}.npz") as z:
        scores = z["scores"]
        metrics = {k: z[k].item() for k in z.files if k != "scores"}
    return scores, metrics


def pick_thresholds(y, scores, max_fpr: float = MAX_FPR):
    """Choose both operating points from (out-of-fold) scores. Returns dict."""
    prec, rec, thr = precision_recall_curve(y, scores)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    thr_f1 = float(thr[np.argmax(f1[:-1])])           # thr is len-1 shorter

    fpr, tpr, rthr = roc_curve(y, scores)
    ok = fpr <= max_fpr
    # highest recall subject to the FPR cap (ok[0] always True: fpr starts 0)
    i = np.argmax(np.where(ok, tpr, -1.0))
    thr_fpr = float(rthr[i])
    return {"f1_max": thr_f1, "fpr_1pct": thr_fpr}


def thresholded_metrics(y, scores, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {
        "threshold": float(threshold),
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(2 * prec * rec / max(prec + rec, 1e-12), 4),
        "mcc": round(float(matthews_corrcoef(y, pred)), 4),
        "fpr": round(fp / max(fp + tn, 1), 4),
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
    }


def summarize_config(y, scores, per_fold) -> dict:
    """One comparison-table row: threshold-free + both op points, from OOF."""
    roc = [f["roc_auc"] for f in per_fold]
    pr = [f["pr_auc"] for f in per_fold]
    thrs = pick_thresholds(y, scores)
    at_f1 = thresholded_metrics(y, scores, thrs["f1_max"])
    at_lo = thresholded_metrics(y, scores, thrs["fpr_1pct"])
    return {
        "roc_auc_mean": round(float(np.mean(roc)), 4),
        "roc_auc_std": round(float(np.std(roc)), 4),
        "pr_auc_mean": round(float(np.mean(pr)), 4),
        "pr_auc_std": round(float(np.std(pr)), 4),
        "mcc_at_f1max": at_f1["mcc"], "f1_at_f1max": at_f1["f1"],
        "precision_at_f1max": at_f1["precision"],
        "recall_at_f1max": at_f1["recall"],
        "recall_at_fpr1": at_lo["recall"],
        "precision_at_fpr1": at_lo["precision"],
        "mcc_at_fpr1": at_lo["mcc"],
        "fit_seconds": round(sum(f["fit_seconds"] for f in per_fold), 1),
    }
