"""Week 5: calibrate champion margins into P(cheater) for the triage framing.

The champion SVM outputs decision-function margins. For a review queue,
"P(cheater) = 0.83" is far more actionable than "margin = -0.2". Platt
(sigmoid) and isotonic calibration are compared with a fold-honest Brier
score on the SAME persisted folds used everywhere else; the winner is refit
on all OOF scores and saved. Test probabilities are produced descriptively
(the frozen thresholds and W4 conclusions are untouched).

Run:  PYTHONPATH=src python scripts/12_calibrate_scores.py
Outputs:
    outputs/models/calibrator.joblib      raw margin -> P(cheater)
    outputs/analysis/calibration.json     Brier comparison + key mappings
    outputs/figures/fig13_calibration.png reliability curve
"""
import sys
import json
import pathlib

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cheatdetect import config as C, error_analysis as EA  # noqa: E402

MODELS_DIR = C.OUTPUT_DIR / "models"
ANA_DIR = C.OUTPUT_DIR / "analysis"


def main():
    ANA_DIR.mkdir(parents=True, exist_ok=True)
    oof = pd.read_parquet(MODELS_DIR / "champion_oof.parquet")
    test = pd.read_parquet(MODELS_DIR / "test_scores.parquet")
    thr = json.loads((MODELS_DIR / "champion.json").read_text()
                     )["thresholds_frozen_from_oof"]
    s, y, folds = (oof["score"].to_numpy(), oof["y"].to_numpy(),
                   oof["fold"].to_numpy())

    # ---- fold-honest method comparison ---------------------------------------
    briers = {m: EA.cv_brier(s, y, folds, m) for m in ("platt", "isotonic")}
    base = brier_score_loss(y, np.full_like(s, y.mean(), dtype=float))
    winner = min(briers, key=briers.get)
    print(f"fold-honest Brier: platt {briers['platt']:.4f}, "
          f"isotonic {briers['isotonic']:.4f} "
          f"(predict-base-rate baseline {base:.4f}) -> {winner}")

    # ---- final calibrator on all OOF, applied descriptively to test ----------
    cal = EA.fit_calibrator(s, y, winner)
    test_brier = float(brier_score_loss(test["y"], cal(test["score"])))
    print(f"test Brier (descriptive): {test_brier:.4f}")

    mappings = {
        "P_at_f1max_threshold": round(float(cal([thr["f1_max"]])[0]), 4),
        "P_at_fpr1pct_threshold": round(float(cal([thr["fpr_1pct"]])[0]), 4),
    }
    print(f"frozen thresholds in probability terms: {mappings}")

    joblib.dump({"method": winner, "calibrator": cal},
                MODELS_DIR / "calibrator.joblib")
    (ANA_DIR / "calibration.json").write_text(json.dumps({
        "cv_brier": {k: round(v, 4) for k, v in briers.items()},
        "brier_baseline_predict_base_rate": round(base, 4),
        "winner": winner, "test_brier_descriptive": round(test_brier, 4),
        **mappings}, indent=2))

    # ---- fig13: reliability curve (OOF, fold-honest predictions) -------------
    p_honest = np.full_like(s, np.nan, dtype=float)
    for f in sorted(np.unique(folds)):
        val = folds == f
        p_honest[val] = EA.fit_calibrator(s[~val], y[~val], winner)(s[val])
    bins = np.quantile(p_honest, np.linspace(0, 1, 11))
    bins[0], bins[-1] = 0, 1
    which = np.digitize(p_honest, bins[1:-1])
    xs = [p_honest[which == b].mean() for b in range(10)]
    ys = [y[which == b].mean() for b in range(10)]
    ns = [(which == b).sum() for b in range(10)]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.plot(xs, ys, "o-", color="#4285f4", label=f"{winner} (OOF, fold-honest)")
    for x_, y_, n_ in zip(xs, ys, ns):
        ax.annotate(f"n={n_}", (x_, y_), textcoords="offset points",
                    xytext=(6, -10), fontsize=7, color="gray")
    ax.set(xlabel="predicted P(cheater)", ylabel="observed cheater fraction",
           title=f"Calibration — {winner} on champion margins "
                 f"(CV Brier {briers[winner]:.3f})",
           xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig13_calibration.png", dpi=150)
    print(f"Saved -> {MODELS_DIR}/calibrator.joblib, "
          f"{ANA_DIR}/calibration.json, fig13")


if __name__ == "__main__":
    main()
