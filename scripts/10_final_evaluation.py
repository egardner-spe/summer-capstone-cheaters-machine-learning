"""Week 4 stage 3: the one-shot frozen-test evaluation.

Protocol: the champion config and both operating thresholds were fixed in
scripts/09 from training-fold data only. This script refits the champion on
the FULL training set, scores the frozen test set exactly once, and reports.
After this run the test set is spent: any further tuning would have to be
declared as such in the report.

Run:  PYTHONPATH=src python scripts/10_final_evaluation.py
Outputs:
    outputs/models/champion.joblib       fitted pipeline (W5/W6 reuse)
    outputs/models/test_results.json     the numbers the report cites
    outputs/models/test_scores.parquet   per-test-row scores (W5 error analysis)
    outputs/figures/fig7_cv_model_comparison.png
    outputs/figures/fig8_test_roc_pr.png
    outputs/figures/fig9_test_confusion.png
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
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             roc_auc_score, roc_curve)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cheatdetect import config as C, evaluation as E, modeling as M, splitting as S  # noqa: E402

MODELS_DIR = C.OUTPUT_DIR / "models"


def main():
    champion = json.loads((MODELS_DIR / "champion.json").read_text())
    thr = champion["thresholds_frozen_from_oof"]
    X_train, y_train, X_test, y_test, _ = S.load_model_ready()
    split = S.load_split()
    test_rows = split.loc[split["split"] == "test", "row"].to_numpy()

    print(f"champion: {champion['model']}+{champion['strategy']} "
          f"{champion['params'] or '(defaults)'}")
    print(f"frozen thresholds: {thr}")

    pipe = M.build_pipeline(champion["model"], champion["strategy"],
                            **champion["params"])
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODELS_DIR / "champion.joblib")

    # ---- the single test evaluation -----------------------------------------
    s_test = E.score_of(pipe, X_test)
    results = {
        "champion": {k: champion[k] for k in ("model", "strategy", "params")},
        "cv_reference": champion["cv"],
        "test_roc_auc": round(float(roc_auc_score(y_test, s_test)), 4),
        "test_pr_auc": round(float(average_precision_score(y_test, s_test)), 4),
        "test_pr_baseline": round(float(y_test.mean()), 4),
        "at_f1max": E.thresholded_metrics(y_test, s_test, thr["f1_max"]),
        "at_fpr1pct": E.thresholded_metrics(y_test, s_test, thr["fpr_1pct"]),
    }
    (MODELS_DIR / "test_results.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame({"row": test_rows, "y": y_test, "score": s_test}).to_parquet(
        MODELS_DIR / "test_scores.parquet", index=False)

    print(f"\nTEST  ROC-AUC {results['test_roc_auc']:.4f}   "
          f"PR-AUC {results['test_pr_auc']:.4f} "
          f"(chance {results['test_pr_baseline']:.3f})")
    for tag, r in (("F1-max", results["at_f1max"]),
                   ("FPR<=1%", results["at_fpr1pct"])):
        print(f"  @{tag:<8s} P {r['precision']:.3f}  R {r['recall']:.3f}  "
              f"F1 {r['f1']:.3f}  MCC {r['mcc']:.3f}  FPR {r['fpr']:.4f}  "
              f"[tp {r['tp']} fp {r['fp']} fn {r['fn']} tn {r['tn']}]")

    make_figures(y_test, s_test, thr, results)
    print(f"\nSaved -> {MODELS_DIR}/champion.joblib, test_results.json, "
          f"test_scores.parquet + figs 7-9")
    print("The test set is now SPENT. No further tuning without declaring it.")


def make_figures(y_test, s_test, thr, results):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    # fig7: CV comparison (stage-1 table)
    cv = pd.read_csv(MODELS_DIR / "cv_comparison.csv")
    cv["config"] = cv["model"] + "+" + cv["strategy"]
    cv = cv.sort_values("pr_auc_mean")
    colors = {"none": "#9aa0a6", "class_weight": "#4285f4",
              "smote": "#ea8600", "smote_cw": "#a142f4"}
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(cv["config"], cv["pr_auc_mean"],
            xerr=cv["pr_auc_std"], color=[colors[s] for s in cv["strategy"]])
    ax.axvline(results["test_pr_baseline"], ls="--", c="k", lw=1,
               label=f"chance ({results['test_pr_baseline']:.3f})")
    ax.set_xlabel("out-of-fold PR-AUC (mean ± std over 5 persisted folds)")
    ax.set_title("Week 4 stage 1: model × imbalance strategy (CV, train only)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig7_cv_model_comparison.png", dpi=150)

    # fig8: test ROC + PR with the two frozen operating points
    fpr, tpr, _ = roc_curve(y_test, s_test)
    prec, rec, _ = precision_recall_curve(y_test, s_test)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    a1.plot(fpr, tpr, lw=2)
    a1.plot([0, 1], [0, 1], "k--", lw=1)
    a2.plot(rec, prec, lw=2)
    a2.axhline(results["test_pr_baseline"], ls="--", c="k", lw=1)
    for tag, t, m in (("F1-max", thr["f1_max"], "o"),
                      ("FPR≤1%", thr["fpr_1pct"], "s")):
        r = E.thresholded_metrics(y_test, s_test, t)
        a1.plot(r["fpr"], r["recall"], m, ms=9, label=tag)
        a2.plot(r["recall"], r["precision"], m, ms=9, label=tag)
    a1.set(xlabel="FPR", ylabel="TPR (recall)",
           title=f"ROC — test (AUC {results['test_roc_auc']:.3f})")
    a2.set(xlabel="recall", ylabel="precision",
           title=f"PR — test (AUC {results['test_pr_auc']:.3f})")
    a1.legend(); a2.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig8_test_roc_pr.png", dpi=150)

    # fig9: confusion matrices at the two frozen points
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, (tag, key) in zip(axes, (("F1-max", "at_f1max"),
                                     ("FPR≤1%", "at_fpr1pct"))):
        r = results[key]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max()/2 else "black", fontsize=13)
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=["pred legit", "pred cheater"],
               yticklabels=["legit", "cheater"],
               title=f"test @ {tag} (thr={r['threshold']:.3f})")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig9_test_confusion.png", dpi=150)


if __name__ == "__main__":
    main()
