"""Week 4 stage 1: compare model x imbalance-strategy configs on the folds.

15 configs (logreg/rf/xgb/svm x none/class_weight/smote/smote_cw, minus
xgb+smote_cw which is provably identical to xgb+smote -- see modeling.py).
Every config is scored OUT-OF-FOLD on the same persisted 5 folds; the frozen
test set is not read by this script.

RESUMABLE: work is checkpointed per (config, fold) unit under
outputs/models/.ckpt_stage1/. Re-run the script until it reports completion;
each run does up to ~30 s of new units. Interruptions cost at most one unit.

Run (repeat until done):  PYTHONPATH=src python scripts/08_compare_models.py
Outputs (written once all 75 units exist):
    outputs/models/cv_comparison.csv    one row per config, ranked by PR-AUC
    outputs/models/oof_scores.parquet   per-train-row OOF score per config
                                        (Week 5 error analysis feeds on this)
    outputs/models/per_fold_metrics.csv fold-level AUCs (variance audit)
"""
import sys
import pathlib
import shutil

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cheatdetect import config as C, evaluation as E, modeling as M, splitting as S  # noqa: E402

MODELS_DIR = C.OUTPUT_DIR / "models"
CKPT = MODELS_DIR / ".ckpt_stage1"
BUDGET_SECONDS = 30

# fast families first so early runs finish many units
ORDER = {"logreg": 0, "xgb": 1, "svm": 2, "rf": 3}


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    X_train, y_train, _, _, folds = S.load_model_ready()
    split = S.load_split()
    train_rows = split.loc[split["split"] == "train", "row"].to_numpy()

    configs = sorted(M.run_matrix(), key=lambda ms: ORDER[ms[0]])
    units = []
    for model, strat in configs:
        for f in sorted(np.unique(folds)):
            uid = f"{model}__{strat}__f{f}"
            fn = (lambda m=model, s=strat, ff=f:
                  E.fit_score_fold(lambda: M.build_pipeline(m, s),
                                   X_train, y_train, folds, ff))
            units.append((uid, fn))

    done, total = E.run_checkpointed(units, CKPT, BUDGET_SECONDS)
    if done < total:
        print(f"progress: {done}/{total} units -- re-run to continue")
        return

    # ---- all units present: assemble final artifacts -------------------------
    rows, fold_rows = [], []
    oof = pd.DataFrame({"row": train_rows, "y": y_train, "fold": folds})
    for model, strat in configs:
        name = f"{model}__{strat}"
        scores = np.full(len(y_train), np.nan)
        per_fold = []
        for f in sorted(np.unique(folds)):
            s, met = E.load_checkpoint(CKPT, f"{name}__f{f}")
            scores[folds == f] = s
            per_fold.append(met)
            fold_rows.append({"model": model, "strategy": strat, **met})
        assert not np.isnan(scores).any()
        oof[name] = scores
        rows.append({"model": model, "strategy": strat,
                     **E.summarize_config(y_train, scores, per_fold)})

    table = pd.DataFrame(rows).sort_values(
        "pr_auc_mean", ascending=False).reset_index(drop=True)
    table.to_csv(MODELS_DIR / "cv_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(
        MODELS_DIR / "per_fold_metrics.csv", index=False)
    oof.to_parquet(MODELS_DIR / "oof_scores.parquet", index=False)
    try:                             # cleanup is best-effort: some
        shutil.rmtree(CKPT)          # environments forbid deletion
    except OSError:
        print(f'note: could not remove {CKPT} -- delete it manually')

    print(f"=== stage 1 complete: {total} units -> "
          f"ranking by PR-AUC (chance = {y_train.mean():.3f}) ===")
    cols = ["model", "strategy", "pr_auc_mean", "roc_auc_mean",
            "mcc_at_f1max", "recall_at_fpr1"]
    print(table[cols].to_string(index=False))
    print(f"\nSaved -> {MODELS_DIR}/cv_comparison.csv, oof_scores.parquet, "
          "per_fold_metrics.csv")
    print("Test set untouched. Stage 2 (tuning) reads the ranking.")


if __name__ == "__main__":
    main()
