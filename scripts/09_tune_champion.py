"""Week 4 stage 2: light tuning on the top configs, champion selection,
threshold freezing.

Compare-then-tune design: stage 1 (scripts/08) ranked 15 model x strategy
configs at sensible defaults; this script grid-tunes only the TOP TWO model
families (each with its best strategy) on the same persisted folds. Champion
= best PR-AUC (ties: MCC at F1-max, then fit time -- prefer the cheaper model).

The two operating thresholds (F1-max, FPR<=1%) are chosen from the champion's
OUT-OF-FOLD scores and frozen into champion.json HERE, before any script
touches the test set. scripts/10 applies them without recomputation.

RESUMABLE: checkpointed per (config, fold) unit under
outputs/models/.ckpt_stage2/ -- re-run until it reports completion.

Run (repeat until done):  PYTHONPATH=src python scripts/09_tune_champion.py
Outputs:
    outputs/models/tuning_results.csv
    outputs/models/champion.json        (config + params + frozen thresholds)
    outputs/models/champion_oof.parquet (champion OOF scores, for Week 5)
"""
import sys
import json
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cheatdetect import config as C, evaluation as E, modeling as M, splitting as S  # noqa: E402

MODELS_DIR = C.OUTPUT_DIR / "models"
CKPT = MODELS_DIR / ".ckpt_stage2"
BUDGET_SECONDS = 30
N_FAMILIES = 2


def candidate_configs():
    """Deterministic tuning candidates from the stage-1 ranking."""
    cv = pd.read_csv(MODELS_DIR / "cv_comparison.csv")
    finalists = (cv.sort_values("pr_auc_mean", ascending=False)
                   .groupby("model", sort=False).head(1).head(N_FAMILIES))
    cands = []
    for _, fin in finalists.iterrows():
        model, strat = fin["model"], fin["strategy"]
        for gi, params in enumerate([{}] + M.TUNING_GRIDS[model]):
            cands.append((f"{model}__{strat}__g{gi}", model, strat, params))
    return cands


def main():
    X_train, y_train, _, _, folds = S.load_model_ready()
    split = S.load_split()
    train_rows = split.loc[split["split"] == "train", "row"].to_numpy()

    cands = candidate_configs()
    units = []
    for cid, model, strat, params in cands:
        for f in sorted(np.unique(folds)):
            fn = (lambda m=model, s=strat, p=params, ff=f:
                  E.fit_score_fold(lambda: M.build_pipeline(m, s, **p),
                                   X_train, y_train, folds, ff))
            units.append((f"{cid}__f{f}", fn))

    done, total = E.run_checkpointed(units, CKPT, BUDGET_SECONDS)
    if done < total:
        print(f"progress: {done}/{total} units -- re-run to continue")
        return

    # ---- assemble, rank, freeze ---------------------------------------------
    rows, best = [], None
    for cid, model, strat, params in cands:
        scores = np.full(len(y_train), np.nan)
        per_fold = []
        for f in sorted(np.unique(folds)):
            s, met = E.load_checkpoint(CKPT, f"{cid}__f{f}")
            scores[folds == f] = s
            per_fold.append(met)
        summ = E.summarize_config(y_train, scores, per_fold)
        rows.append({"model": model, "strategy": strat,
                     "params": json.dumps(params), **summ})
        key = (summ["pr_auc_mean"], summ["mcc_at_f1max"],
               -summ["fit_seconds"])
        if best is None or key > best["key"]:
            best = {"key": key, "row": rows[-1], "scores": scores}

    tuning = pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False)
    tuning.to_csv(MODELS_DIR / "tuning_results.csv", index=False)
    print("=== stage 2 complete: top 8 of "
          f"{len(cands)} tuned configs ===")
    print(tuning[["model", "strategy", "params", "pr_auc_mean",
                  "roc_auc_mean", "mcc_at_f1max"]]
          .head(8).to_string(index=False))

    champ = best["row"]
    thresholds = E.pick_thresholds(y_train, best["scores"])
    champion = {
        "model": champ["model"], "strategy": champ["strategy"],
        "params": json.loads(champ["params"]),
        "cv": {k: champ[k] for k in
               ("pr_auc_mean", "pr_auc_std", "roc_auc_mean", "roc_auc_std",
                "mcc_at_f1max", "recall_at_fpr1", "precision_at_fpr1")},
        "thresholds_frozen_from_oof": thresholds,
        "protocol_note": "thresholds chosen on out-of-fold training scores; "
                         "frozen before any test evaluation",
    }
    (MODELS_DIR / "champion.json").write_text(json.dumps(champion, indent=2))
    pd.DataFrame({"row": train_rows, "y": y_train, "fold": folds,
                  "score": best["scores"]}).to_parquet(
        MODELS_DIR / "champion_oof.parquet", index=False)

    print(f"\nCHAMPION: {champ['model']}+{champ['strategy']} "
          f"{champion['params'] or '(stage-1 defaults)'}")
    print(f"  CV PR-AUC {champ['pr_auc_mean']:.4f}+/-{champ['pr_auc_std']:.4f}"
          f"  ROC-AUC {champ['roc_auc_mean']:.4f}"
          f"  MCC@F1max {champ['mcc_at_f1max']:.4f}")
    print(f"  frozen thresholds: {thresholds}")
    print(f"Saved -> {MODELS_DIR}/champion.json, tuning_results.csv, "
          "champion_oof.parquet")


if __name__ == "__main__":
    main()
