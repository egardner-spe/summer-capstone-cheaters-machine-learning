"""Week 6: SHAP interpretability -- split-role design (see interpretability.py).

Part A (exact): TreeExplainer on the refit RF runner-up, whole test set --
global ranking, beeswarm, per-error-group signed attributions.
Part B (sampled): KernelExplainer on the SVM champion's margins, 120
balanced test instances, then rank-agreement between the two models.

RESUMABLE: checkpointed under outputs/models/.ckpt_shap/ (the RF part is one
unit; the kernel part is 8 batches). Re-run until it reports completion.

Run (repeat until done):  PYTHONPATH=src python scripts/13_shap_analysis.py
Outputs:
    outputs/analysis/shap_ranking_rf.csv        mean|SHAP| all 34 (full test)
    outputs/analysis/shap_group_attribution.csv signed SHAP per error group
    outputs/analysis/shap_agreement.json        RF-vs-SVM rank agreement
    outputs/models/rf_runnerup.joblib
    outputs/figures/fig14_shap_beeswarm_rf.png
    outputs/figures/fig15_shap_agreement.png
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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cheatdetect import (config as C, evaluation as E, interpretability as I,  # noqa: E402
                         modeling as M, splitting as S)

MODELS_DIR = C.OUTPUT_DIR / "models"
ANA_DIR = C.OUTPUT_DIR / "analysis"
CKPT = MODELS_DIR / ".ckpt_shap"
N_SAMPLE, BATCH, NSAMPLES, BG_K = 120, 15, 500, 25


def get_data():
    import json as _json
    feats = pd.read_parquet(C.FEAT_DIR / "features.parquet")
    final = _json.loads((C.FEAT_DIR / "final_features.json").read_text())
    X_train, y_train, X_test, y_test, _ = S.load_model_ready()
    split = S.load_split()
    test_rows = split.loc[split["split"] == "test", "row"].to_numpy()
    Xte_df = feats.loc[test_rows, final].reset_index(drop=True)
    return final, X_train, y_train, X_test, y_test, Xte_df


def best_rf_params():
    tun = pd.read_csv(MODELS_DIR / "tuning_results.csv")
    rf = tun[tun.model == "rf"].sort_values("pr_auc_mean", ascending=False)
    p = json.loads(rf.iloc[0]["params"])
    return {k: (None if v is None else v) for k, v in p.items()}


def sample_idx(y_test):
    rng = np.random.default_rng(C.RANDOM_SEED)
    ch = np.flatnonzero(y_test == 1)
    lg = np.flatnonzero(y_test == 0)
    k = N_SAMPLE // 2
    return np.sort(np.concatenate([rng.choice(ch, k, replace=False),
                                   rng.choice(lg, k, replace=False)]))


def rf_piped_unit(X_train, y_train, X_test, y_test):
    """Unit 1a: fit the PIPED best-config RF, record its test AUC."""
    from sklearn.metrics import roc_auc_score
    params = best_rf_params()
    piped = M.build_pipeline("rf", "none", **params)
    piped.fit(X_train, y_train)
    auc = roc_auc_score(y_test, piped.predict_proba(X_test)[:, 1])
    return np.array([auc]), {"piped_auc": round(float(auc), 4)}


def rf_unscaled_unit(X_train, y_train, X_test, y_test):
    """Unit 1b: fit the unscaled RF, assert equivalence, persist model."""
    piped_auc = float(E.load_checkpoint(CKPT, "rf_piped")[0][0])
    rf, auc = I.refit_rf_unscaled(X_train, y_train, X_test, y_test,
                                  best_rf_params(), C.RANDOM_SEED, piped_auc)
    joblib.dump(rf, MODELS_DIR / "rf_runnerup.joblib")
    return np.array([auc]), {"unscaled_auc": round(float(auc), 4)}


TREE_BATCH = 150


def rf_shap_unit(Xte_df, b):
    """Unit 2 (batched): exact TreeExplainer on the persisted RF."""
    rf = joblib.load(MODELS_DIR / "rf_runnerup.joblib")
    chunk = Xte_df.iloc[b * TREE_BATCH:(b + 1) * TREE_BATCH]
    sv = I.tree_shap(rf, chunk)
    return sv, {"n": sv.shape[0], "batch": b}


def kernel_batch(b, final, X_train, X_test, y_test):
    import shap
    champ = json.loads((MODELS_DIR / "champion.json").read_text())
    pipe = joblib.load(MODELS_DIR / "champion.joblib")
    idx = sample_idx(y_test)[b * BATCH:(b + 1) * BATCH]
    bg = shap.kmeans(X_train, BG_K)
    ex = shap.KernelExplainer(pipe.decision_function, bg)
    sv = ex.shap_values(X_test[idx], nsamples=NSAMPLES, silent=True)
    return np.asarray(sv), {"batch": b, "model": champ["model"]}


def main():
    ANA_DIR.mkdir(parents=True, exist_ok=True)
    final, X_train, y_train, X_test, y_test, Xte_df = get_data()

    units = [("rf_piped", lambda: rf_piped_unit(X_train, y_train,
                                                X_test, y_test)),
             ("rf_unscaled", lambda: rf_unscaled_unit(X_train, y_train,
                                                      X_test, y_test)),
             ]
    n_tree_batches = (len(Xte_df) + TREE_BATCH - 1) // TREE_BATCH
    for b in range(n_tree_batches):
        units.append((f"rf_tree_shap_b{b}",
                      lambda bb=b: rf_shap_unit(Xte_df, bb)))
    for b in range(N_SAMPLE // BATCH):
        units.append((f"kernel_b{b}",
                      lambda bb=b: kernel_batch(bb, final, X_train,
                                                X_test, y_test)))
    done, total = E.run_checkpointed(units, CKPT, budget_seconds=20)
    if done < total:
        print(f"progress: {done}/{total} units -- re-run to continue")
        return

    # ---- assemble ------------------------------------------------------------
    met_a = E.load_checkpoint(CKPT, "rf_piped")[1]
    met_b = E.load_checkpoint(CKPT, "rf_unscaled")[1]
    n_tb = (len(Xte_df) + TREE_BATCH - 1) // TREE_BATCH
    sv_rf = np.vstack([E.load_checkpoint(CKPT, f"rf_tree_shap_b{b}")[0]
                       for b in range(n_tb)])
    print(f"RF equivalence: piped AUC {met_a['piped_auc']}, "
          f"unscaled AUC {met_b['unscaled_auc']}")
    sv_svm = np.vstack([E.load_checkpoint(CKPT, f"kernel_b{b}")[0]
                        for b in range(N_SAMPLE // BATCH)])
    idx = sample_idx(y_test)

    # global ranking (RF, full test) + agreement on the identical sample
    rank_rf_full = I.mean_abs_ranking(sv_rf, final)
    rank_rf_sample = I.mean_abs_ranking(sv_rf[idx], final)
    rank_svm = I.mean_abs_ranking(sv_svm, final)
    agree = I.rank_agreement(rank_rf_sample, rank_svm, top_k=8)
    rank_rf_full.to_csv(ANA_DIR / "shap_ranking_rf.csv",
                        header=["mean_abs_shap"])
    (ANA_DIR / "shap_agreement.json").write_text(json.dumps(agree, indent=2))
    print(f"\ntop-10 RF mean|SHAP| (full test):\n{rank_rf_full.head(10)}")
    print(f"\nRF-vs-SVM agreement: rho={agree['spearman_rho']}, "
          f"top8 overlap={agree['top8_overlap']}/8")

    # per-error-group signed attribution (strict threshold, test)
    thr = json.loads((MODELS_DIR / "champion.json").read_text()
                     )["thresholds_frozen_from_oof"]["fpr_1pct"]
    ts = pd.read_parquet(MODELS_DIR / "test_scores.parquet")
    flag = (ts["score"] >= thr).to_numpy()
    yv = ts["y"].to_numpy()
    groups = {"TP": (yv == 1) & flag, "FN": (yv == 1) & ~flag,
              "FP": (yv == 0) & flag, "TN": (yv == 0) & ~flag}
    ga = I.group_attribution(sv_rf, final, groups)
    ga.to_csv(ANA_DIR / "shap_group_attribution.csv")
    top_fp = ga.iloc[:, 2].abs().sort_values(ascending=False).head(5)
    print(f"\nFP group -- strongest signed attributions:\n"
          f"{ga.loc[top_fp.index].iloc[:, 2]}")

    make_figures(sv_rf, Xte_df, rank_rf_sample, rank_svm, agree)
    print(f"\nSaved -> {ANA_DIR}/shap_* + figs 14-15 + rf_runnerup.joblib")


def make_figures(sv_rf, Xte_df, rank_rf_sample, rank_svm, agree):
    import shap
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(sv_rf, Xte_df, max_display=15, show=False)
    plt.title("RF runner-up — exact SHAP, full test set (positive = "
              "pushes toward 'cheater')", fontsize=11)
    plt.tight_layout()
    plt.savefig(C.FIG_DIR / "fig14_shap_beeswarm_rf.png", dpi=150,
                bbox_inches="tight")
    plt.close("all")

    fig, ax = plt.subplots(figsize=(7, 6))
    a = rank_rf_sample / rank_rf_sample.sum()
    b = (rank_svm / rank_svm.sum())[a.index]
    ax.scatter(a, b, s=25, color="#4285f4")
    for name in a.head(8).index.union(b.sort_values(ascending=False)
                                      .head(8).index):
        ax.annotate(name, (a[name], b[name]), fontsize=7,
                    textcoords="offset points", xytext=(4, 3))
    lim = max(a.max(), b.max()) * 1.15
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set(xlabel="RF share of mean|SHAP| (exact)",
           ylabel="SVM champion share of mean|SHAP| (kernel, sampled)",
           title=f"Two models, one story — Spearman ρ = "
                 f"{agree['spearman_rho']}, top-8 overlap "
                 f"{agree['top8_overlap']}/8")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig15_shap_agreement.png", dpi=150)


if __name__ == "__main__":
    main()
