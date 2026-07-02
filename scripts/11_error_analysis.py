"""Week 5: error analysis -- who gets caught, who gets missed, who gets flagged.

Three questions, answered on the champion's OUT-OF-FOLD training predictions
at the thresholds frozen in Week 4 (the test set appears only in a final
descriptive confirmation -- no decisions are made from it):

  1. Subtle vs blatant: does detection rate rise with a cheater's mechanical
     extremity (blatancy index -- see error_analysis.py)?
  2. Skilled-legit false positives: do FPs concentrate among legit players
     whose mechanics most resemble an aimbot's (same index, read as skill)?
  3. Missed vs caught: what do the FN cheaters look like, feature by feature?

Run:  PYTHONPATH=src python scripts/11_error_analysis.py
Outputs:
    outputs/analysis/{recall_by_blatancy,fpr_by_skill,fn_tp_profile}.csv
    outputs/analysis/index_values.parquet   (per-instance index, OOF + test)
    outputs/analysis/test_confirmation.json
    outputs/figures/fig10_score_distributions.png
    outputs/figures/fig11_recall_by_blatancy.png
    outputs/figures/fig12_fp_by_skill.png
"""
import sys
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cheatdetect import config as C, error_analysis as EA  # noqa: E402

MODELS_DIR = C.OUTPUT_DIR / "models"
ANA_DIR = C.OUTPUT_DIR / "analysis"
QUARTILE_LABELS = ["Q1 subtle", "Q2", "Q3", "Q4 blatant"]


def main():
    ANA_DIR.mkdir(parents=True, exist_ok=True)

    feats = pd.read_parquet(C.FEAT_DIR / "features.parquet")
    oof = pd.read_parquet(MODELS_DIR / "champion_oof.parquet")
    test = pd.read_parquet(MODELS_DIR / "test_scores.parquet")
    thr = json.loads((MODELS_DIR / "champion.json").read_text()
                     )["thresholds_frozen_from_oof"]

    # ---- index, anchored to legit TRAIN only --------------------------------
    legit_train_rows = oof.loc[oof["y"] == 0, "row"]
    idx = EA.ExtremityIndex().fit(feats.loc[legit_train_rows])
    oof["index"] = idx.transform(feats.loc[oof["row"]])
    test["index"] = idx.transform(feats.loc[test["row"]])
    pd.concat([oof.assign(subset="train_oof"),
               test.assign(subset="test", fold=-1)]
              ).to_parquet(ANA_DIR / "index_values.parquet", index=False)

    ch, lg = oof[oof["y"] == 1], oof[oof["y"] == 0]
    print(f"index sanity: legit-train mean {lg['index'].mean():.3f} "
          f"(~0.50 by construction), cheater mean {ch['index'].mean():.3f}")

    # ---- 1. recall by blatancy quartile (cuts from TRAIN cheaters) ----------
    q_edges = np.quantile(ch["index"], [0, .25, .5, .75, 1.0])
    rows = []
    for tag, t in (("f1_max", thr["f1_max"]), ("fpr_1pct", thr["fpr_1pct"])):
        r = EA.rate_by_bin(ch["index"].to_numpy(),
                           (ch["score"] >= t).to_numpy(),
                           q_edges, QUARTILE_LABELS)
        r.insert(0, "threshold", tag)
        rows.append(r)
    recall_tab = pd.concat(rows, ignore_index=True)
    recall_tab.to_csv(ANA_DIR / "recall_by_blatancy.csv", index=False)
    print("\n[1] OOF recall by blatancy quartile:")
    print(recall_tab.to_string(index=False))

    # ---- 2. FP rate by legit skill decile -----------------------------------
    d_edges = np.quantile(lg["index"], np.linspace(0, 1, 11))
    rows = []
    for tag, t in (("f1_max", thr["f1_max"]), ("fpr_1pct", thr["fpr_1pct"])):
        r = EA.rate_by_bin(lg["index"].to_numpy(),
                           (lg["score"] >= t).to_numpy(),
                           d_edges, [f"D{i+1}" for i in range(10)])
        r.insert(0, "threshold", tag)
        rows.append(r)
    fp_tab = pd.concat(rows, ignore_index=True)
    fp_tab.to_csv(ANA_DIR / "fpr_by_skill.csv", index=False)
    for tag in ("f1_max", "fpr_1pct"):
        sub = fp_tab[fp_tab["threshold"] == tag]
        total_fp = sub["n_flagged"].sum()
        top = sub.iloc[-1]["n_flagged"] / max(total_fp, 1)
        top2 = sub.iloc[-2:]["n_flagged"].sum() / max(total_fp, 1)
        print(f"\n[2] @{tag}: {total_fp} OOF FPs; top skill decile carries "
              f"{top:.0%}, top two deciles {top2:.0%}")

    # ---- 3. missed vs caught cheaters (strict threshold) --------------------
    caught = (ch["score"] >= thr["fpr_1pct"]).to_numpy()
    key_feats = [f for f, _ in EA.INDEX_FEATURES]
    prof = EA.profile_groups(
        feats.loc[ch["row"]].reset_index(drop=True),
        {"caught (TP)": caught, "missed (FN)": ~caught},
        key_feats)
    prof.loc["blatancy_index"] = [ch["index"][caught].mean().round(4),
                                  ch["index"][~caught].mean().round(4)]
    prof.to_csv(ANA_DIR / "fn_tp_profile.csv")
    print(f"\n[3] missed-vs-caught profile (strict threshold):\n{prof}")

    # ---- 4. test confirmation (descriptive ONLY, train-derived cuts) --------
    tch, tlg = test[test["y"] == 1], test[test["y"] == 0]
    t_recall = EA.rate_by_bin(tch["index"].to_numpy(),
                              (tch["score"] >= thr["fpr_1pct"]).to_numpy(),
                              q_edges, QUARTILE_LABELS)
    fp_mask = (tlg["score"] >= thr["fpr_1pct"]).to_numpy()
    # skill percentile of each test FP relative to LEGIT TRAIN index values
    ref = np.sort(lg["index"].to_numpy())
    fp_pct = np.searchsorted(ref, tlg.loc[fp_mask, "index"].to_numpy()) / len(ref)
    confirmation = {
        "test_recall_by_blatancy_quartile_at_fpr1pct":
            {r["bin"]: r["rate"] for _, r in t_recall.iterrows()},
        "n_test_fps_at_fpr1pct": int(fp_mask.sum()),
        "test_fp_skill_percentiles": {
            "median": round(float(np.median(fp_pct)), 4),
            "frac_above_p80": round(float((fp_pct >= 0.8).mean()), 4),
            "frac_above_p90": round(float((fp_pct >= 0.9).mean()), 4),
        },
    }
    (ANA_DIR / "test_confirmation.json").write_text(
        json.dumps(confirmation, indent=2))
    print(f"\n[4] test confirmation @fpr_1pct: recall by quartile "
          f"{confirmation['test_recall_by_blatancy_quartile_at_fpr1pct']}")
    print(f"    {confirmation['n_test_fps_at_fpr1pct']} test FPs: median skill "
          f"percentile {confirmation['test_fp_skill_percentiles']['median']:.2f}, "
          f"{confirmation['test_fp_skill_percentiles']['frac_above_p90']:.0%} "
          "above P90")

    make_figures(oof, ch, lg, thr, recall_tab, fp_tab)
    print(f"\nSaved -> {ANA_DIR}/ tables + figs 10-12")


def make_figures(oof, ch, lg, thr, recall_tab, fp_tab):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    # fig10: score distributions + frozen thresholds
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.linspace(oof["score"].min(), oof["score"].max(), 80)
    ax.hist(lg["score"], bins=bins, density=True, alpha=0.55,
            label=f"legit (n={len(lg):,})")
    ax.hist(ch["score"], bins=bins, density=True, alpha=0.55,
            label=f"cheater (n={len(ch):,})")
    ax.axvline(thr["f1_max"], ls="--", c="k", lw=1.2, label="F1-max thr")
    ax.axvline(thr["fpr_1pct"], ls=":", c="k", lw=1.5, label="FPR≤1% thr")
    ax.set(xlabel="champion decision score (OOF)", ylabel="density",
           title="Week 5: out-of-fold score distributions — the overlap IS the problem")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig10_score_distributions.png", dpi=150)

    # fig11: recall by blatancy quartile, both thresholds
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.38
    for i, (tag, label, color) in enumerate(
            (("f1_max", "@F1-max", "#4285f4"),
             ("fpr_1pct", "@FPR≤1%", "#ea8600"))):
        sub = recall_tab[recall_tab["threshold"] == tag]
        x = np.arange(len(sub))
        ax.bar(x + (i - 0.5) * width, sub["rate"], width,
               label=label, color=color)
        for xi, r in zip(x, sub["rate"]):
            ax.text(xi + (i - 0.5) * width, r + 0.01, f"{r:.2f}",
                    ha="center", fontsize=9)
    ax.set_xticks(np.arange(4), QUARTILE_LABELS)
    ax.set(ylabel="recall (OOF)", ylim=(0, 1),
           title="Detection rate rises with blatancy — the subtle tail is the hard core")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig11_recall_by_blatancy.png", dpi=150)

    # fig12: FP rate by legit skill decile
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    for tag, label, color in (("f1_max", "@F1-max", "#4285f4"),
                              ("fpr_1pct", "@FPR≤1%", "#ea8600")):
        sub = fp_tab[fp_tab["threshold"] == tag]
        ax.plot(np.arange(1, 11), sub["rate"], "o-", label=label, color=color)
    ax.set_xticks(np.arange(1, 11))
    ax.set(xlabel="legit mechanical-skill decile (D10 = most aimbot-like mechanics)",
           ylabel="false-positive rate (OOF)",
           title="False positives concentrate on the most mechanically skilled legit players")
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig12_fp_by_skill.png", dpi=150)


if __name__ == "__main__":
    main()
