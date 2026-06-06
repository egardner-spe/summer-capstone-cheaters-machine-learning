"""Week 2 EDA figures + univariate separability table.

Compares cheater vs legit behaviour and writes figures to outputs/figures/ and
a univariate-AUC table to outputs/features/univariate_auc.csv.

Run:  PYTHONPATH=src python scripts/01_eda_figures.py
"""
import sys
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cheatdetect import config as C, data_loading as D, features as F  # noqa: E402

SAMPLE = 2000
plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})
COL = {"cheater": "#c0392b", "legit": "#2c7fb8"}


def load_sample(label, n):
    arr = D.load_class(label)
    idx = np.linspace(0, arr.shape[0] - 1, min(n, arr.shape[0])).astype(int)
    return np.asarray(arr[idx], dtype=np.float32)


def auc(pos, neg):
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    allv = np.concatenate([pos, neg]); order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    a = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    return max(a, 1 - a)


def hist_compare(ax, ch, le, title, xlabel, bins=60, clip=None):
    data = {"cheater": ch, "legit": le}
    if clip is not None:
        data = {k: np.clip(v, *clip) for k, v in data.items()}
    rng = (min(v.min() for v in data.values()), max(v.max() for v in data.values()))
    for k, v in data.items():
        ax.hist(v, bins=bins, range=rng, density=True, histtype="step",
                lw=2, color=COL[k], label=k)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("density"); ax.legend()


def main():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.FEAT_DIR.mkdir(parents=True, exist_ok=True)
    Cb = load_sample(1, SAMPLE); Lb = load_sample(0, SAMPLE)
    fc = F.extract_chunk(Cb); fl = F.extract_chunk(Lb)

    # ---- univariate separability table ----
    rows = [(k, float(np.nanmean(fc[k])), float(np.nanmean(fl[k])), float(auc(fc[k], fl[k])))
            for k in fc]
    tbl = pd.DataFrame(rows, columns=["feature", "cheater_mean", "legit_mean", "auc"])
    tbl = tbl.sort_values("auc", ascending=False).reset_index(drop=True)
    tbl.to_csv(C.FEAT_DIR / "univariate_auc.csv", index=False)
    print(tbl.to_string(index=False))

    # ---- fig 1: raw tick-speed distribution ----
    vc = np.sqrt(Cb[..., 0] ** 2 + Cb[..., 1] ** 2).ravel()
    vl = np.sqrt(Lb[..., 0] ** 2 + Lb[..., 1] ** 2).ravel()
    sub = lambda x: x[np.linspace(0, len(x) - 1, 300000).astype(int)]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    hist_compare(ax, sub(vc), sub(vl), "Per-tick aim speed", "deg/tick", bins=80, clip=(0, 20))
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig1_speed_distribution.png"); plt.close(fig)

    # ---- fig 2: shot-centric speeds (the discriminative zone) ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    hist_compare(axes[0], fc["speed_at_shot_mean"], fl["speed_at_shot_mean"],
                 "Aim speed AT shot (per instance)", "deg/tick", clip=(0, 3))
    hist_compare(axes[1], fc["speed_pre_shot_mean"], fl["speed_pre_shot_mean"],
                 "Aim speed in 6 ticks BEFORE shot", "deg/tick", clip=(0, 3))
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig2_shot_centric_speed.png"); plt.close(fig)

    # ---- fig 3: medium-velocity shot fraction + settle ratio ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    hist_compare(axes[0], fc["frac_shots_medspeed"], fl["frac_shots_medspeed"],
                 "Fraction of medium-velocity shots", "fraction", clip=(0, 0.6))
    hist_compare(axes[1], fc["settle_ratio"], fl["settle_ratio"],
                 "Settle ratio  (speed_at_shot / speed_pre_shot)", "ratio", clip=(0, 2))
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig3_medspeed_settle.png"); plt.close(fig)

    # ---- fig 4: smoothness (jerk) + cross-engagement consistency ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    hist_compare(axes[0], fc["jerk_absmean"], fl["jerk_absmean"],
                 "Mean |jerk| (smoothness)", "deg/tick^3", clip=(0, 2))
    hist_compare(axes[1], fc["xeng_speed_cv"], fl["xeng_speed_cv"],
                 "Cross-engagement speed CV (consistency)", "CV", clip=(0, 1.5))
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig4_smoothness_consistency.png"); plt.close(fig)

    # ---- fig 5: univariate AUC ranking ----
    top = tbl.head(18).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["feature"], top["auc"], color="#34495e")
    ax.axvline(0.5, color="grey", ls="--"); ax.set_xlim(0.5, max(0.62, top["auc"].max() + 0.02))
    ax.set_title("Univariate separability (sample)"); ax.set_xlabel("AUC (separating power)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig5_univariate_auc.png"); plt.close(fig)

    # ---- fig 6: engineered-feature correlation heatmap ----
    M = pd.DataFrame({k: np.concatenate([fc[k], fl[k]]) for k in fc}).corr().values
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(M, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(fc))); ax.set_xticklabels(list(fc), rotation=90, fontsize=6)
    ax.set_yticks(range(len(fc))); ax.set_yticklabels(list(fc), fontsize=6)
    fig.colorbar(im, fraction=0.046); ax.set_title("Engineered-feature correlation")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fig6_feature_correlation.png"); plt.close(fig)

    print(f"\nSaved 6 figures -> {C.FIG_DIR}")


if __name__ == "__main__":
    main()
