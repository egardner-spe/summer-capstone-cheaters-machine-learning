"""Week 6: adversarial robustness sweep -- how cheaply can a cheat evade?

For each perturbation (robustness.PERTURBATIONS) applied to the RAW telemetry
of the 396 test cheaters: re-extract features, score with the FROZEN champion,
and measure recall at the frozen thresholds plus the blatancy-index shift --
quantifying Week 5's open question (how much perturbation pushes a blatant
cheater into the invisible subtle region?). Three moderate settings are also
applied to the 1,984 test legit players as an FPR sanity check (a humaniser
should not make INNOCENT players look guilty; smoothing might).

Model + thresholds stay frozen; this is evaluation of the deployed detector.
RESUMABLE: one checkpoint unit per (population, setting).

Run (repeat until done):  PYTHONPATH=src python scripts/14_robustness.py
Outputs:
    outputs/analysis/robustness.csv
    outputs/figures/fig16_evasion_curves.png
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
from cheatdetect import (config as C, data_loading as D, error_analysis as EA,  # noqa: E402
                         evaluation as E, features as F, robustness as R,
                         splitting as S)

MODELS_DIR = C.OUTPUT_DIR / "models"
ANA_DIR = C.OUTPUT_DIR / "analysis"
CKPT = MODELS_DIR / ".ckpt_robust"
LEGIT_SETTINGS = ["none", "smooth_0.5", "jitter_0.10", "delay_2"]


def load_raw(label: int, rows: np.ndarray) -> np.ndarray:
    """Raw blocks for the given feature-matrix rows, via the identity table."""
    tab = pd.read_parquet(C.QUALITY_DIR / "instance_table.parquet")
    sub = tab.set_index("row").loc[rows]
    src = "cheaters" if label == 1 else "legit"
    assert (sub["source"] == src).all()
    arr = D.load_class(label)
    return np.asarray(arr[sub["source_index"].to_numpy()], dtype=np.float32)


def unit(population: str, setting: str):
    """Perturb -> re-extract -> score with frozen champion -> summarize."""
    import json as _json
    final = _json.loads((C.FEAT_DIR / "final_features.json").read_text())
    thr = _json.loads((MODELS_DIR / "champion.json").read_text()
                      )["thresholds_frozen_from_oof"]
    pipe = joblib.load(MODELS_DIR / "champion.joblib")
    split = S.load_split()
    label = 1 if population == "cheater" else 0
    rows = split.loc[(split["split"] == "test"), "row"].to_numpy()
    feats_all = pd.read_parquet(C.FEAT_DIR / "features.parquet")
    rows = rows[feats_all.loc[rows, "label"].to_numpy() == label]

    blocks = load_raw(label, rows)
    pert, dropped = R.PERTURBATIONS[setting](blocks)
    fdf = pd.DataFrame(F.extract_chunk(pert))

    # blatancy/skill index anchored to legit TRAIN (same anchor as Week 5)
    oof = pd.read_parquet(MODELS_DIR / "champion_oof.parquet")
    idx = EA.ExtremityIndex().fit(
        feats_all.loc[oof.loc[oof["y"] == 0, "row"]])
    index = idx.transform(fdf)

    scores = pipe.decision_function(fdf[final].to_numpy(np.float64))
    met = {
        "population": population, "setting": setting,
        "n": len(rows),
        "recall_or_fpr_strict": round(float((scores >= thr["fpr_1pct"]).mean()), 4),
        "recall_or_fpr_f1max": round(float((scores >= thr["f1_max"]).mean()), 4),
        "mean_score": round(float(scores.mean()), 4),
        "mean_index": round(float(index.mean()), 4),
        "fires_dropped": round(float(dropped), 4),
    }
    return scores, met


def main():
    ANA_DIR.mkdir(parents=True, exist_ok=True)
    units = []
    for setting in R.PERTURBATIONS:
        units.append((f"cheater__{setting}",
                      lambda s=setting: unit("cheater", s)))
    for setting in LEGIT_SETTINGS:
        units.append((f"legit__{setting}",
                      lambda s=setting: unit("legit", s)))

    done, total = E.run_checkpointed(units, CKPT, budget_seconds=20)
    if done < total:
        print(f"progress: {done}/{total} units -- re-run to continue")
        return

    rows = [E.load_checkpoint(CKPT, uid)[1] for uid, _ in units]
    tab = pd.DataFrame(rows)
    tab.to_csv(ANA_DIR / "robustness.csv", index=False)
    print("=== robustness sweep (frozen champion, frozen thresholds) ===")
    print(tab.to_string(index=False))
    make_figure(tab)
    print(f"\nSaved -> {ANA_DIR}/robustness.csv + fig16")


def make_figure(tab):
    ch = tab[tab["population"] == "cheater"].set_index("setting")
    base_strict = ch.loc["none", "recall_or_fpr_strict"]
    base_f1 = ch.loc["none", "recall_or_fpr_f1max"]
    base_idx = ch.loc["none", "mean_index"]

    fams = {"smooth": ["none", "smooth_0.7", "smooth_0.5", "smooth_0.3"],
            "jitter": ["none", "jitter_0.05", "jitter_0.10", "jitter_0.20"],
            "delay": ["none", "delay_1", "delay_2", "delay_4"]}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = {"smooth": "#4285f4", "jitter": "#ea8600", "delay": "#a142f4"}
    for fam, order in fams.items():
        sub = ch.loc[order]
        x = np.arange(len(order))
        axes[0].plot(x, sub["recall_or_fpr_strict"], "o-",
                     color=colors[fam], label=fam)
        axes[0].plot(x, sub["recall_or_fpr_f1max"], "o--",
                     color=colors[fam], alpha=0.45)
        axes[1].plot(x, sub["mean_index"], "o-", color=colors[fam], label=fam)
    for ax, base, ttl, yl in (
            (axes[0], base_strict, "recall vs perturbation strength\n"
             "(solid @FPR≤1%, dashed @F1-max)", "recall on test cheaters"),
            (axes[1], base_idx, "blatancy-index shift", "mean blatancy index")):
        ax.axhline(base, ls=":", c="k", lw=1)
        ax.set_xticks(range(4),
                      ["none", "weak", "medium", "strong"])
        ax.set(title=ttl, ylabel=yl, xlabel="perturbation strength")
        ax.legend()
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "fig16_evasion_curves.png", dpi=150)


if __name__ == "__main__":
    main()
