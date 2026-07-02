# Machine-Learning Detection of Cheating in Online FPS Games

Senior cybersecurity capstone (CYBR 4950, Summer 2026) — Ethan Gardner.

Behaviour-based detection of aim-assist ("aimbot") cheating in Counter-Strike,
using **view-angle telemetry** instead of cheat-software signatures. The premise:
a player's aim and firing behaviour leave statistical fingerprints, and an ML
model can be trained to tell cheaters from legitimate (including highly skilled)
players.

> **Status — Weeks 1–6 complete. All quantitative results are in.**
> Champion RBF-SVM: test ROC-AUC 0.708, PR-AUC 0.411; errors follow one
> gradient (subtle cheaters invisible, FPs = the most skilled legit players);
> SHAP confirms the model wins for the predicted shot-centric reasons; and
> the robustness sweep shows mild smoothing evades the deployed threshold
> almost for free (recall 13.1% → 1.5%) by stripping the cheat's own
> humanisation jitter. What remains is Weeks 7–8: demo, final report,
> presentation.

## What the data is

Raw per-tick telemetry, two arrays (not committed — 1.4 GB):

```
instance -> 30 engagements -> 192 ticks (~3 s @ 64-tick) -> 5 channels
channels: [d_yaw, d_pitch, yaw, pitch, fire]
cheaters.npy = (2000, 30, 192, 5)   legit.npy = (10000, 30, 192, 5)
```

Channel meanings were verified empirically, not assumed — see
[`reports/data_schema.md`](reports/data_schema.md).

## Key Week-6 finding

**Evasion is cheap, and we know the mechanism.** A barely-perceptible causal
EMA over the mouse stream (α=0.7) collapses strict-threshold recall from
13.1% to **1.5%** — not by hiding the aimbot's lock (shot-centric features
barely move) but by stripping the cheat's own **humanisation jitter**
(`zcr_dpitch` −69%), which SHAP shows the model partly keys on. The
humaniser's countermeasure is detectable; smoothing the countermeasure
defeats detection. Meanwhile SHAP confirms the 26 test FPs are flagged by
exactly the aimbot signature — the skilled-legit FP problem is structural.
Write-up: [`reports/week6_interpretability_robustness.md`](reports/week6_interpretability_robustness.md).

## Key Week-5 finding

**The detector's errors are two mirror images of one gradient.** Recall is a
monotone function of a cheater's mechanical extremity (strict threshold:
0.8% on the subtlest quartile → 36% on the most blatant — subtle cheats are
effectively invisible), while false positives concentrate on legit players
whose mechanics most resemble an aimbot's: the top skill decile carries 60%
of strict-threshold FPs, and the 26 real test FPs sit at a median skill
percentile of **0.95**. Missed cheaters are statistically legit on every
axis (settle ratio 0.997 vs legit 1.015). Threshold-banning would punish the
best players; the system is a review-queue prioritiser, now with calibrated
P(cheater) outputs. Write-up:
[`reports/week5_error_analysis.md`](reports/week5_error_analysis.md).

## Key Week-4 finding

**SMOTE hurts.** Every nonlinear model lost PR-AUC with SMOTE (SVM 0.432 →
0.371, RF 0.426 → 0.383, XGB 0.419 → 0.388): with subtle, humanised cheaters
the classes genuinely overlap, and synthetic interpolation injects noise
exactly where the decision boundary must live. No resampling won for every
family. The champion RBF-SVM, at the strict FPR≤1% operating point, catches
13% of cheaters at 0.667 precision — a 4× lift over the 16.6% base rate:
useful as **review-queue triage**, nowhere near an auto-ban trigger. Write-up:
[`reports/week4_modeling.md`](reports/week4_modeling.md).

## Key Week-3 finding

The 73 "duplicates" flagged in Week 2 were the visible part of a bigger issue:
hashing raw instances across **both** arrays found 84 byte-identical duplicate
groups — **15 of which contain the same recording labelled both cheater and
legit** (label noise, 30 rows). Mixed-label groups dropped, extra copies
deduped: 103 rows removed, 11,897 kept, imbalance unchanged (5.0:1). Audit
trail in `outputs/quality/`; write-up in
[`reports/week3_quality_split_imbalance.md`](reports/week3_quality_split_imbalance.md).

## Key Week-2 finding

No single feature separates the classes (best univariate AUC ≈ 0.63): these are
**subtle, humanised cheaters**. The signal is **shot-centric** — cheaters fire
pre-settled (crosshair already locked, fewer mid-velocity correction shots).
Combined, the 39 features reach ~0.72 ROC-AUC in a leakage-controlled smoke test.
Full write-up: [`reports/eda_findings.md`](reports/eda_findings.md).

## Repository layout

```
repo/
├── src/cheatdetect/        # importable package
│   ├── config.py           #   paths, channel semantics, thresholds, seeds
│   ├── data_loading.py     #   mmap loading + light cleaning
│   ├── features.py         #   (30,192,5) instance -> 39 behavioural features
│   ├── data_quality.py     #   raw-hash dedup + drop policy (W3)
│   ├── splitting.py        #   frozen split + persisted CV folds + loaders (W3)
│   ├── feature_selection.py#   train-only conservative prune (W3)
│   ├── imbalance.py        #   SMOTE / class-weight pipeline factory (W3)
│   ├── modeling.py         #   model zoo + tuning grids (W4)
│   ├── evaluation.py       #   OOF scoring, op points, resumable runs (W4)
│   ├── error_analysis.py   #   blatancy/skill index, profiles, calibration (W5)
│   ├── interpretability.py #   split-role SHAP machinery (W6)
│   └── robustness.py       #   raw-telemetry humaniser perturbations (W6)
├── scripts/
│   ├── 00_inspect_data.py  #   data reality check (W1)
│   ├── 01_eda_figures.py   #   EDA figures + univariate AUC (W2)
│   ├── 02_build_features.py#   build model-ready feature matrix (W2)
│   ├── 03_check_features.py#   QA: finiteness, balance, leakage control (W2)
│   ├── 04_data_quality.py  #   raw dedup verification + keep flags (W3)
│   ├── 05_make_split.py    #   frozen 80/20 split + persisted folds (W3)
│   ├── 06_finalize_features.py # 39 -> 34 prune, train-only stats (W3)
│   ├── 07_check_imbalance.py   # SMOTE wiring QA + permuted control (W3)
│   ├── 08_compare_models.py    # stage 1: 15-config comparison (W4, resumable)
│   ├── 09_tune_champion.py     # stage 2: tune top-2, freeze thresholds (W4)
│   ├── 10_final_evaluation.py  # stage 3: ONE-SHOT frozen-test eval (W4)
│   ├── 11_error_analysis.py    # blatancy/skill gradients, FN-TP profile (W5)
│   ├── 12_calibrate_scores.py  # margins -> P(cheater), reliability (W5)
│   ├── 13_shap_analysis.py     # split-role SHAP + agreement (W6, resumable)
│   └── 14_robustness.py        # evasion sweep + legit sanity (W6, resumable)
├── reports/                # data_schema, eda_findings, feature_rationale,
│                           # literature_notes, week3_quality_split_imbalance,
│                           # methodology (draft)
├── notebooks/01_eda.ipynb  # narrative EDA
├── outputs/
│   ├── figures/            # EDA plots + model comparison / ROC-PR / confusion
│   ├── features/           # features.parquet, final_features.json, prune_log
│   ├── quality/            # instance_table, duplicate_groups, imbalance_check
│   ├── splits/             # splits.parquet (frozen), split_summary
│   ├── models/             # cv_comparison, oof/test scores, champion.* (W4)
│   └── analysis/           # error-analysis tables, index, calibration (W5)
└── data/                   # where to place the .npy arrays (see data/README.md)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Point the code at the data (default is `../archive`):

```bash
export CHEATDETECT_DATA="/path/to/archive"   # contains cheaters/ and legit/
```

## Run the pipeline

```bash
PYTHONPATH=src python scripts/00_inspect_data.py      # schema / reality check
PYTHONPATH=src python scripts/02_build_features.py    # -> outputs/features/features.parquet
PYTHONPATH=src python scripts/01_eda_figures.py       # -> outputs/figures/*.png
PYTHONPATH=src python scripts/03_check_features.py    # QA + leakage control
PYTHONPATH=src python scripts/04_data_quality.py      # raw dedup -> outputs/quality/
PYTHONPATH=src python scripts/05_make_split.py        # frozen split -> outputs/splits/
PYTHONPATH=src python scripts/06_finalize_features.py # -> final_features.json (34)
PYTHONPATH=src python scripts/07_check_imbalance.py   # SMOTE wiring QA + controls
PYTHONPATH=src python scripts/08_compare_models.py    # W4 stage 1 (re-run until done)
PYTHONPATH=src python scripts/09_tune_champion.py     # W4 stage 2 (re-run until done)
PYTHONPATH=src python scripts/10_final_evaluation.py  # W4 stage 3: one-shot test eval
PYTHONPATH=src python scripts/11_error_analysis.py    # W5: error gradients + profiles
PYTHONPATH=src python scripts/12_calibrate_scores.py  # W5: score calibration
PYTHONPATH=src python scripts/13_shap_analysis.py     # W6: SHAP (re-run until done)
PYTHONPATH=src python scripts/14_robustness.py        # W6: evasion sweep (re-run until done)
```

Scripts 08–09 checkpoint per (config, fold) under `outputs/models/.ckpt_*`
and resume if interrupted — repeat until they print completion.

## Roadmap (per the implementation plan)

- **W3** ✅ finalise features, SMOTE + class weights, stratified split (dedup-aware).
- **W4** ✅ RF / XGBoost / SVM baselines; precision/recall/F1, MCC, PR-AUC; ROC/PR.
- **W5** ✅ subtle-vs-blatant cheater analysis; false positives among skilled legit.
- **W6** ✅ SHAP interpretability; adversarial smoothing/jitter/delay robustness.
- **W7–8** demo, figures, report, presentation.
