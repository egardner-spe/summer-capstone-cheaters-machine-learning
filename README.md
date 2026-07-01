# Machine-Learning Detection of Cheating in Online FPS Games

Senior cybersecurity capstone (CYBR 4950, Summer 2026) — Ethan Gardner.

Behaviour-based detection of aim-assist ("aimbot") cheating in Counter-Strike,
using **view-angle telemetry** instead of cheat-software signatures. The premise:
a player's aim and firing behaviour leave statistical fingerprints, and an ML
model can be trained to tell cheaters from legitimate (including highly skilled)
players.

> **Status — Weeks 1–3 complete.** Data reality check, EDA, feature pipeline,
> data-quality pass (dedup + label-noise removal), frozen 80/20 split with
> persisted CV folds, finalized 34-feature set, and leakage-safe imbalance
> handling (SMOTE / class weights) are done. Model training (RF/XGBoost/SVM)
> starts in **Week 4** and is intentionally **not** in this codebase yet.

## What the data is

Raw per-tick telemetry, two arrays (not committed — 1.4 GB):

```
instance -> 30 engagements -> 192 ticks (~3 s @ 64-tick) -> 5 channels
channels: [d_yaw, d_pitch, yaw, pitch, fire]
cheaters.npy = (2000, 30, 192, 5)   legit.npy = (10000, 30, 192, 5)
```

Channel meanings were verified empirically, not assumed — see
[`reports/data_schema.md`](reports/data_schema.md).

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
│   └── imbalance.py        #   SMOTE / class-weight pipeline factory (W3)
├── scripts/
│   ├── 00_inspect_data.py  #   data reality check (W1)
│   ├── 01_eda_figures.py   #   EDA figures + univariate AUC (W2)
│   ├── 02_build_features.py#   build model-ready feature matrix (W2)
│   ├── 03_check_features.py#   QA: finiteness, balance, leakage control (W2)
│   ├── 04_data_quality.py  #   raw dedup verification + keep flags (W3)
│   ├── 05_make_split.py    #   frozen 80/20 split + persisted folds (W3)
│   ├── 06_finalize_features.py # 39 -> 34 prune, train-only stats (W3)
│   └── 07_check_imbalance.py   # SMOTE wiring QA + permuted control (W3)
├── reports/                # data_schema, eda_findings, feature_rationale,
│                           # literature_notes, week3_quality_split_imbalance,
│                           # methodology (draft)
├── notebooks/01_eda.ipynb  # narrative EDA
├── outputs/
│   ├── figures/            # EDA plots
│   ├── features/           # features.parquet, final_features.json, prune_log
│   ├── quality/            # instance_table, duplicate_groups, imbalance_check
│   └── splits/             # splits.parquet (frozen), split_summary
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
```

## Roadmap (per the implementation plan)

- **W3** ✅ finalise features, SMOTE + class weights, stratified split (dedup-aware).
- **W4** RF / XGBoost / SVM baselines; precision/recall/F1, MCC, PR-AUC; ROC/PR.
- **W5** subtle-vs-blatant cheater analysis; false positives among skilled legit.
- **W6** SHAP interpretability; adversarial smoothing/jitter/delay robustness.
- **W7–8** demo, figures, report, presentation.
