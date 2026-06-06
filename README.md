# Machine-Learning Detection of Cheating in Online FPS Games

Senior cybersecurity capstone (CYBR 4950, Summer 2026) — Ethan Gardner.

Behaviour-based detection of aim-assist ("aimbot") cheating in Counter-Strike,
using **view-angle telemetry** instead of cheat-software signatures. The premise:
a player's aim and firing behaviour leave statistical fingerprints, and an ML
model can be trained to tell cheaters from legitimate (including highly skilled)
players.

> **Status — Weeks 1–2 complete.** Data reality check, EDA, and the behavioural
> feature pipeline are done; the model-ready feature matrix is built. Modelling,
> the stratified split, and imbalance handling (SMOTE + class weights) start in
> **Week 3+** and are intentionally **not** in this codebase yet.

## What the data is

Raw per-tick telemetry, two arrays (not committed — 1.4 GB):

```
instance -> 30 engagements -> 192 ticks (~3 s @ 64-tick) -> 5 channels
channels: [d_yaw, d_pitch, yaw, pitch, fire]
cheaters.npy = (2000, 30, 192, 5)   legit.npy = (10000, 30, 192, 5)
```

Channel meanings were verified empirically, not assumed — see
[`reports/data_schema.md`](reports/data_schema.md).

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
│   ├── config.py           #   paths, channel semantics, thresholds
│   ├── data_loading.py     #   mmap loading + light cleaning
│   └── features.py         #   (30,192,5) instance -> 39 behavioural features
├── scripts/
│   ├── 00_inspect_data.py  #   data reality check (W1)
│   ├── 01_eda_figures.py   #   EDA figures + univariate AUC (W2)
│   ├── 02_build_features.py#   build model-ready feature matrix (W2)
│   └── 03_check_features.py#   QA: finiteness, balance, leakage control
├── reports/                # data_schema, eda_findings, feature_rationale, literature_notes
├── notebooks/01_eda.ipynb  # narrative EDA
├── outputs/
│   ├── figures/            # EDA plots
│   └── features/           # features.parquet, feature_names.json, summaries
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
```

## Roadmap (per the implementation plan)

- **W3** finalise features, SMOTE + class weights, stratified split (dedup-aware).
- **W4** RF / XGBoost / SVM baselines; precision/recall/F1, MCC, PR-AUC; ROC/PR.
- **W5** subtle-vs-blatant cheater analysis; false positives among skilled legit.
- **W6** SHAP interpretability; adversarial smoothing/jitter/delay robustness.
- **W7–8** demo, figures, report, presentation.
