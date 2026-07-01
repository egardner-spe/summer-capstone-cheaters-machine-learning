# Week 3 — Data Quality, Split Design, and Imbalance Handling

**Scope.** Finalize the feature set, resolve the duplicate problem flagged in
Week 2, build the train/test split the rest of the project will live on, and
wire up imbalance handling so Week-4 modelling can treat it as a hyperparameter.
No model training happens here — the classifiers below are wiring checks only.

## 1. The duplicate problem was bigger — and more interesting — than W2 thought

Week 2 counted 73 exact-duplicate instances checking each class separately.
Hashing every raw `(30,192,5)` instance across **both** arrays
(`scripts/04_data_quality.py`) tells the full story:

- **172 rows form 84 duplicate groups**, and every group is **byte-identical
  in the raw float32 data** — these are literal repeated recordings, not
  feature-space collisions. The feature pipeline loses nothing.
- **15 groups span both classes**: the same recording appears in
  `cheaters.npy` *and* `legit.npy`. That is outright **label noise** (30 rows,
  0.25% of the data) — an instance cannot be both. It also says something
  about provenance: the dataset was probably assembled from overlapping
  session pools, worth a sentence in the final report's limitations section.
- The remaining 69 same-label groups carry 73 extra copies — exactly the
  number W2 found. W2's check was right; it just never looked *across* the
  arrays.

**Policy applied** (`outputs/quality/duplicate_groups.csv` is the audit trail):
mixed-label groups are dropped entirely — identical inputs with contradictory
labels are unlearnable, and keeping either copy would contaminate whichever
side of the split received it. Same-label groups keep one copy. Net: **103
rows dropped, 11,897 kept** (9,916 legit / 1,981 cheater — the 5.0:1 imbalance
is essentially unchanged).

## 2. One split, made once, frozen

`scripts/05_make_split.py` → `outputs/splits/splits.parquet`, one row per
original instance, so every downstream question ("where did row 7312 go, and
why?") has a single answer file.

| subset | n | legit | cheater | ratio |
|---|---|---|---|---|
| train | 9,517 | 7,932 | 1,585 | 5.004 |
| test (frozen) | 2,380 | 1,984 | 396 | 5.010 |
| dropped | 103 | 84 | 19 | — |

Design choices, and why:

- **Stratified 80/20 holdout, seed 42.** The test set is written once and is
  untouched until final evaluation — nothing in Weeks 3–5 reads it except to
  count rows.
- **5-fold CV assignments are persisted now**, inside train (folds of
  1,903–1,904, each holding the 5.0:1 ratio). Every Week-4 model — RF,
  XGBoost, SVM, baselines — is tuned and compared on the *same* folds, so
  model differences can't be fold luck.
- **Dedup-awareness comes for free**: because the keep policy removes every
  duplicate copy, the kept matrix contains no identical feature vectors at
  all — asserted in `splitting.make_split()`, not assumed. No group-aware
  splitter needed.

`splitting.load_model_ready()` hands Week 4 `(X_train, y_train, X_test,
y_test, fold_ids)` so no later script ever re-derives any of this.

## 3. Feature set finalized: 39 → 34

Conservative, **unsupervised, train-only** pruning
(`scripts/06_finalize_features.py`; log in `outputs/features/prune_log.csv`).
No AUC filtering, no importance-based selection — the Week-2 EDA showed the
signal is spread across many weak features, so we remove only what is provably
redundant and let the models weigh the rest. Supervised selection either lives
inside the CV loop or it leaks; not worth it for 39 features.

Dropped, with the physical reading:

| dropped | kept instead | \|r\| | why it's redundant |
|---|---|---|---|
| `shots_per_eng` | `shot_rate` | 1.000 | deterministic rescaling (×192/30) |
| `speed_mean` | `yaw_speed_mean` | 0.995 | aim speed is yaw-dominated; the 2-D mean adds nothing |
| `accel_absmean` | `jerk_absmean` | 0.982 | at 64-tick granularity accel and jerk magnitudes track together |
| `accel_absstd` | `jerk_absstd` | 0.972 | same, second moment |
| `frac_fast` | `speed_p90` | 0.969 | both measure the fast tail of the speed distribution |

None of the Week-2 top discriminators (`frac_shots_locked`,
`frac_shots_medspeed`, `zcr_dpitch`, `speed_at_shot_*`, `settle_ratio`) were
touched. The full 39-column matrix is preserved; `final_features.json` is just
a list, so this decision is reversible.

## 4. Imbalance handling: wired, verified, deliberately not compared yet

`src/cheatdetect/imbalance.py` exposes one factory,
`make_pipeline(estimator, strategy)`, with four strategies: `none`,
`class_weight`, `smote`, `smote_cw`. Week 4 treats the strategy as a
hyperparameter instead of committing blind. Details that matter:

- Everything is an `imblearn` Pipeline: inside CV, scaling is fit and SMOTE
  resamples **only on each fold's training portion**. Synthetic rows cannot
  reach a validation fold or the frozen test set *by construction*.
- SMOTE runs **after** standardisation — it interpolates between kNN
  neighbours, and those distances belong in the standardised space.
- Honest caveat, recorded before Week 4 tempts us to forget it: at full
  parity, SMOTE makes `class_weight="balanced"` a near no-op (post-resampling
  weights ≈ 1), so `smote_cw` only meaningfully differs when
  `sampling_strategy < 1`. And XGBoost has no `class_weight` at all — use
  `imbalance.scale_pos_weight(y_train)` = **5.004**.

**Verification** (`scripts/07_check_imbalance.py` →
`outputs/quality/imbalance_check.json`):

1. **Per-fold resample audit** — each fold's training portion holds 1,268 real
   cheaters → 6,345–6,346 after SMOTE (+~5,077 synthetic); every synthetic
   sample stays inside the real minority's per-feature range (checked, not
   assumed — SMOTE interpolates, so escape would mean a bug).
2. **Plumbing smoke test** (LogReg on the persisted folds, QA only):
   ROC-AUC 0.712–0.715 across all four strategies, PR-AUC ≈ 0.41 (chance
   0.167). Two readings: (a) the pipeline runs end-to-end and matches the
   Week-2 smoke test (0.716 on 39 features / 12,000 rows vs 0.714 on 34
   features / 9,517 deduped rows — **dedup and pruning cost nothing**);
   (b) for a *linear* model, ROC-AUC barely moves with resampling — expected,
   since ROC-AUC is insensitive to class priors and monotone score shifts.
   The strategies should differentiate on trees/SVM and on
   threshold-dependent metrics (MCC, recall at fixed FPR) in Week 4. If they
   don't, that's a finding, not a failure.
3. **Permuted-label control through the SMOTE pipeline: 0.511** — the
   strongest wiring check. If resampling leaked into validation folds,
   permuted labels would score above chance.

## 5. Artifacts produced this week

```
outputs/quality/instance_table.parquet   per-instance identity, raw hash, dup groups, keep flag
outputs/quality/duplicate_groups.csv     per-group audit (size, labels, raw-identical, action)
outputs/quality/imbalance_check.json     all QA numbers cited above
outputs/splits/splits.parquet            frozen split + persisted CV folds
outputs/splits/split_summary.csv         balance table
outputs/features/final_features.json     the 34 features Week 4 trains on
outputs/features/prune_log.csv           what was dropped and why
```

New modules: `data_quality.py`, `splitting.py`, `feature_selection.py`,
`imbalance.py`. New scripts: `04`–`07`. Methodology draft begun in
`reports/methodology.md`.

**Week 4 picks up here**: RF / XGBoost / SVM on `load_model_ready()`, strategy
as a hyperparameter, metrics incl. MCC and PR-AUC, all comparisons on the
persisted folds. The frozen test set stays frozen until the end.
