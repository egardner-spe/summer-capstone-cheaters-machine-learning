# Methodology (draft — begun Week 3)

> Draft status: sections 1–6 are substantive and reflect implemented,
> verified work (Weeks 1–3). Sections 7–9 are design commitments whose
> results land in Weeks 4–6; they are written as plans, not findings.
> This draft is written to be lifted into the final capstone report.

## 1. Problem statement and detection philosophy

Aim-assist ("aimbot") software gives a player superhuman target acquisition.
Signature-based anticheat detects the *software*; this project detects the
*behaviour* — statistical fingerprints in how the crosshair moves and when the
trigger is pulled. The behavioural approach matters precisely where signatures
fail: private or humanised cheats whose binaries are never seen twice, but
whose aim dynamics must still, at some point, do an inhuman thing.

The dataset makes this the hard version of the problem. Exploratory analysis
(Week 2) showed these are *subtle, humanised* cheaters: no single behavioural
statistic separates the classes (best univariate AUC 0.63), and on coarse
aggregates cheaters resemble skilled legitimate players. Detection must come
from combining many weak, shot-aligned signals — and the expected failure mode,
false positives on highly skilled legitimate players, is analysed explicitly
(Week 5).

## 2. Data

Two labelled arrays of per-tick view-angle telemetry from Counter-Strike
(64-tick):

```
instance = (30 engagements, 192 ticks ≈ 3 s, 5 channels)
channels = [d_yaw, d_pitch, yaw, pitch, fire]
cheaters.npy (2,000, 30, 192, 5)    legit.npy (10,000, 30, 192, 5)
```

Channel semantics were verified empirically rather than assumed (Week 1):
channels 0–1 are per-tick angular velocities consistent with the differenced
absolute angles (|corr| ≈ 0.93, sign convention flipped), channel 4 is a
binary fire flag (~5% of ticks). Zero-padding affects <0.1% of ticks and is
left in place. The unit of classification is the **instance** (one player's 30
engagements), not the engagement, giving each prediction ~90 seconds of
behavioural evidence.

Class imbalance is 5:1 legit:cheater, which is *optimistic* relative to
deployment (real cheating prevalence is far below 17%); consequences for
metric choice are handled in §7, and the limitation is acknowledged in the
final report.

## 3. Data quality and label noise (Week 3)

Every raw instance was content-hashed (blake2b over the raw float32 bytes)
across both arrays. Findings:

- 172 of 12,000 instances are exact byte-level duplicates, forming 84 groups.
  All 84 groups are raw-identical — repeated recordings, not feature
  collisions.
- **15 groups contain the same recording labelled both cheater and legit** —
  unambiguous label noise (30 rows, 0.25%), likely an artifact of the dataset
  being assembled from overlapping session pools.

Policy: mixed-label groups are removed entirely (identical inputs with
contradictory labels are unlearnable, and either copy would contaminate its
side of the split); same-label groups keep one copy. 103 rows dropped, 11,897
kept, imbalance unchanged (5.0:1). A per-instance audit table and per-group
log are persisted (`outputs/quality/`), so every removal is traceable.

## 4. Feature engineering (Weeks 2–3)

Each instance is reduced to behavioural features in five families (full
rationale: `reports/feature_rationale.md`):

A. **speed distribution** — shape of the aim-velocity distribution
   (percentiles, band occupancies), motivated by the published finding that
   cheaters under-occupy the medium-velocity band;
B. **dynamics / smoothness** — acceleration/jerk magnitudes, zero-crossing
   rates, spectral entropy: humanisation jitter and servo-like corrections
   live here;
C. **aim placement** — pitch discipline and yaw sweep;
D. **shot-centric kinematics** — aim speed at/before/after the trigger,
   settle ratio, locked-shot and medium-speed-shot fractions, pre-shot flick
   peak and overshoot. The EDA concentrated nearly all separability here:
   the aimbot is on target *before* the trigger is pulled;
E. **cross-engagement consistency** — variability of the above across the 30
   engagements (humans vary; assistance is consistent).

The final set was pruned 39 → 34 using **unsupervised, train-only** rules:
near-zero variance, and greedy removal within near-perfectly correlated pairs
(|r| > 0.95 on training rows; e.g. `shots_per_eng` is a deterministic
rescaling of `shot_rate`, r = 1.0). No label-aware selection is performed
outside cross-validation, by design. The prune is logged and reversible
(`outputs/features/prune_log.csv`, `final_features.json`).

## 5. Experimental design (Week 3)

A single stratified 80/20 split (seed 42) was made **after** deduplication and
persisted to disk (`outputs/splits/splits.parquet`, one row per original
instance including drop reasons):

- train 9,517 (7,932 legit / 1,585 cheater), test 2,380 (1,984 / 396);
- the **test set is frozen** — no Week 3–5 analysis reads it;
- **5-fold stratified CV assignments are fixed inside the training set and
  persisted**, so every model in Week 4 is tuned and compared on identical
  folds — differences between models cannot be fold luck;
- deduplication guarantees no identical feature vectors exist in the kept
  data (asserted programmatically), so no memorisation path crosses the
  train/test boundary.

## 6. Class imbalance handling (Week 3, compared in Week 4)

Four strategies are implemented behind one leakage-safe pipeline factory
(`cheatdetect.imbalance`): none, class weighting, SMOTE, SMOTE + class
weights. All preprocessing lives inside an `imblearn` Pipeline, so within any
CV loop standardisation is fit and SMOTE resamples **only on the training
portion of each fold** — synthetic instances cannot reach evaluation data by
construction. SMOTE operates in standardised space (its kNN interpolation
assumes comparable scales); XGBoost, lacking `class_weight`, uses
`scale_pos_weight` = 5.004.

Wiring was verified three ways (Week 3): a per-fold resample audit
(1,268 real → ~6,345 minority samples per training fold, all synthetics within
the real minority's feature range); a strategy plumbing test with a plain
logistic regression (ROC-AUC 0.712–0.715 across strategies, consistent with
the Week-2 smoke test — dedup and pruning cost no signal); and a
permuted-label control run *through* the SMOTE pipeline (0.511 ≈ chance,
confirming resampling does not leak). The near-identical plumbing scores also
set expectations honestly: for a linear model, ROC-AUC is largely insensitive
to resampling; the strategies are expected to differentiate on nonlinear
models and threshold-dependent metrics, and if they do not, that is itself a
reportable result.

## 7. Models and evaluation (Week 4)

A two-stage, pre-registered protocol. **Stage 1** compared logistic
regression, random forest, XGBoost, and RBF-SVM, each crossed with the four
imbalance strategies (15 configs; `xgb+smote_cw` excluded as provably
identical to `xgb+smote` at SMOTE parity), all scored out-of-fold on the
persisted folds. **Stage 2** grid-tuned only the top two families, and
selected the champion by a rule declared in advance: PR-AUC, then MCC at the
F1-max point, then fit cost. Because the 5:1 balance flatters accuracy-like
metrics and deployment prevalence is far lower, headline metrics are
**PR-AUC** and **MCC**, with ROC-AUC for comparability, plus
precision/recall/F1 at two operating points chosen on out-of-fold scores and
frozen before test evaluation: the F1-maximising threshold (balanced
characterisation) and a strict FPR ≤ 1% threshold (deployment-relevant, since
anticheat economics punish false accusations far more than misses).

Results (full tables in `outputs/models/`, discussion in
`reports/week4_modeling.md`): the anticipated strategy differentiation
appeared — **negatively**. SMOTE reduced PR-AUC for every nonlinear model
(e.g. SVM 0.432 → 0.371); with genuinely overlapping classes, synthetic
interpolation adds noise precisely in the overlap region where the boundary
must live. No resampling ("none") was the best strategy for all families.
The champion is an **RBF-SVM (C = 2, no resampling)**: CV PR-AUC 0.4345 ±
0.0287. The single frozen-test evaluation gave ROC-AUC 0.708 and PR-AUC 0.411
(chance 0.166) — a ~0.02 generalisation gap. At the strict FPR≤1% point the
model catches 13.1% of cheaters at 0.667 precision (a 4× lift over the base
rate), supporting a review-queue triage framing rather than automated
enforcement; the frozen threshold transferred to an actual test FPR of 1.31%,
a quantified limitation of finite-sample threshold calibration.

## 8. Error analysis (Week 5 — planned)

Subtle-vs-blatant cheater analysis (where does the model's confidence
concentrate?), and a focused study of false positives among highly skilled
legitimate players — the failure mode this dataset's humanised cheaters make
most likely. Inputs are already persisted: out-of-fold scores for every
config, champion OOF scores, and per-instance test scores.

## 9. Interpretability and robustness (Week 6 — planned)

SHAP attribution to verify the model leans on the shot-centric signals the EDA
predicts (a model that wins on artefacts is worse than a weaker honest one),
and adversarial-style robustness probes: temporal smoothing, added jitter, and
firing-delay perturbations that a humanised cheat could plausibly apply.

## 10. Reproducibility

One seed (42) drives the split, folds, and SMOTE. All decisions are persisted
as artifacts (split table with drop reasons, prune log, QA numbers in JSON)
rather than recomputed, and every pipeline stage is a numbered script over a
reusable package (`src/cheatdetect/`). Raw data is excluded from the
repository; everything derived is regenerable via scripts `00`–`07`.
