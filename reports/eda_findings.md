# Exploratory Data Analysis — Findings (Week 2)

**Method.** Even sample of 2,000 instances per class, reduced to the 39
behavioural features in `src/cheatdetect/features.py`, then ranked by univariate
separability (AUC = probability a random cheater scores above a random legit on
that single feature; reported as separating power, so 0.50 = useless). Figures
in `outputs/figures/`, table in `outputs/features/univariate_auc.csv`.

## Headline finding

**No single behavioural feature separates cheaters from legitimate players.**
The best univariate AUC is **0.63**; most features sit at 0.50–0.55. This is not
a failure of the features — it is the nature of *this* dataset. These are
**subtle / humanised cheaters**, not crosshair-teleporting bots, so on coarse
aggregates they look like skilled players. That is precisely the hard problem
the proposal set out to study, and it shapes everything downstream: detection
has to come from **combining many weak, shot-aligned signals**, and we should
expect false positives concentrated among skilled legit players (Week 5).

The signal that *does* exist is concentrated in two places: **around the moment
of firing**, and in the **shape of the aim-velocity distribution** — not in
overall means.

## Top discriminators (sample)

| Feature | Cheater | Legit | AUC | Reading |
|---|---|---|---|---|
| `frac_shots_locked` | 0.393 | 0.331 | **0.63** | cheaters more often fire with the crosshair already still (aim "locked") |
| `frac_shots_medspeed` | 0.301 | 0.338 | **0.62** | cheaters fire **fewer** mid-speed correction shots — the classic aimbot tell |
| `zcr_dpitch` | 0.239 | 0.224 | 0.61 | more vertical micro-oscillation (recoil-control / humanisation jitter?) |
| `speed_at_shot_mean` | 0.585 | 0.654 | 0.60 | lower aim speed at the instant of firing |
| `speed_at_shot_std` | 1.00 | 1.12 | 0.58 | *less variable* shot-time aim speed (more robotic) |
| `speed_pre_shot_mean` | 0.609 | 0.649 | 0.57 | less last-moment correction just before the shot |
| `settle_ratio` | 0.975 | 1.015 | 0.57 | aim already settled into the shot |
| `speed_max` | 48.2 | 57.0 | 0.55 | smaller peak flicks — they don't need to whip onto target |

Every one of these points the same way: **the aimbot is on target before the
trigger is pulled**, so the human signatures of *acquiring* a target (a fast
flick, then a braking correction, then a mid-speed micro-adjustment as you fire)
are muted. This is consistent with the published "medium-velocity shots are rare
for cheaters" result (see `literature_notes.md`).

## Figures

- `fig1_speed_distribution.png` — per-tick aim speed; distributions overlap
  almost completely (why global means fail).
- `fig2_shot_centric_speed.png` — aim speed **at** and **just before** the shot;
  cheaters shifted toward a low, sharp lock peak. *This is the money plot.*
- `fig3_medspeed_settle.png` — fraction of medium-velocity shots and the settle
  ratio; cheaters skew low on both.
- `fig4_smoothness_consistency.png` — jerk and cross-engagement speed CV.
- `fig5_univariate_auc.png` — separability ranking of all features.
- `fig6_feature_correlation.png` — collinearity map (the speed-distribution
  family is internally correlated; shot-centric and consistency families add
  comparatively independent information).

## Does it combine into signal? (QA, not Week-4 results)

As a **smoke test only** — real modelling, tuning, and metrics are Week 4 — a
plain logistic regression with 5-fold CV over the full 12,000-row matrix scores:

- ROC-AUC **0.716 ± 0.012**, PR-AUC **0.41** (baseline 0.167)
- permuted-label control **0.498** → no leakage

So the 39 weak features **do** combine into genuine multivariate signal (~0.72),
well above the 0.63 ceiling of any single feature. Tree ensembles (RF/XGBoost)
should meet or beat this in Week 4. The gap between strong multivariate signal
and weak univariate signal is the empirical case for the engineered, shot-centric
feature set.

## Data-quality items carried into Week 3

- Dedup the 73 exact-duplicate instances (or keep duplicates together at the
  split) to prevent train/test leakage.
- Padding (<0.1% of ticks) is benign and left as-is.
- The class imbalance (5:1) is untouched here by design — SMOTE + class weights
  are Week 3.
