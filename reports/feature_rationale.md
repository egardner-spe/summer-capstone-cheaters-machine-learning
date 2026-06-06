# Feature Engineering — Rationale & Selection Process (Week 2)

This documents *why* each feature exists, what behaviour it targets, and the
evidence it carries signal. Implementation: `src/cheatdetect/features.py`.
Each instance `(30, 192, 5)` becomes one row of 39 features.

## Selection process (how this set was arrived at)

1. **Hypothesis-driven from the literature.** Behavioural anti-cheat work
   (VACnet, BotScreen, GAN-Aimbots; see `literature_notes.md`) consistently uses
   (a) derivatives of the aim signal — velocity, acceleration, jerk; (b) a
   **window around the shot** (mouse movement ~0.5 s before / ~0.25 s after
   firing); and (c) **consistency / regularity** measures, since bots are
   suspiciously repeatable. We built features in each of those families plus
   aim-placement descriptors.
2. **Broad then ranked.** Rather than pre-pruning, we engineered a wide set (39)
   and ranked by univariate AUC (`eda_findings.md`) and inspected collinearity
   (`fig6_feature_correlation.png`).
3. **Kept everything for now, on purpose.** Several features are individually
   weak (AUC ≈ 0.50). They stay in for Week 4 because tree ensembles exploit
   interactions, and **SHAP (Week 6)** is the principled place to prune. Features
   flagged as low-signal/redundant are noted below as pruning candidates.

The channels we build on (verified in `data_schema.md`): `d_yaw`, `d_pitch`
(angular velocity), `yaw`, `pitch` (absolute placement), `fire` (shot flag).
Derived per tick: `speed = sqrt(d_yaw² + d_pitch²)`, `accel = |Δspeed|`,
`jerk = |Δaccel|`.

## A. Aim-speed distribution
`speed_mean, speed_std, speed_median, speed_p90, speed_p99, speed_max,
speed_iqr, frac_still, frac_slow, frac_med, frac_fast`

The shape of how fast the crosshair moves. **Hypothesis:** humans spread across
still → slow tracking → fast flicks; an aimbot compresses this (less time at the
"correction" speeds). `frac_med` (share of time at 0.5–3 deg/tick) is the direct
analogue of the published *medium-velocity shots* signal. **Evidence:** the
fraction features and `speed_max` carry modest signal (AUC ~0.51–0.55); plain
means are near-useless (AUC ~0.51) — the reason we don't rely on them.

## B. Dynamics / smoothness
`accel_absmean, accel_absstd, jerk_absmean, jerk_absstd, zcr_dyaw, zcr_dpitch,
spec_entropy`

How smooth vs jerky the motion is. **Hypothesis:** software aim is either
*too* smooth (interpolated correction → low jerk) or carries a tell-tale
high-frequency jitter from humanisation. Zero-crossing rate of the velocity
channels captures micro-oscillation; spectral entropy captures regularity of the
speed signal over an engagement. **Evidence:** `zcr_dpitch` is a top-3 feature
(AUC 0.61) — cheaters show *more* vertical micro-oscillation, plausibly recoil
compensation or injected jitter — a good candidate to interrogate with SHAP.

## C. Aim placement
`yaw_speed_mean, pitch_speed_mean, yaw_pitch_ratio, pitch_abs_mean,
pitch_abs_std, pitch_abs_p95, yaw_range_mean`

Where and how widely the player looks. **Hypothesis:** crosshair discipline
(staying near the horizon, pre-aiming common angles) differs between skilled
play and assisted play. **Evidence:** weak on this dataset (AUC ~0.50–0.53);
`pitch_abs_mean` is essentially non-discriminative (0.50) → **pruning
candidate**. Kept for now as it may interact with shot features.

## D. Shot-centric kinematics (the discriminative core)
`shot_rate, shots_per_eng, speed_at_shot_mean, speed_at_shot_std,
speed_pre_shot_mean, speed_post_shot_mean, settle_ratio, frac_shots_locked,
frac_shots_medspeed, prefire_peak_mean, overshoot_mean`

Behaviour in the ±window around each `fire` tick (pre = 6 ticks ≈ 94 ms, post =
3 ticks ≈ 47 ms). **Hypothesis:** the human act of *acquiring* a target —
flick, brake, micro-correct, fire — leaves a kinematic fingerprint an aimbot
skips because it is already on target. `settle_ratio` (speed at shot ÷ speed
before shot) and `frac_shots_locked`/`frac_shots_medspeed` operationalise this.
**Evidence:** this family supplies the strongest features (AUC up to 0.63), all
pointing the same direction (cheaters fire pre-settled). This is the heart of the
model.

## E. Cross-engagement / shot consistency
`xeng_speed_std, xeng_speed_cv, xshot_speed_cv`

Variability *across* an instance's 30 engagements and across its shots.
**Hypothesis:** bots are eerily consistent; human performance fluctuates with
duels, positions, and fatigue. **Evidence:** weak univariately here (AUC ~0.51),
likely because these cheats are humanised; retained because consistency is a
known evasion-robust signal and may matter under the Week-6 adversarial tests.

## Known redundancy / pruning candidates (for SHAP in Week 6)
- Speed-distribution family is internally collinear (`fig6`); `speed_mean`,
  `speed_median`, `yaw_speed_mean` largely echo each other.
- `shot_rate` and `shots_per_eng` are near-duplicates by construction.
- Lowest-signal singletons: `pitch_abs_mean`, `frac_still`, `frac_fast`,
  `frac_med`, `speed_median`, `yaw_speed_mean` (AUC ≈ 0.50–0.51).

Nothing is dropped in Week 2 — selection is documented and deferred to the
interpretability stage, where feature removal can be justified by SHAP rather
than by a single-feature score.
