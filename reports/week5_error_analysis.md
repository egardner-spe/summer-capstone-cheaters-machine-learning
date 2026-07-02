# Week 5 — Error Analysis: Who Gets Caught, Who Gets Missed, Who Gets Flagged

**Scope.** Deep-dive on the champion's errors using out-of-fold training
predictions at the Week-4 frozen thresholds; the spent test set appears only
in a descriptive confirmation at the end (no decisions were derived from it).
Plus light score calibration (margins → P(cheater)) for the triage framing.
Interpretability (SHAP) and adversarial robustness are Week 6.

## 0. The measuring stick: a mechanical-extremity index

The dataset has no subtle/blatant labels, so Week 5 needed an operational
one. For each instance we compute the average percentile rank — **relative to
the legit training distribution only** — on the eight shot-centric features
Week 2 identified as the discriminative core, each in the cheater-ward
direction (`error_analysis.INDEX_FEATURES`). Read on a cheater it is a
**blatancy index**; read on a legit player it is a **mechanical-skill index**.
That one number serves both planned analyses is not a convenience, it is the
point: *the more a legit player's mechanics resemble an aimbot's, the more a
behavioural detector is structurally tempted to flag them.*

Honesty caveat, stated up front: the index is defined by Week-2 EDA
directions, not by the champion's score, but the champion consumes these same
features among its 34 — independence is partial. The index answers "does
detection track *mechanical extremity*?", which is the meaningful form of the
subtle-vs-blatant question, not a tautology.

Sanity: legit-train mean 0.500 (by construction), cheater mean **0.592** —
the cheater population is only modestly extreme on average, which *is* the
subtlety finding of Week 2, now in one number.

## 1. Subtle vs blatant: detection is a monotone function of blatancy

OOF recall by cheater blatancy quartile (quartile cuts from training
cheaters; `outputs/analysis/recall_by_blatancy.csv`, fig11):

| quartile | @F1-max | @FPR≤1% |
|---|---|---|
| Q1 subtle | 0.26 | **0.008** (3 of 396) |
| Q2 | 0.36 | 0.048 |
| Q3 | 0.50 | 0.116 |
| Q4 blatant | 0.77 | 0.363 |

Two readings:

- The gradient is strong and monotone at both operating points — the model
  is doing exactly what the feature analysis predicted: catching mechanical
  extremity.
- **The subtle quartile is effectively invisible at the deployable
  threshold** (0.8% recall). A cheat that keeps its mechanical footprint
  inside Q1 — which commercial "humanised" cheats explicitly try to do —
  defeats this detector outright. That is the honest boundary of the method
  at this signal strength, and it sets up Week 6's adversarial question
  precisely: how much smoothing/jitter does it take to push a Q4 cheater's
  telemetry into Q1?

Test confirmation (train-derived cuts, strict threshold): 0.023 / 0.057 /
0.106 / 0.337 — the same staircase, no train-specific artifact.

## 2. False positives are the best legit players — quantified

OOF false positives by legit mechanical-skill decile
(`outputs/analysis/fpr_by_skill.csv`, fig12):

- **@FPR≤1%** (77 OOF FPs): the top skill decile alone carries **60%** of all
  false positives; the top two deciles carry **81%**. Per-decile FPR climbs
  from ~0.13% (D1–D3) to **5.7%** in D10 — a ~44× gradient.
- **@F1-max** (1,059 OOF FPs): same direction, softer concentration (top
  decile 26%, top two 41%).
- **Test set, descriptively**: the 26 real FPs from Week 4's one-shot
  evaluation have a **median skill percentile of 0.95**; 73% sit above P90.
  The Week-2 hypothesis — false positives will be skilled legit players — is
  confirmed about as cleanly as data can confirm it.

Deployment implication (this goes in the final report's discussion): a
threshold-ban policy at ANY operating point would systematically punish the
game's best players. The 4× triage lift from Week 4 is real, but the review
queue it produces is dominated by exactly the accounts where a wrong ban is
most damaging (visible, competitive, often streamed). Behavioural detection
at this signal level is a *prioritisation* layer for human or corroborating
review — e.g. combined with the information this dataset does not have:
wall-bang angles, spectator reports, hardware fingerprints.

One residual noted for completeness: at the loose F1-max threshold, D1 shows
a mild FPR uptick (~13%) — a small second FP population whose mechanics are
*atypically human* rather than aimbot-like. It vanishes at the strict
threshold; likely driven by the non-index features. Parked unless Week 7
polish time allows.

## 3. Missed vs caught cheaters: the missed ones look legit

Feature profile at the strict threshold
(`outputs/analysis/fn_tp_profile.csv`; legit means from Week 2 for
reference):

| feature | caught TP (n=212) | missed FN (n=1,373) | legit mean |
|---|---|---|---|
| frac_shots_locked | 0.583 | 0.364 | 0.331 |
| frac_shots_medspeed | 0.197 | 0.318 | 0.338 |
| speed_at_shot_mean | 0.407 | 0.609 | 0.654 |
| settle_ratio | 0.817 | 0.997 | 1.015 |
| blatancy index | 0.771 | 0.564 | 0.500 |

The missed cheaters sit far closer to the legit population than to the caught
cheaters on **every** axis — e.g. their settle ratio (0.997) is statistically
indistinguishable from legit (1.015), while caught cheaters "pre-settle"
hard (0.817). The false negatives are not a modelling failure to be tuned
away; they are cheaters whose aggregate telemetry is *genuinely* inside the
legit distribution. Catching them would require either finer-grained signals
(per-engagement sequences rather than 90-second aggregates) or corroborating
non-telemetry evidence.

## 4. Calibration: margins → P(cheater)

Platt and isotonic compared with fold-honest Brier on the persisted folds
(`outputs/analysis/calibration.json`, fig13): isotonic 0.1193, Platt 0.1194 —
a statistical tie; isotonic kept by the pre-stated rule (lower Brier), both
clearly beat the predict-base-rate baseline (0.1388). Applied to test
descriptively: Brier 0.1201, so the mapping transfers.

Useful translations for the report: the strict FPR≤1% threshold corresponds
to **P(cheater) ≈ 0.59**, the F1-max threshold to **P ≈ 0.24**. A review
queue sorted by calibrated probability can now say "players above 59%
likelihood" instead of quoting SVM margins. Saved as
`outputs/models/calibrator.joblib` (fit on OOF scores only).

## 5. Synthesis

Weeks 2–5 now tell one coherent story: the discriminative signal is
shot-centric mechanical extremity; the model converts it into a monotone
blatancy-detection gradient; its false positives are the mirror image of that
same gradient reflected onto the legit population's skill ceiling; and its
false negatives are cheaters who stay mechanically inside the human envelope.
Week 6 tests the two remaining pillars: does the model actually *rely* on the
shot-centric features (SHAP), and how cheaply can an adversary exploit the
Q1-invisibility result (perturbation robustness)?

## 6. Artifacts

```
outputs/analysis/recall_by_blatancy.csv   OOF recall per quartile x threshold
outputs/analysis/fpr_by_skill.csv         OOF FPR per skill decile x threshold
outputs/analysis/fn_tp_profile.csv        missed-vs-caught feature means
outputs/analysis/index_values.parquet     per-instance index (OOF + test)
outputs/analysis/test_confirmation.json   descriptive test-set checks
outputs/analysis/calibration.json         Brier comparison + P-mappings
outputs/models/calibrator.joblib          margin -> P(cheater), OOF-fit
outputs/figures/fig10..fig13              distributions, gradients, calibration
```

New module: `error_analysis.py`. New scripts: `11`–`12`. Test set: read
twice, descriptively, zero decisions derived from it.
