# Week 6 — Interpretability (SHAP) and Adversarial Robustness

**Scope.** The two remaining analytical pillars: does the champion actually
*rely* on the shot-centric signals the EDA predicted (SHAP, split-role
design), and how cheaply can an adversary evade it (raw-telemetry
perturbations against the frozen champion at the frozen thresholds)? Model
and thresholds stay frozen throughout — this is evaluation, not tuning.

## 1. SHAP — split-role design

The champion RBF-SVM only supports the slow, approximate KernelExplainer, so
roles were split: **exact TreeExplainer on the RF runner-up** (same `none`
strategy, PR-AUC 0.426 vs 0.435; refit unscaled after asserting equivalence —
piped 0.7303 vs unscaled 0.7303 test AUC) carries the detailed story over the
full test set, and a **checkpointed KernelExplainer sample** (120 balanced
test instances, 25-centroid background, 500 coalitions) tests whether the
actual champion ranks features the same way.

### 1a. The global story (RF, exact, full test — fig14)

Top of the mean|SHAP| ranking: `frac_shots_locked` (0.032, ~1.5× the #2),
then `zcr_dpitch`, `speed_iqr`, `speed_at_shot_std`, `speed_at_shot_mean`,
`shot_rate`, `settle_ratio`, `frac_shots_medspeed`. Every direction in the
beeswarm matches the Week-2 EDA: high locked-shot fraction, high vertical
micro-oscillation, low shot-time speed and low settle ratio push toward
"cheater". The model is winning for the predicted reasons — no artifact
shortcut appeared.

### 1b. What flags the false positives

Signed group attributions at the strict threshold
(`shap_group_attribution.csv`): the 26 test FPs are pushed over the line by
`frac_shots_locked` (+0.135 — by far the largest group attribution anywhere
in the table), then `frac_shots_medspeed`, `zcr_dpitch`, `speed_at_shot_mean`,
`settle_ratio`. That is *precisely the aimbot signature*. Week 5 showed FPs
are the most mechanically skilled legit players; SHAP now shows they are
flagged **for that skill** — the false-positive problem is not a fixable bug
but the structural cost of this feature space.

### 1c. Does the champion agree? Partially — and honestly reported

Rank agreement on the identical 120-instance sample: Spearman ρ = **0.39**
(p = 0.024), top-8 overlap **3/8** (fig15). Both models put
`frac_shots_locked` first and share `zcr_dpitch` and `speed_iqr`; beyond
that, the SVM spreads attribution across the smoothness/dynamics family
(`jerk_absmean`, `jerk_absstd`, `zcr_dyaw`, `speed_median`) where the RF
concentrates on the shot-centric family (`settle_ratio`, `speed_at_shot_*`,
`frac_shots_medspeed`). Same headline, different supporting cast. Plausible
contributors, in honest order: genuinely different geometry (an RBF margin
can exploit the correlated smoothness family that axis-aligned trees ignore —
possibly the source of the SVM's PR-AUC edge), SHAP's known credit-splitting
under feature correlation, and KernelExplainer approximation noise. The
robustness results below independently corroborate that the champion leans on
the smoothness family.

## 2. Adversarial robustness — evasion is cheap, and we know exactly why

Three humaniser-style perturbations applied to the **raw telemetry** of the
396 test cheaters (channel conventions and padding preserved — see
`robustness.py`), re-extracted through the real feature pipeline, scored by
the frozen champion (`robustness.csv`, fig16):

| perturbation | strict recall (base 0.131) | F1-max recall (base 0.439) | blatancy index (base 0.589) |
|---|---|---|---|
| smooth α=0.7 (weak) | **0.015** | 0.654 | 0.526 |
| smooth α=0.3 (strong) | **0.005** | 0.914 | 0.593 |
| jitter σ=0.05 | 0.177 ↑ | 0.694 | 0.625 ↑ |
| jitter σ=0.20 | 0.086 | 0.995 | 0.501 |
| delay k=1..4 | 0.131→0.106 | 0.44→0.48 | 0.579→0.501 |

Findings, in order of consequence:

1. **Mild smoothing is a near-free evasion of the deployed threshold.** A
   barely-perceptible causal EMA (α = 0.7) collapses strict-threshold recall
   from 13.1% to **1.5%** — an 88% reduction. The mechanism is identified,
   not guessed: that smoothing pass leaves the shot-centric features almost
   untouched (`frac_shots_locked` 0.390 → 0.354, `settle_ratio` ~unchanged)
   but obliterates the micro-oscillation family (`zcr_dpitch` 0.238 → 0.074,
   −69%; `zcr_dyaw` −67%). The detector, especially the SVM (§1c), partly
   keys on **the cheat's own humanisation jitter** — and a smoothing pass
   strips exactly that. One step of the arms race, quantified: the
   humaniser's countermeasure (jitter) is detectable; the
   counter-countermeasure (smoothing the jitter) defeats detection.
2. **Jitter is counterproductive for the evader.** Weak jitter *raises*
   strict-threshold recall (0.131 → 0.177) — it amplifies the very zcr/jerk
   signature the model rewards. Even heavy jitter only reaches 0.086.
3. **Firing delay is nearly useless as evasion** (0.131 → 0.106 at four
   ticks): our features measure the state *at* the shot, and a delayed shot
   is still a locked shot. Notably, delay lowers the blatancy *index* far
   more than the *score* — the model sees through the proxy.
4. **The F1-max threshold fails in the opposite direction.** Applying the
   same moderate perturbations to the 1,984 test *legit* players leaves
   strict-threshold FPR sane (0.2–1.7%) but explodes F1-max FPR: smoothing
   0.117 → **0.540**, jitter → **0.754**. Any innocent player whose input
   chain resembles a smoothing/jitter transform (controller aim-assist,
   driver-level mouse smoothing, high-DPI interpolation) would be flagged
   wholesale at the loose threshold. This independently re-confirms Week 4's
   verdict: the F1-max point is characterisation only, never enforcement.

**Stated limitation** (also in `robustness.py`): with no outcome labels we
cannot measure how much each perturbation degrades the cheat's actual
effectiveness. These curves are an upper bound on evasion cheapness — though
a causal EMA at α = 0.7 adds only ~1–2 ticks of effective aim lag, so the
smoothing attack in particular is unlikely to cost the cheater much.

## 3. What Weeks 2–6 now add up to

The signal is shot-centric mechanical extremity plus humaniser jitter; the
champion uses both (SHAP); detection rate is a monotone function of the first
(W5); and the second can be stripped by an adversary for near-zero cost (W6).
Behavioural aim-telemetry detection at this aggregation level is therefore a
**triage layer with a quantified evasion ceiling**, not a standalone
anticheat — the honest, defensible thesis the final report can argue with
every number already in hand.

## 4. Artifacts

```
outputs/analysis/shap_ranking_rf.csv         mean|SHAP| all 34 (exact, full test)
outputs/analysis/shap_group_attribution.csv  signed SHAP: TP/FN/FP/TN groups
outputs/analysis/shap_agreement.json         RF-vs-SVM rank agreement
outputs/analysis/robustness.csv              full sweep incl. legit sanity
outputs/models/rf_runnerup.joblib
outputs/figures/fig14_shap_beeswarm_rf.png
outputs/figures/fig15_shap_agreement.png
outputs/figures/fig16_evasion_curves.png
```

New modules: `interpretability.py`, `robustness.py`. New scripts: `13`–`14`
(both resumable). Test set: evaluated descriptively under fixed
perturbations; no decisions fed back into any model or threshold.

**Week 7–8 picks up here**: demo, final figures, report assembly,
presentation. All quantitative results are complete.
