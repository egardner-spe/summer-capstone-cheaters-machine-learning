# Week 4 — Model Comparison, Champion Selection, and the One-Shot Test Evaluation

**Scope.** First real modelling week: 15 model × imbalance-strategy configs
compared out-of-fold on the persisted Week-3 folds (stage 1), light tuning on
the top two families (stage 2), champion refit and the **single** frozen-test
evaluation (stage 3). Thresholds were frozen from out-of-fold scores *before*
the test set was read. Error analysis is Week 5; interpretability and
robustness are Week 6.

## 1. Stage 1 — the imbalance-strategy result nobody advertises

All 15 configs (logreg / RF / XGBoost / SVM-RBF × none / class-weight / SMOTE
/ SMOTE+CW, minus `xgb+smote_cw` which is provably identical to `xgb+smote`
at SMOTE parity), each scored out-of-fold on the same 5 persisted folds.
Full table: `outputs/models/cv_comparison.csv`; figure:
`outputs/figures/fig7_cv_model_comparison.png`.

| config | PR-AUC | ROC-AUC | MCC@F1max | recall@FPR≤1% |
|---|---|---|---|---|
| **svm + none** | **0.432** | 0.728 | 0.319 | 0.127 |
| rf + none | 0.426 | 0.731 | 0.315 | 0.119 |
| svm + class_weight | 0.425 | **0.745** | **0.326** | 0.103 |
| xgb + none | 0.419 | 0.730 | 0.306 | 0.113 |
| logreg + none | 0.419 | 0.714 | 0.295 | 0.126 |
| … | | | | |
| xgb + smote | 0.388 | 0.703 | 0.265 | 0.100 |
| rf + smote(_cw) | 0.383 | 0.718 | 0.267 | 0.079 |
| svm + smote(_cw) | 0.371 | 0.709 | 0.257 | 0.076 |

Three findings, in decreasing order of importance:

1. **SMOTE actively hurts every nonlinear model** — RF 0.426→0.383, SVM
   0.432→0.371, XGBoost 0.419→0.388 PR-AUC. The Week-2 EDA explains why:
   these are subtle, humanised cheaters whose feature distributions genuinely
   overlap the legit class. SMOTE interpolates synthetic minority points
   between real cheater neighbours — i.e. **directly into the overlap
   region**, exactly where the decision boundary has to live. It doesn't add
   information; it adds noise where the problem is hardest. This is a
   reportable negative result, and treating the strategy as a hyperparameter
   (rather than committing to SMOTE up front, as the original plan implied)
   is what surfaced it.
2. **Doing nothing is a strong default.** `none` is the best strategy for
   every model family on PR-AUC. Class weights buy the single best ROC-AUC
   and MCC (svm+class_weight) but cost PR-AUC — they push the boundary toward
   recall at the expense of the precision-sensitive metric this project
   headlines. Neither headline metric (PR-AUC, MCC) rewards resampling here.
3. **The parity prediction verified itself**: `smote` and `smote_cw` produced
   *identical* scores for logreg, RF, and SVM (balanced weights are a no-op
   after SMOTE brings 1:1 parity) — empirical confirmation of the argument
   used to exclude `xgb+smote_cw` from the matrix.

Model families are packed tightly (0.419–0.432 for the four `none` configs):
consistent with Week 2's conclusion that the ceiling is set by the **feature
signal**, not model capacity.

## 2. Stage 2 — tuning is a rounding error (as expected)

Light grids on the top two families (svm+none: 4 configs, rf+none: 10
configs), same folds, same machinery (`outputs/models/tuning_results.csv`):

- **Champion: SVM-RBF, C = 2.0, no resampling** — CV PR-AUC **0.4345 ±
  0.0287**, ROC-AUC 0.727, MCC@F1max 0.320.
- Tuning gained +0.003 PR-AUC over the stage-1 default; RF's best grid point
  gained +0.0005. The landscape is flat — more grid would polish decimals,
  not change conclusions (this is why stage 2 was kept light).
- Selection followed the pre-registered rule (PR-AUC, then MCC, then fit
  cost). Note for honesty: `svm C=0.5` had a slightly better MCC@F1max
  (0.327 vs 0.320) — the rule picked PR-AUC first, as declared *before*
  results existed.

Both operating thresholds were computed from the champion's out-of-fold
scores and frozen into `champion.json` at this point — before any test read.

## 3. Stage 3 — the test set is spent

Champion refit on all 9,517 training rows, evaluated exactly once on the
frozen 2,380-row test set (`outputs/models/test_results.json`, figures
`fig8_test_roc_pr.png`, `fig9_test_confusion.png`):

| metric | CV (OOF) | test |
|---|---|---|
| ROC-AUC | 0.727 | **0.708** |
| PR-AUC (chance 0.166) | 0.435 | **0.411** |

A ~0.02 generalisation gap — the CV protocol was honest; no evidence of
overfitting to fold structure.

At the two **frozen** operating points:

| | precision | recall | F1 | MCC | FPR | confusion (tp/fp/fn/tn) |
|---|---|---|---|---|---|---|
| @F1-max | 0.427 | 0.439 | 0.433 | 0.319 | 11.7% | 174 / 233 / 222 / 1751 |
| @FPR≤1% | 0.667 | 0.131 | 0.219 | 0.247 | 1.31% | 52 / 26 / 344 / 1958 |

How to read this honestly:

- **The strict point is the deployment-relevant one.** At a ~1% FPR the
  system flags 78 players and is right about 52 of them: precision 0.667
  against a 16.6% base rate — a **4× lift over random review**. As an
  auto-ban trigger this is nowhere near acceptable (1 in 3 accusations
  wrong); as a **review-queue prioritiser** it is already useful. That
  framing — triage, not verdict — matches how commercial anticheat teams
  actually use behavioural signals.
- **Threshold transfer slipped slightly**: the FPR≤1% threshold was chosen on
  out-of-fold scores and landed at 1.31% on test. Expected behaviour with a
  finite calibration sample; worth one sentence in the final report's
  limitations.
- **The F1-max point is a monitoring point, not an enforcement point** — 11.7%
  FPR would flag ~232 innocent players per 2,000. It exists to characterise
  the balanced trade-off, not to be deployed.
- Champion scores are SVM decision-function margins, not calibrated
  probabilities. If Week 5+ needs probability statements ("this player is 90%
  likely cheating"), add isotonic/Platt calibration fitted on OOF scores —
  flagged now so it isn't invented after seeing test data.

## 4. A decision point flagged for Week 6

SHAP interpretability on an RBF-SVM requires `KernelExplainer` —
model-agnostic, slow, and approximate. The runner-up RF (PR-AUC 0.426, gap
of 0.009) supports fast, exact `TreeExplainer`. If Week 6's interpretability
analysis is central to the report's argument, running SHAP on the RF
runner-up (clearly labelled as such) alongside — or instead of — kernel SHAP
on the champion is a defensible choice. Deferring that decision to Week 6;
both models' full CV artifacts are persisted.

## 5. Artifacts produced this week

```
outputs/models/cv_comparison.csv      stage-1 ranking (15 configs)
outputs/models/per_fold_metrics.csv   fold-level AUCs (variance audit)
outputs/models/oof_scores.parquet     OOF score per train row per config (W5)
outputs/models/tuning_results.csv     stage-2 grids (14 configs)
outputs/models/champion.json          config + params + FROZEN thresholds
outputs/models/champion_oof.parquet   champion OOF scores (W5)
outputs/models/champion.joblib        fitted pipeline (W5/W6 reuse)
outputs/models/test_results.json      the one-shot test numbers
outputs/models/test_scores.parquet    per-test-row scores (W5 error analysis)
outputs/figures/fig7..fig9            comparison, ROC/PR, confusions
```

New modules: `modeling.py` (model zoo + grids), `evaluation.py` (OOF
machinery, op-point selection, resumable checkpointing). New scripts:
`08`–`10`. Scripts 08/09 are **resumable** — they checkpoint per
(config, fold) unit and can be re-run until complete; delete
`outputs/models/.ckpt_stage*/` after completion if present (the sandbox that
produced them cannot delete files).

**Week 5 picks up here**: error analysis on `oof_scores.parquet` +
`test_scores.parquet` — subtle-vs-blatant score distribution, and who the 26
false positives at the strict operating point actually are (the
skilled-legit hypothesis from Week 2).
