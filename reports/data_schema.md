# Data Reality Check & Schema (Week 1)

**Question this answers:** what is actually in the dataset, and how much of the
project is feature engineering? (The proposal flagged this as the decisive
Week-1 question.)

**Short answer:** the data is **raw per-tick view-angle telemetry**, not
pre-engineered features. There are no ready-made "aim speed" or "reaction time"
columns — those have to be *created*. So this project is feature-engineering-led:
the classifier in Week 4 will train on a table we derive, not on the arrays as
shipped.

## Files

| File | Shape | Dtype | On disk | Instances |
|------|-------|-------|---------|-----------|
| `archive/cheaters/cheaters.npy` | `(2000, 30, 192, 5)` | float32 | 220 MB | 2,000 cheater |
| `archive/legit/legit.npy` | `(10000, 30, 192, 5)` | float32 | 1.10 GB | 10,000 legit |

Class balance is **10,000 : 2,000 (5:1, 16.7% cheaters)** — the minority-class
problem the proposal anticipated. (Imbalance handling — SMOTE + class weights —
is Week 3, not done here.)

## The 4-D layout

```
instance  ->  30 engagements  ->  192 ticks  ->  5 channels
 (sample)      (sequences)        (~3.0 s @ 64-tick)
```

Each labelled **instance** is one classification unit. It contains 30
fixed-length **engagement** windows; each window is 192 server **ticks**
(192 / 64 ≈ 3.0 s of play); each tick has 5 **channels**. So one instance is
30 × 192 = 5,760 tick-observations. This is consistent with demo-extracted CS
telemetry (the "publicly available Counter-Strike dataset" in the proposal):
short windows of look/aim behaviour sampled at the tickrate.

## Channel semantics (verified, not assumed)

Statistics below are over an even sample of each file (`scripts/00_inspect_data.py`).

| idx | name | meaning | central range (p1–p99) | extremes | notes |
|-----|------|---------|------------------------|----------|-------|
| 0 | `d_yaw` | horizontal angular velocity (deg/tick) | −11 … +11 | ±180 | ~31% exactly 0 (no horizontal motion) |
| 1 | `d_pitch` | vertical angular velocity (deg/tick) | −1.7 … +1.7 | ±99 | ~40% exactly 0; smaller than yaw |
| 2 | `yaw` | absolute horizontal view angle (deg) | −130 … +132 | ±180 (wraps) | full compass heading |
| 3 | `pitch` | absolute vertical view angle (deg) | −25 … +20 | a few \|·\|>90 | mostly near horizon (~0) |
| 4 | `fire` | shot-fired flag | {0, 1} | — | =1 on ~5.2% of ticks |

**Why these labels (evidence):**

- Channels 0/1 are **angular velocity**. Correlation of channel 0 with the
  wrapped first-difference of channel 2 (yaw) is **≈ −0.93**, and channel 1 vs
  the first-difference of channel 3 (pitch) is **≈ −0.90**. The sign is negative
  (a convention choice in how the delta was stored), and the magnitude is high
  but not 1.0 — the stored absolute angles appear slightly smoothed/quantised
  relative to the velocity channels, so **0/1 (velocity) and 2/3 (placement) are
  complementary, not redundant.** Both are kept.
- Channel 2 is **yaw**: it spans the full ±180 with high spread (std ≈ 40) and
  wraps — that is a heading. Channel 3 is **pitch**: tight spread (std ≈ 7.5)
  centred just below 0 — players mostly hold the crosshair near the horizon.
  (A small number of \|pitch\|>90 values likely reflect weapon punch/recoil
  offset or wrap artefacts; they are rare and do not affect the magnitude-based
  features.)
- Channel 4 is a **fire flag**: binary, on ~5% of ticks, with ~10 shots per
  3-second engagement (max ~40) and *every* engagement containing at least one
  shot — i.e. these windows are built around shooting activity.

## Cleaning notes (what "messy" looks like here)

- **No NaNs or Infs** anywhere.
- **Padding:** ~0.04% (cheaters) / ~0.07% (legit) of ticks are fully zero —
  short engagements zero-padded out to 192 ticks. Negligible; treated as
  zero-velocity, no-fire ticks. `data_loading.valid_tick_mask()` exposes them if
  strict denominators are ever needed.
- **Exact-duplicate instances:** 4 in cheaters (0.20%) and 69 in legit (0.69%)
  are byte-identical to another instance. Harmless for EDA, but **they must be
  deduplicated or forced onto the same side of the Week-3 train/test split** to
  avoid train/test leakage. Flagged for Week 3.

## Implication for the project

Because the arrays are raw, the model is only as good as the features we build
from these five channels. The discriminative work happens in
`src/cheatdetect/features.py` (see `feature_rationale.md`), and the EDA
(`eda_findings.md`) shows the signal is subtle — which is the point of the
research question.
