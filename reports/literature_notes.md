# Literature Notes — Behavioural Cheat Detection (Week 1, ongoing)

Focused first pass to seed feature ideas and frame the research question. This is
a living document; depth increases through the project (the proposal treats the
review as ongoing).

## Where our approach sits

Mainstream behavioural anti-cheat treats a player's input/aim as a **time series**
and asks whether the *motion* looks human, independent of the cheat binary. That
is exactly our setup (view-angle telemetry → behavioural features → classifier).
Named systems for context:

- **VACnet** (Valve) — server-side deep model over CS player input/behaviour;
  the production proof-of-concept that aim behaviour alone is discriminative.
- **BotScreen** — RNN-based detector over input sequences.
- **VADNet** — vision/CNN approach.
- Multivariate time-series CNNs over mouse + keyboard input.

## Feature ideas pulled from the literature (and where we used them)

- **Derivatives of the aim signal** — velocity, acceleration, **jerk** (1st/2nd/
  3rd derivatives). → families A and B in `feature_rationale.md`.
- **A window around the shot** — feature vectors built from mouse movement
  ~**0.5 s before and ~0.25 s after** firing, optionally with hit/miss. We
  implement pre = 6 ticks (~94 ms) and post = 3 ticks (~47 ms); widening these
  toward the literature's 0.5 s/0.25 s is an easy follow-up. → family D.
- **Medium-velocity shots are rare for cheaters** — a striking reported result
  (cheaters disproportionately have ~0% mid-velocity shots vs natural players).
  Directly motivates `frac_shots_medspeed`, which is a top-2 feature for us. → D.
- **Recoil compensation** — aimbots counteract spray with fixed compensation
  curves, a detectable vertical-motion signature. Plausibly why `zcr_dpitch`
  ranks high for us. → family B.
- **Consistency / velocity-transition ratios** — bots show suspiciously low
  variance in correction curves; humans vary. → family E.

## Evasion / adversarial framing (sets up Week 6)

Modern cheats **humanise**: smoothing (interpolate the correction into a curve),
reaction **delay**, micro-**jitter**, and overshoot-then-correct. The "Aim Low,
Shoot High" line of work shows detectors that rely on blatant tells can be evaded
by mimicking human behaviour, and GAN-based aimbots generate adaptive,
human-like motion. Our dataset already behaves like this (subtle cheaters,
univariate AUC ≤ 0.63), which is why Week 6 deliberately re-tests detection after
applying smoothing/jitter/delay to simulate a more evasive cheat.

## Core challenge (the research question)

Every source converges on the same hard case: **distinguishing skilled
legitimate players from subtle cheats.** This justifies our imbalance-aware
metrics (MCC, PR-AUC) and the Week-5 stratified analysis of where false
positives fall among strong legit players.

## Sources

- [GAN-Aimbots: Using Machine Learning for Cheating in First Person Shooters (arXiv 2205.07060)](https://arxiv.org/pdf/2205.07060)
- [Aim Low, Shoot High: Evading Aimbot Detectors by Mimicking User Behavior (EuroSec 2020)](https://intellisec.de/research/aimbots/2020-eurosec.pdf) ([arXiv mirror](https://arxiv.org/pdf/2004.12183))
- [yviler/cs2-cheat-detection — detecting aimbot behaviour with NNs on engineered aim data from demos](https://github.com/yviler/cs2-cheat-detection)
- [Cheat Detection using Machine Learning within Counter-Strike (Wooster I.S.)](https://openworks.wooster.edu/cgi/viewcontent.cgi?article=11803&context=independentstudy)
- [Machine learning anti-cheating algorithm and a test against computer-vision aimbot](https://www.ewadirect.com/proceedings/ace/article/view/2574)
- [New usage of telemetry for anti-cheating in FPS games (ResearchGate)](https://www.researchgate.net/publication/377645285_New_usage_of_telemetry_for_anti-cheating_in_FPS_game)
- [Identify As A Human Does: Next-Generation Anti-Cheat Framework for FPS (arXiv 2409.14830)](https://arxiv.org/html/2409.14830v1)
- [Anti-Cheat in FPS Game for Aimbot Detection (IEEE Xplore)](https://ieeexplore.ieee.org/abstract/document/11007386/)
