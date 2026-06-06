"""Behavioural feature engineering.

Reduces each (30, 192, 5) instance to a fixed-length behavioural feature vector.

Design rationale (see reports/feature_rationale.md): the Week-2 EDA showed that
*global* aim aggregates barely separate the classes (univariate AUC ~0.50-0.60);
the discriminative signal concentrates (a) around firing events and (b) in the
shape/consistency of the aim-velocity distribution. Features are therefore
grouped into five families:

    A. speed distribution        D. shot-centric kinematics
    B. dynamics / smoothness     E. cross-engagement / shot consistency
    C. aim placement

All features are aggregated over the 30 engagements x 192 ticks of an instance,
so the output is one row per labelled instance -- ready for classical models
(RF / XGBoost / SVM) in Week 4.
"""
from __future__ import annotations
import numpy as np
from . import config as C

EPS = 1e-9


def _speed(block):
    """Angular speed per tick (deg/tick) from the velocity channels. -> (n,30,192)."""
    return np.sqrt(block[..., C.CH_DYAW] ** 2 + block[..., C.CH_DPITCH] ** 2)


def _roll_max_back(x, k):
    """For each tick t, the max of x over the k preceding ticks (t-1..t-k)."""
    out = np.full_like(x, -np.inf)
    for j in range(1, k + 1):
        shifted = np.full_like(x, -np.inf)
        shifted[..., j:] = x[..., :-j]
        out = np.maximum(out, shifted)
    out[~np.isfinite(out)] = 0.0
    return out


def _within_next(mask, k):
    """True at tick t if `mask` is True within the next 1..k ticks (pre-event window)."""
    out = np.zeros_like(mask)
    for j in range(1, k + 1):
        out[..., :-j] |= mask[..., j:]
    return out


def _within_prev(mask, k):
    """True at tick t if `mask` was True within the previous 1..k ticks (post-event window)."""
    out = np.zeros_like(mask)
    for j in range(1, k + 1):
        out[..., j:] |= mask[..., :-j]
    return out


def extract_chunk(block):
    """block: (n,30,192,5) float32 -> dict {feature_name: (n,) float32}."""
    n = block.shape[0]
    v = _speed(block)                         # (n,30,192) deg/tick
    dy = block[..., C.CH_DYAW]
    dp = block[..., C.CH_DPITCH]
    yaw = block[..., C.CH_YAW]
    pit = block[..., C.CH_PITCH]
    fire = block[..., C.CH_FIRE] > 0.5        # bool

    accel = np.abs(np.diff(v, axis=2))        # (n,30,191)
    jerk = np.abs(np.diff(accel, axis=2))     # (n,30,190)

    flat = lambda a: a.reshape(n, -1)
    vF = flat(v); dyF = flat(dy); dpF = flat(dp)
    pitF = flat(pit); fireF = flat(fire)
    accF = flat(accel); jerkF = flat(jerk)
    pct = lambda a, q: np.percentile(a, q, axis=1)

    f = {}

    # --- A. speed distribution ------------------------------------------------
    f["speed_mean"] = vF.mean(1)
    f["speed_std"] = vF.std(1)
    f["speed_median"] = np.median(vF, 1)
    f["speed_p90"] = pct(vF, 90)
    f["speed_p99"] = pct(vF, 99)
    f["speed_max"] = vF.max(1)
    f["speed_iqr"] = pct(vF, 75) - pct(vF, 25)
    f["frac_still"] = (vF < C.STILL_SPEED).mean(1)
    f["frac_slow"] = ((vF >= C.STILL_SPEED) & (vF < C.SLOW_SPEED)).mean(1)
    f["frac_med"] = ((vF >= C.SLOW_SPEED) & (vF < C.MED_SPEED)).mean(1)
    f["frac_fast"] = (vF >= C.MED_SPEED).mean(1)

    # --- B. dynamics / smoothness --------------------------------------------
    f["accel_absmean"] = accF.mean(1)
    f["accel_absstd"] = accF.std(1)
    f["jerk_absmean"] = jerkF.mean(1)
    f["jerk_absstd"] = jerkF.std(1)
    zcr = lambda a: (np.diff(np.sign(a), axis=2) != 0).reshape(n, -1).mean(1)
    f["zcr_dyaw"] = zcr(dy)        # micro-jitter / oscillation, horizontal
    f["zcr_dpitch"] = zcr(dp)      # micro-jitter / oscillation, vertical
    # spectral entropy of the speed signal (regularity); averaged over engagements
    P = np.abs(np.fft.rfft(v, axis=2)) ** 2
    P = P / (P.sum(axis=2, keepdims=True) + EPS)
    se = -(P * np.log(P + EPS)).sum(axis=2) / np.log(P.shape[2])
    f["spec_entropy"] = se.mean(1)

    # --- C. aim placement -----------------------------------------------------
    f["yaw_speed_mean"] = np.abs(dyF).mean(1)
    f["pitch_speed_mean"] = np.abs(dpF).mean(1)
    f["yaw_pitch_ratio"] = np.abs(dyF).mean(1) / (np.abs(dpF).mean(1) + EPS)
    f["pitch_abs_mean"] = np.abs(pitF).mean(1)
    f["pitch_abs_std"] = np.abs(pitF).std(1)
    f["pitch_abs_p95"] = pct(np.abs(pitF), 95)
    yaw_range = yaw.max(axis=2) - yaw.min(axis=2)   # per-engagement sweep proxy (n,30)
    f["yaw_range_mean"] = yaw_range.mean(1)

    # --- D. shot-centric ------------------------------------------------------
    nshots = fireF.sum(1).astype(np.float64)
    f["shot_rate"] = fireF.mean(1)
    f["shots_per_eng"] = fire.sum(axis=2).mean(1)
    at = np.where(fireF, vF, np.nan)                # aim speed on shot ticks
    f["speed_at_shot_mean"] = np.nanmean(at, 1)
    f["speed_at_shot_std"] = np.nanstd(at, 1)
    preF = flat(_within_next(fire, C.PRE_SHOT_TICKS))
    postF = flat(_within_prev(fire, C.POST_SHOT_TICKS))
    f["speed_pre_shot_mean"] = np.nanmean(np.where(preF, vF, np.nan), 1)
    f["speed_post_shot_mean"] = np.nanmean(np.where(postF, vF, np.nan), 1)
    f["settle_ratio"] = f["speed_at_shot_mean"] / (f["speed_pre_shot_mean"] + EPS)
    f["frac_shots_locked"] = (fireF & (vF < C.LOWSHOT_SPEED)).sum(1) / (nshots + EPS)
    f["frac_shots_medspeed"] = (
        (fireF & (vF >= C.SLOW_SPEED) & (vF < C.MED_SPEED)).sum(1) / (nshots + EPS))
    peak_pre = np.where(fireF, flat(_roll_max_back(v, C.PRE_SHOT_TICKS)), np.nan)
    f["prefire_peak_mean"] = np.nanmean(peak_pre, 1)        # flick height before shots
    f["overshoot_mean"] = np.nanmean(peak_pre - at, 1)      # flick-then-settle magnitude

    # --- E. cross-engagement / shot consistency -------------------------------
    eng_speed_mean = v.mean(axis=2)                 # (n,30)
    f["xeng_speed_std"] = eng_speed_mean.std(1)
    f["xeng_speed_cv"] = eng_speed_mean.std(1) / (eng_speed_mean.mean(1) + EPS)
    f["xshot_speed_cv"] = np.nanstd(at, 1) / (np.nanmean(at, 1) + EPS)

    # instances with no shots would yield NaN shot features; data has >=30 shots
    # per instance, but clean defensively anyway.
    for k in f:
        f[k] = np.nan_to_num(np.asarray(f[k], dtype=np.float64),
                             nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return f


def feature_names(block_like=None):
    """Deterministic feature order (matches extract_chunk insertion order)."""
    dummy = np.zeros((1, C.N_ENGAGEMENTS, C.WINDOW_TICKS, C.N_CHANNELS), np.float32)
    return list(extract_chunk(dummy).keys())


def extract_dataframe(block):
    import pandas as pd
    return pd.DataFrame(extract_chunk(block))
