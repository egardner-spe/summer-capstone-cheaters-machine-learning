"""Adversarial robustness machinery (Week 6): raw-telemetry perturbations.

Simulates what a humanised cheat could plausibly do to its own mouse stream,
at the RAW telemetry level -- perturbed (30,192,5) blocks are pushed back
through the real feature pipeline, so every downstream feature responds
exactly as it would in deployment:

    smooth : causal EMA over the view-angle streams (a 'smoothing' humaniser).
             alpha = weight on the new sample; 1.0 = identity, lower = heavier.
    jitter : zero-mean Gaussian noise added to angular velocity, integrated
             into the absolute angles (a 'jitter' humaniser).
    delay  : fire flags shifted k ticks later (firing-delay humaniser).

Consistency rules, enforced not assumed:
  * channel convention ch0/1 ~ -delta(ch2/3) is preserved: smoothing is the
    same linear filter on velocities and (unwrapped) angles, so the
    difference relation commutes; jitter added to velocity is subtracted
    cumulatively from angles;
  * yaw is unwrapped (period 360) before filtering and re-wrapped after;
  * zero-padded ticks stay exactly zero (filters are causal, so padding at
    the tail never bleeds backward into valid ticks); fire flags shifted
    onto padding are DROPPED, not kept (a fire on a dead tick would create
    a fake 'perfectly locked shot' artifact).

Limitation stated once, honestly: we cannot measure how much each
perturbation degrades the cheat's actual effectiveness (no outcome labels).
These curves are therefore an UPPER BOUND on evasion cheapness -- a real
cheat must stay useful while perturbing; we only require it to stay plausible.
"""
from __future__ import annotations

import numpy as np

from . import config as C


def valid_mask(block: np.ndarray) -> np.ndarray:
    """(n,30,192) True where the tick carries data (any channel nonzero)."""
    return ~np.all(block == 0.0, axis=-1)


def _rewrap(yaw: np.ndarray) -> np.ndarray:
    return (yaw + 180.0) % 360.0 - 180.0


def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    """Causal EMA along the tick axis (axis=2); y[0] = x[0]."""
    y = np.empty_like(x)
    y[..., 0] = x[..., 0]
    for t in range(1, x.shape[-1]):
        y[..., t] = alpha * x[..., t] + (1.0 - alpha) * y[..., t - 1]
    return y


def smooth(block: np.ndarray, alpha: float) -> np.ndarray:
    """EMA-smooth velocities and angles with the same linear filter."""
    out = np.array(block, dtype=np.float32, copy=True)
    m = valid_mask(block)
    for ch in (C.CH_DYAW, C.CH_DPITCH, C.CH_PITCH):
        out[..., ch] = _ema(block[..., ch], alpha)
    yaw_u = np.unwrap(block[..., C.CH_YAW], axis=2, period=360.0)
    out[..., C.CH_YAW] = _rewrap(_ema(yaw_u, alpha))
    out[~m] = 0.0                      # padding stays exactly zero
    return out


def jitter(block: np.ndarray, sigma: float, seed: int = C.RANDOM_SEED
           ) -> np.ndarray:
    """Gaussian velocity noise, integrated into the absolute angles."""
    out = np.array(block, dtype=np.float32, copy=True)
    m = valid_mask(block)
    rng = np.random.default_rng(seed)
    n_yaw = rng.normal(0, sigma, block.shape[:-1]).astype(np.float32) * m
    n_pit = rng.normal(0, sigma, block.shape[:-1]).astype(np.float32) * m
    out[..., C.CH_DYAW] += n_yaw
    out[..., C.CH_DPITCH] += n_pit
    # ch0 ~ -delta(ch2): +noise on velocity accumulates as -cumsum on angle
    out[..., C.CH_YAW] = _rewrap(block[..., C.CH_YAW]
                                 - np.cumsum(n_yaw, axis=2))
    out[..., C.CH_PITCH] = block[..., C.CH_PITCH] - np.cumsum(n_pit, axis=2)
    out[~m] = 0.0
    return out


def fire_delay(block: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Shift fire flags k ticks later; fires landing on padding are dropped.

    Returns (perturbed block, fraction of fires dropped)."""
    out = np.array(block, dtype=np.float32, copy=True)
    fire = block[..., C.CH_FIRE] > 0.5
    shifted = np.zeros_like(fire)
    shifted[..., k:] = fire[..., :-k]
    kept = shifted & valid_mask(block)
    n0, n1 = int(fire.sum()), int(kept.sum())
    out[..., C.CH_FIRE] = kept.astype(np.float32)
    return out, (n0 - n1) / max(n0, 1)


PERTURBATIONS = {
    "none": lambda b: (b, 0.0),
    "smooth_0.7": lambda b: (smooth(b, 0.7), 0.0),
    "smooth_0.5": lambda b: (smooth(b, 0.5), 0.0),
    "smooth_0.3": lambda b: (smooth(b, 0.3), 0.0),
    "jitter_0.05": lambda b: (jitter(b, 0.05), 0.0),
    "jitter_0.10": lambda b: (jitter(b, 0.10), 0.0),
    "jitter_0.20": lambda b: (jitter(b, 0.20), 0.0),
    "delay_1": lambda b: fire_delay(b, 1),
    "delay_2": lambda b: fire_delay(b, 2),
    "delay_4": lambda b: fire_delay(b, 4),
}
