"""Loading + light cleaning for the Counter-Strike view-angle arrays.

Each .npy is shaped (n_instances, 30, 192, 5):
    instance -> 30 engagements -> 192 ticks -> 5 channels.
Arrays are memory-mapped so we never hold the full ~1.4 GB in RAM at once.
"""
from __future__ import annotations
import numpy as np
from . import config as C


def load_array(path, mmap=True):
    return np.load(path, mmap_mode="r" if mmap else None)


def load_class(label, mmap=True):
    """label: 0=legit, 1=cheater. Returns a (memory-mapped) ndarray."""
    path = C.LEGIT_NPY if label == 0 else C.CHEATERS_NPY
    return load_array(path, mmap=mmap)


def valid_tick_mask(block):
    """True for ticks that carry data.

    A handful of trailing ticks in short engagements are zero-padded (all 5
    channels exactly 0.0). Exclude them if you want strict denominators.
    In: (..., 192, 5) -> out: (..., 192).
    """
    return ~np.all(block == 0.0, axis=-1)


def padding_fraction(block):
    """Fraction of fully zero-padded ticks in a block (sanity / cleaning metric)."""
    m = valid_tick_mask(block)
    return float(1.0 - m.mean())


def iter_chunks(arr, chunk=1000, dtype=np.float32):
    """Yield (start, block) with `chunk` instances materialised into RAM."""
    n = arr.shape[0]
    for s in range(0, n, chunk):
        yield s, np.asarray(arr[s:s + chunk], dtype=dtype)


def basic_stats(arr, sample=2000):
    """Per-channel summary on an evenly spaced sample (for the reality check)."""
    n = arr.shape[0]
    idx = np.linspace(0, n - 1, min(sample, n)).astype(int)
    s = np.asarray(arr[idx], dtype=np.float32)
    out = {}
    for ch in range(arr.shape[-1]):
        v = s[..., ch].ravel()
        out[C.CHANNEL_NAMES[ch]] = dict(
            min=float(v.min()), max=float(v.max()),
            mean=float(v.mean()), std=float(v.std()),
            p1=float(np.percentile(v, 1)), p50=float(np.percentile(v, 50)),
            p99=float(np.percentile(v, 99)), frac_zero=float(np.mean(v == 0)),
        )
    return out
