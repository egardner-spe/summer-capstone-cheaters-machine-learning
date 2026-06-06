"""Week 1 data reality check (reproducible).

Prints shapes, per-channel statistics, padding fraction, the empirical check
that channels 0/1 are angular velocities, and fire-flag statistics. This is the
script behind reports/data_schema.md.

Run:  PYTHONPATH=src python scripts/00_inspect_data.py
"""
import sys
import pathlib
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cheatdetect import config as C, data_loading as D  # noqa: E402


def wrapdiff(x):
    d = np.diff(x, axis=2)
    return (d + 180) % 360 - 180


def corr(a, b):
    a = a.ravel().astype(np.float64); b = b.ravel().astype(np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[m], b[m])[0, 1])


def main():
    for label, name in [(1, "cheaters"), (0, "legit")]:
        arr = D.load_class(label)
        print(f"\n########## {name}.npy  shape={arr.shape}  dtype={arr.dtype} ##########")
        stats = D.basic_stats(arr, sample=2000)
        for ch, s in stats.items():
            print(f"  {ch:8s} min={s['min']:8.2f} max={s['max']:8.2f} "
                  f"mean={s['mean']:8.4f} std={s['std']:7.3f} "
                  f"p1={s['p1']:7.2f} p50={s['p50']:6.2f} p99={s['p99']:7.2f} "
                  f"%zero={100*s['frac_zero']:5.1f}")
        idx = np.linspace(0, arr.shape[0] - 1, 2000).astype(int)
        s = np.asarray(arr[idx], dtype=np.float32)
        print(f"  padding (fully-zero ticks): {100*D.padding_fraction(s):.3f}%")
        dyaw = wrapdiff(s[..., C.CH_YAW]); dpit = wrapdiff(s[..., C.CH_PITCH])
        print(f"  corr(ch0, d_yaw)   = {corr(s[..., 0][:, :, 1:], dyaw):+.3f}  "
              f"corr(ch1, d_pitch) = {corr(s[..., 1][:, :, 1:], dpit):+.3f}   "
              "(=> ch0/ch1 are angular velocity)")
        fire = s[..., C.CH_FIRE]; per_eng = fire.sum(axis=2)
        print(f"  fire: tick_rate={fire.mean():.4f}  eng_with_fire="
              f"{(per_eng>0).mean():.3f}  mean_shots/eng={per_eng.mean():.2f}  "
              f"max/eng={int(per_eng.max())}")


if __name__ == "__main__":
    main()
