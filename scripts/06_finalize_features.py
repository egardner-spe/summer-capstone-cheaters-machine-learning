"""Week 3: finalize the feature set (train-only conservative prune).

Applies the two unsupervised rules in src/cheatdetect/feature_selection.py
(near-zero variance, |r| > 0.95 redundancy) using TRAINING rows only, and
persists the final feature list Week 4 will train on. The full 39-feature
matrix is untouched -- this only writes the list + an audit log, so any
decision here is reversible.

Run:  PYTHONPATH=src python scripts/06_finalize_features.py
Outputs:
    outputs/features/final_features.json   (ordered list Week 4 consumes)
    outputs/features/prune_log.csv         (what was dropped, why, with |r|)
"""
import sys
import json
import pathlib

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cheatdetect import config as C, feature_selection as FS, splitting as S  # noqa: E402


def main():
    features = pd.read_parquet(C.FEAT_DIR / "features.parquet")
    split = S.load_split()

    final, log = FS.finalize_features(features, split)
    n_all = features.shape[1] - 1

    print(f"features in : {n_all}")
    print(f"features out: {len(final)}  (dropped {n_all - len(final)})")
    if len(log):
        print("\nprune log (train-only statistics):")
        for _, r in log.iterrows():
            tail = f" [|r|={r['abs_r']}]" if pd.notna(r["abs_r"]) else ""
            print(f"  - {r['dropped']:<22s} {r['reason']}{tail}")
    else:
        print("nothing pruned")

    out_json = C.FEAT_DIR / "final_features.json"
    out_log = C.FEAT_DIR / "prune_log.csv"
    out_json.write_text(json.dumps(final, indent=2))
    log.to_csv(out_log, index=False)
    print(f"\nSaved -> {out_json}\nSaved -> {out_log}")


if __name__ == "__main__":
    main()
