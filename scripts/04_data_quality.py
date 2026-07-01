"""Week 3: data-quality pass -- verify duplicates against the RAW arrays.

Week 2 flagged 73 exact-duplicate instances, but that check ran per-class.
This script hashes every raw (30,192,5) instance across BOTH arrays, finds
duplicate groups within and across classes, reconciles them with the
feature-space duplicate groups, and applies the agreed drop policy:

    mixed-label group -> drop all members (contradictory labels)
    same-label group  -> keep first occurrence, drop extras

Run:  PYTHONPATH=src python scripts/04_data_quality.py
Outputs:
    outputs/quality/instance_table.parquet   (one row per instance: identity,
                                              raw hash, dup groups, keep flag)
    outputs/quality/duplicate_groups.csv     (per-group audit trail)
"""
import sys
import pathlib

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cheatdetect import config as C, data_quality as Q  # noqa: E402


def main():
    C.QUALITY_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading feature matrix...")
    features = pd.read_parquet(C.FEAT_DIR / "features.parquet")

    print("Hashing raw instances (both arrays, ~1.4 GB via mmap)...")
    table = Q.build_instance_table(features)
    table = Q.apply_drop_policy(table)
    groups = Q.group_summary(table)

    # ---- report ----------------------------------------------------------
    n_raw_dup = int((table["raw_dup_group"] >= 0).sum())
    n_feat_dup = int((table["feat_dup_group"] >= 0).sum())
    mixed = groups[groups["mixed_label"]]
    same = groups[~groups["mixed_label"]]

    print(f"\ninstances                     : {len(table)}")
    print(f"rows in RAW duplicate groups  : {n_raw_dup}")
    print(f"rows in FEATURE dup groups    : {n_feat_dup}  "
          f"({len(groups)} groups; feature dups are a superset of raw dups)")
    print(f"  mixed-label groups          : {len(mixed)} "
          f"({int(mixed['size'].sum())} rows -> drop all)")
    print(f"  same-label groups           : {len(same)} "
          f"({int(same['size'].sum())} rows -> keep one each)")
    print(f"  groups byte-identical in raw: {int(groups['raw_identical'].sum())}"
          f"/{len(groups)}")

    # cross-class raw duplicates = strongest evidence of label noise
    raw_groups = table[table["raw_dup_group"] >= 0].groupby("raw_dup_group")
    cross_raw = sum(1 for _, g in raw_groups if g["label"].nunique() > 1)
    print(f"  RAW groups spanning classes : {cross_raw} "
          "(same recording present in both .npy files)")

    dropped = table[~table["keep"]]
    print(f"\ndrop summary: {len(dropped)} rows "
          f"({dict(dropped['drop_reason'].value_counts())})")
    kept = table[table["keep"]]
    print("kept class balance:",
          kept["label"].map(C.LABELS).value_counts().to_dict())

    # ---- persist -----------------------------------------------------------
    out_tab = C.QUALITY_DIR / "instance_table.parquet"
    out_grp = C.QUALITY_DIR / "duplicate_groups.csv"
    table.to_parquet(out_tab, index=False)
    groups.to_csv(out_grp, index=False)
    print(f"\nSaved -> {out_tab}\nSaved -> {out_grp}")
    print("NOTE: the split (scripts/05_make_split.py) consumes the keep flag; "
          "features.parquet itself is left untouched.")


if __name__ == "__main__":
    main()
