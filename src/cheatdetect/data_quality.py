"""Data-quality pass (Week 3): exact-duplicate detection and the drop policy.

Two levels of duplicate matter here, and they are NOT the same set:

  * RAW duplicates      -- byte-identical (30,192,5) instances. Found by hashing
                           each instance's raw float32 bytes across BOTH arrays.
                           These are the unambiguous "same recording" cases.
  * FEATURE duplicates  -- identical 39-float feature vectors. A superset of raw
                           duplicates (raw-identical => feature-identical). Extra
                           members are distinct recordings that collapse to the
                           same point in feature space.

For split integrity the FEATURE level is what counts: two identical vectors on
opposite sides of the split let the model "generalise" by memorisation, whatever
the raw bytes say. So duplicate *groups* are defined in feature space, and the
raw hash is recorded per group as evidence of how each group arose.

Drop policy (agreed 2026-07-01):
  * mixed-label group (contains both classes)  -> drop ALL members. Identical
    inputs with contradictory labels are unlearnable, and keeping either copy
    would poison whichever side of the split it landed on.
  * same-label group                           -> keep the first occurrence,
    drop the extra copies.

Everything here is deterministic: groups are ordered by first row index, and
"first occurrence" means lowest row index in the canonical feature-matrix order
(cheaters block first, then legit -- see scripts/02_build_features.py).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from . import config as C
from . import data_loading as D

# feature-matrix row order (must match scripts/02_build_features.py: label 1
# block first, then label 0). Verified by an assert in build_instance_table().
_BUILD_ORDER = (1, 0)


def hash_instances(arr, chunk: int = 1000) -> list[str]:
    """blake2b digest of each instance's raw bytes. arr: (n, 30, 192, 5)."""
    out: list[str] = []
    for _, block in D.iter_chunks(arr, chunk=chunk, dtype=np.float32):
        for inst in block:
            out.append(hashlib.blake2b(np.ascontiguousarray(inst).tobytes(),
                                       digest_size=16).hexdigest())
    return out


def build_instance_table(features: pd.DataFrame) -> pd.DataFrame:
    """One row per instance: identity, label, raw hash, duplicate-group ids.

    Columns:
        row          -- position in features.parquet (canonical id everywhere)
        label        -- 0 legit / 1 cheater
        source       -- which .npy the instance came from
        source_index -- index inside that .npy
        raw_hash     -- blake2b-128 of the raw (30,192,5) float32 bytes
        raw_dup_group / feat_dup_group -- group id, -1 if not duplicated
    """
    # identity: reproduce the exact row order 02_build_features.py wrote
    parts = []
    for label in _BUILD_ORDER:
        arr = D.load_class(label)
        name = "cheaters" if label == 1 else "legit"
        parts.append(pd.DataFrame({
            "label": label,
            "source": name,
            "source_index": np.arange(arr.shape[0]),
            "raw_hash": hash_instances(arr),
        }))
    table = pd.concat(parts, ignore_index=True)
    table.insert(0, "row", np.arange(len(table)))

    if len(table) != len(features):
        raise ValueError(f"instance table ({len(table)}) != feature matrix "
                         f"({len(features)}) -- rebuild features first")
    if not (table["label"].to_numpy() == features["label"].to_numpy()).all():
        raise AssertionError("label order mismatch: features.parquet was not "
                             "built cheaters-first -- row mapping is invalid")

    table["raw_dup_group"] = _group_ids(table["raw_hash"])
    feat_cols = [c for c in features.columns if c != "label"]
    feat_key = pd.util.hash_pandas_object(features[feat_cols], index=False)
    table["feat_dup_group"] = _group_ids(feat_key)
    return table


def _group_ids(keys: pd.Series) -> np.ndarray:
    """Group id per row for keys occurring >1 time; -1 for unique rows.

    Ids are assigned in order of each group's first appearance (deterministic).
    """
    counts = keys.value_counts()
    dup_keys = counts[counts > 1].index
    ids = np.full(len(keys), -1, dtype=np.int64)
    first_pos = {}
    next_id = 0
    dup_set = set(dup_keys)
    for pos, k in enumerate(keys):
        if k in dup_set:
            if k not in first_pos:
                first_pos[k] = next_id
                next_id += 1
            ids[pos] = first_pos[k]
    return ids


def apply_drop_policy(table: pd.DataFrame) -> pd.DataFrame:
    """Add keep / drop_reason columns (policy documented in the module header)."""
    table = table.copy()
    table["keep"] = True
    table["drop_reason"] = ""

    grouped = table[table["feat_dup_group"] >= 0].groupby("feat_dup_group")
    for _, g in grouped:
        idx = g.index.to_numpy()
        if g["label"].nunique() > 1:                       # mixed-label group
            table.loc[idx, "keep"] = False
            table.loc[idx, "drop_reason"] = "mixed_label_duplicate"
        else:                                              # same-label group
            table.loc[idx[1:], "keep"] = False             # keep first row
            table.loc[idx[1:], "drop_reason"] = "duplicate_extra_copy"
    return table


def group_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per-duplicate-group audit table (for outputs/quality/duplicate_groups.csv)."""
    rows = []
    for gid, g in table[table["feat_dup_group"] >= 0].groupby("feat_dup_group"):
        raw_identical = g["raw_hash"].nunique() == 1
        mixed = g["label"].nunique() > 1
        rows.append({
            "feat_dup_group": int(gid),
            "size": len(g),
            "n_cheater": int((g["label"] == 1).sum()),
            "n_legit": int((g["label"] == 0).sum()),
            "raw_identical": raw_identical,
            "mixed_label": mixed,
            "action": "drop_all" if mixed else "keep_first",
            "rows": ";".join(map(str, g["row"].tolist())),
        })
    return pd.DataFrame(rows)
