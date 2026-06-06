"""Week 2: build the model-ready feature matrix.

Streams both classes in chunks, extracts behavioural features per instance, and
writes a tidy table to outputs/features/. Deliberately stops BEFORE the
train/test split and SMOTE -- those are Week 3.

Run:  PYTHONPATH=src python scripts/02_build_features.py
Outputs:
    outputs/features/features.parquet     (one row per instance + 'label')
    outputs/features/feature_names.json
    outputs/features/feature_summary.csv  (per-class describe())
"""
import sys
import json
import pathlib
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cheatdetect import config as C, data_loading as D, features as F  # noqa: E402

CHUNK = 1000


def build_for_class(label):
    arr = D.load_class(label)
    rows = []
    pad = []
    for start, block in D.iter_chunks(arr, chunk=CHUNK):
        pad.append(D.padding_fraction(block))
        feats = F.extract_chunk(block)
        rows.append(pd.DataFrame(feats))
        print(f"   {C.LABELS[label]}: {start + block.shape[0]:>6d}/{arr.shape[0]}")
    df = pd.concat(rows, ignore_index=True)
    df.insert(0, "label", label)
    print(f"   {C.LABELS[label]}: padding ~{100*np.mean(pad):.3f}% of ticks (left as zero-velocity)")
    return df


def main():
    C.FEAT_DIR.mkdir(parents=True, exist_ok=True)
    print("Extracting features (this reads ~1.4 GB via mmap in chunks)...")
    df = pd.concat([build_for_class(1), build_for_class(0)], ignore_index=True)

    names = [c for c in df.columns if c != "label"]
    out_parquet = C.FEAT_DIR / "features.parquet"
    df.to_parquet(out_parquet, index=False)
    (C.FEAT_DIR / "feature_names.json").write_text(json.dumps(names, indent=2))

    summary = df.groupby("label")[names].mean().T
    summary.columns = [C.LABELS[c] for c in summary.columns]
    summary.to_csv(C.FEAT_DIR / "feature_summary.csv")

    print(f"\nFeature matrix: {df.shape[0]} instances x {len(names)} features")
    print("Class balance:")
    print(df["label"].map(C.LABELS).value_counts().to_string())
    print(f"Saved -> {out_parquet}")
    print("NOTE: no train/test split and no SMOTE here -- that is Week 3.")


if __name__ == "__main__":
    main()
