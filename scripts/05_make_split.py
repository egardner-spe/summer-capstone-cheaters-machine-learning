"""Week 3: build and persist the dedup-aware stratified train/test split.

Consumes the keep flags from scripts/04_data_quality.py, makes an 80/20
stratified holdout over the kept instances, assigns 5-fold stratified CV ids
inside the training set, and writes the whole assignment to disk. Week 4+
loads this file -- the split is made exactly once, here.

Run:  PYTHONPATH=src python scripts/05_make_split.py
Outputs:
    outputs/splits/splits.parquet   (row, label, keep, drop_reason, split, cv_fold)
    outputs/splits/split_summary.csv
"""
import sys
import pathlib

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from cheatdetect import config as C, splitting as S  # noqa: E402


def main():
    C.SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    table = pd.read_parquet(C.QUALITY_DIR / "instance_table.parquet")
    features = pd.read_parquet(C.FEAT_DIR / "features.parquet")
    split = S.make_split(table, features)

    # ---- summary -----------------------------------------------------------
    def balance(mask):
        v = split[mask]["label"].value_counts()
        n0, n1 = int(v.get(0, 0)), int(v.get(1, 0))
        return n0, n1, (n0 / n1 if n1 else float("nan"))

    rows = []
    for name, mask in [("train", split["split"] == "train"),
                       ("test", split["split"] == "test"),
                       ("dropped", split["split"] == "dropped")]:
        n0, n1, r = balance(mask)
        rows.append({"subset": name, "n": int(mask.sum()),
                     "legit": n0, "cheater": n1, "ratio": round(r, 3)})
        print(f"{name:>8s}: {mask.sum():>6d}  (legit {n0}, cheater {n1}, "
              f"ratio {r:.3f})")
    for fold in range(C.N_FOLDS):
        mask = split["cv_fold"] == fold
        n0, n1, r = balance(mask)
        rows.append({"subset": f"fold_{fold}", "n": int(mask.sum()),
                     "legit": n0, "cheater": n1, "ratio": round(r, 3)})
        print(f"  fold {fold}: {mask.sum():>6d}  (legit {n0}, cheater {n1}, "
              f"ratio {r:.3f})")

    # ---- persist -------------------------------------------------------------
    out = C.SPLIT_DIR / "splits.parquet"
    split.to_parquet(out, index=False)
    pd.DataFrame(rows).to_csv(C.SPLIT_DIR / "split_summary.csv", index=False)
    print(f"\nSaved -> {out}")
    print(f"seed={C.RANDOM_SEED}, test_size={C.TEST_SIZE}, folds={C.N_FOLDS}")
    print("The test set is now frozen: nothing touches it until final "
          "evaluation.")


if __name__ == "__main__":
    main()
