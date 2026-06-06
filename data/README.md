# Data

The raw arrays are **not committed** (~1.4 GB total). Place them so the code can
find them, then either keep them at the default location or set `CHEATDETECT_DATA`.

## Expected layout

```
archive/
├── cheaters/cheaters.npy   # (2000, 30, 192, 5) float32
└── legit/legit.npy         # (10000, 30, 192, 5) float32
```

## Where the code looks

`src/cheatdetect/config.py` resolves the data directory as:

1. the `CHEATDETECT_DATA` environment variable, if set; otherwise
2. `../archive` relative to the repo root (the capstone folder layout).

```bash
export CHEATDETECT_DATA="/absolute/path/to/archive"
```

## Schema

Full, verified schema and channel semantics are in
[`../reports/data_schema.md`](../reports/data_schema.md). In brief, each instance
is `30 engagements × 192 ticks × 5 channels`, channels =
`[d_yaw, d_pitch, yaw, pitch, fire]`, labels are the file of origin
(`cheaters` = 1, `legit` = 0).
