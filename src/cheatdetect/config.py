"""Central configuration: paths, dataset constants, channel semantics, thresholds.

The channel semantics below were verified empirically (not assumed) during the
Week 1 data reality check -- see reports/data_schema.md.
"""
from __future__ import annotations
from pathlib import Path
import os

# --- paths -------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
# Raw arrays live OUTSIDE the repo (1.4 GB, not committed). Default: ../archive
DATA_DIR = Path(os.environ.get("CHEATDETECT_DATA", REPO_ROOT.parent / "archive"))
CHEATERS_NPY = DATA_DIR / "cheaters" / "cheaters.npy"
LEGIT_NPY = DATA_DIR / "legit" / "legit.npy"

OUTPUT_DIR = REPO_ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"
FEAT_DIR = OUTPUT_DIR / "features"

# --- dataset shape -----------------------------------------------------------
N_ENGAGEMENTS = 30          # sequences per labelled instance
WINDOW_TICKS = 192          # ticks per engagement (~3.0 s @ 64 tick)
N_CHANNELS = 5
TICKRATE = 64               # CS:GO standard; 192 ticks ~ 3 s
SECONDS_PER_TICK = 1.0 / TICKRATE

# --- channel semantics (last axis) -------------------------------------------
#   ch0/ch1: per-tick angular velocity (~ -d(yaw)/-d(pitch); |corr| ~ 0.93 with
#            the differenced absolute angles -- sign flipped by convention).
#   ch2/ch3: absolute view angles (yaw spans +/-180; pitch narrow, ~horizon).
#   ch4    : binary fire/shot flag (~5% of ticks).
CH_DYAW = 0     # horizontal angular velocity (deg/tick)
CH_DPITCH = 1   # vertical angular velocity (deg/tick)
CH_YAW = 2      # absolute yaw (deg, wraps +/-180)
CH_PITCH = 3    # absolute pitch (deg)
CH_FIRE = 4     # 1.0 on ticks a shot is fired
CHANNEL_NAMES = {0: "d_yaw", 1: "d_pitch", 2: "yaw", 3: "pitch", 4: "fire"}

# --- feature-engineering thresholds (deg/tick unless noted) ------------------
STILL_SPEED = 0.05      # aim "still" / micro-hold below this
SLOW_SPEED = 0.5
MED_SPEED = 3.0         # boundary for "medium-velocity" (lit: cheaters under-represented)
LOWSHOT_SPEED = 0.20    # shot counted as "locked/snapped" if aim speed below this
PRE_SHOT_TICKS = 6      # ~94 ms window before a shot
POST_SHOT_TICKS = 3     # ~47 ms window after a shot

# --- labels ------------------------------------------------------------------
LABELS = {0: "legit", 1: "cheater"}
