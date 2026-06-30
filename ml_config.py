"""Shared paths and parameter-space bounds for the ML repo.

Kept deliberately small. Path resolution prefers an env var so this repo never
hard-codes a machine layout; defaults point at the sibling physics repo.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- where the physics engine and its on-disk data live ----------------------
# Override with THREEBODY_PHYSICS_REPO if the repo moves.
PHYSICS_REPO = Path(
    os.environ.get("THREEBODY_PHYSICS_REPO", Path.home() / "PycharmProjects" / "PythonProject1")
).expanduser()

# Scan maps, continuation JSONs, ll_orbits.json, etc. (Section 4 inventory).
MINI_RESULTS = PHYSICS_REPO / "mini_results"

# --- where this repo writes its own outputs ----------------------------------
REPO_ROOT = Path(__file__).resolve().parent
DATASETS = REPO_ROOT / "datasets"   # gitignored: built label tables
RESULTS = REPO_ROOT / "results"     # gitignored data + tracked reports

# --- parameter-space bounds for candidate pools ------------------------------
# Starter box for (a, c, L). These are NOT physically authoritative — they are a
# pragmatic envelope around the seed catalogue. Tighten them against the real
# Jankovic/BHH ranges (build_labels can print the seed extent) before trusting a
# discovery run. See decision log, Section 11.
# Measured from the 75 Jankovic seeds (build_labels --seeds, 2026-06-29),
# padded slightly. NOTE: c is roughly symmetric (seeds span -3.2..+3.9), not
# negative-only as first guessed; L is a narrow band, not (0,1).
BOUNDS = {
    "a": (0.05, 0.55),
    "c": (-3.3, 4.0),
    "L": (0.65, 1.07),
}

# Stability threshold (Section 3/5): all Floquet multipliers on the unit circle.
STABLE_LAMBDA_MAX = 1.0 + 1e-3


def ensure_dirs() -> None:
    DATASETS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)


# --- feature engineering (physics-free; shared by labeller and family loader) -
# Section 5 schema. Kept here so any module can build features without importing
# the physics engine.
FEATURE_COLS = ["a", "c", "L", "ac", "L_minus_ac", "sign_c"]


def add_features(df):
    """Add the Section 5 derived features to a DataFrame with a, c, L columns.
    Cheap, deterministic, no leakage. Returns df unchanged if empty."""
    import numpy as np
    if df.empty:
        return df
    df = df.copy()
    df["ac"] = df["a"] * df["c"]
    df["L_minus_ac"] = df["L"] - df["ac"]
    df["sign_c"] = np.sign(df["c"])
    return df
