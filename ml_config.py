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
BOUNDS = {
    "a": (0.0, 0.5),
    "c": (-3.0, 0.0),
    "L": (0.0, 1.0),
}

# Stability threshold (Section 3/5): all Floquet multipliers on the unit circle.
STABLE_LAMBDA_MAX = 1.0 + 1e-3


def ensure_dirs() -> None:
    DATASETS.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
