"""Load pre-traced continuation families into a labelled pool.

Continuation (`continuation.trace_family` in the physics repo) is the cheap
label factory: it walks a periodic-orbit family from a converged seed, emitting
one fully-classified point per arclength step — each already carrying
`lambda_max` (no re-search, no off-manifold near-collision blow-ups). The traced
families are saved as JSON in the physics repo's mini_results/.

This module reads those JSONs into a tidy, physics-free DataFrame that serves as
the **ground-truth pool** for the active-learning benchmark (experiments/
exp02_active_vs_random.py). No physics engine needed to load it.

Point schema in each JSON (key 'points'):
    a, c, T, L, E, multiplier_magnitudes, lambda_max, n_unstable
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import ml_config as cfg


def family_files() -> list[Path]:
    """Every continuation-family JSON currently on disk."""
    return sorted(cfg.MINI_RESULTS.glob("continuation_family_*.json"))


def load_families(paths: list[Path] | None = None,
                  dedup_round: int | None = 6) -> pd.DataFrame:
    """Read family JSONs into one labelled DataFrame (the pool).

    Columns: family, a, c, L, T, E, lambda_max, n_unstable,
             y_reg=log10(lambda_max), y_cls=(lambda_max<threshold),
             + Section 5 derived features (ac, L_minus_ac, sign_c).

    `dedup_round`: round (a,c,L) to this many decimals and drop duplicates
    (the 'validate' families overlap). None to keep everything.
    """
    paths = paths or family_files()
    rows: list[dict] = []
    for fp in paths:
        d = json.loads(Path(fp).read_text())
        name = d.get("name", Path(fp).stem)
        for p in d["points"]:
            rows.append({
                "family": name,
                "a": p["a"], "c": p["c"], "L": p["L"], "T": p["T"],
                "E": p.get("E", np.nan),
                "lambda_max": p["lambda_max"],
                "n_unstable": p.get("n_unstable", np.nan),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if dedup_round is not None:
        key = df[["a", "c", "L"]].round(dedup_round)
        df = df.loc[~key.duplicated()].reset_index(drop=True)

    df["y_reg"] = np.log10(np.maximum(df["lambda_max"], 1e-12))
    df["y_cls"] = df["lambda_max"] < cfg.STABLE_LAMBDA_MAX
    return cfg.add_features(df)


if __name__ == "__main__":
    pool = load_families()
    print(f"Loaded {len(pool)} pooled points from {pool['family'].nunique()} "
          f"families (after dedup).")
    print(f"  log10(lambda_max): min={pool.y_reg.min():.2f} "
          f"med={pool.y_reg.median():.2f} max={pool.y_reg.max():.2f}")
    print(f"  stable points: {int(pool.y_cls.sum())} / {len(pool)}")
    print(f"  per family: {pool['family'].value_counts().to_dict()}")
