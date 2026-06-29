"""The comparison arm: spend the same N evaluations on a non-adaptive sampler.

This is the honest baseline the whole project is measured against (Section 11
success metric: stable-orbits-found per evaluation, ML-guided vs this). Two
flavours of "uniform": a regular grid, and a scrambled Sobol sequence (usually
the fairer non-adaptive competitor — better space-filling than a coarse grid).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import qmc

import ml_config as cfg
from data.build_labels import add_features, label_point


def _grid_points(n: int) -> np.ndarray:
    """~n points on a regular 3D lattice over the configured box."""
    per_axis = max(1, round(n ** (1 / 3)))
    axes = [np.linspace(*cfg.BOUNDS[k], per_axis) for k in ("a", "c", "L")]
    mesh = np.meshgrid(*axes, indexing="ij")
    pts = np.column_stack([m.ravel() for m in mesh])
    return pts[:n]


def _sobol_points(n: int, seed: int) -> np.ndarray:
    lo = np.array([cfg.BOUNDS[k][0] for k in ("a", "c", "L")])
    hi = np.array([cfg.BOUNDS[k][1] for k in ("a", "c", "L")])
    return lo + qmc.Sobol(d=3, scramble=True, seed=seed).random(n) * (hi - lo)


def run_uniform(budget: int, mode: str = "sobol", seed: int = 0) -> pd.DataFrame:
    """Label `budget` non-adaptively chosen points. Returns the labelled table
    plus a cumulative `stable_found` column for direct comparison with the loop."""
    pts = _sobol_points(budget, seed) if mode == "sobol" else _grid_points(budget)

    rows, found = [], 0
    found_curve = []
    for j, (a, c, L) in enumerate(pts, start=1):
        lab = label_point(a, c, L, T_guess=5.0, source=f"grid_{mode}")
        if lab is not None:
            rows.append(lab.__dict__)
            found += int(lab.y_cls)
            tag = "STABLE" if lab.y_cls else "unstable"
            print(f"[{j:>3}/{budget}] a={a:.4f} c={c:.4f} L={L:.4f}  {tag}")
        else:
            print(f"[{j:>3}/{budget}] a={a:.4f} c={c:.4f} L={L:.4f}  no label")
        found_curve.append(found)

    df = add_features(pd.DataFrame(rows)) if rows else pd.DataFrame()
    if not df.empty:
        df.attrs["stable_found_curve"] = found_curve
    return df


if __name__ == "__main__":
    df = run_uniform(budget=10, mode="sobol")
    print(f"\nStable found: {int(df['y_cls'].sum()) if not df.empty else 0}/"
          f"{len(df)} labelled")
