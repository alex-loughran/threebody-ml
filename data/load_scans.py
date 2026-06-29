"""Read the physics repo's 500x500 BHH RPF scan maps into feature/label arrays.

These are the *free* labels for Track B (discovery selection function, Section 6)
and useful priors for Track A. One scan file gives, per (a, c) cell, the
periodicity score -log10(d_min): high = near-periodic, NaN = skipped cell.

File schema (Section 4):
    keys: row_vals (=a), col_vals (=c), rpf_map (shape 500x500), completed_rows
    RPF[i, j] = -log10(d_min) at (a=row_vals[i], c=col_vals[j])
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import ml_config as cfg


def scan_path(L: float, n: int = 500) -> Path:
    """Conventional filename for a BHH scan at angular momentum L."""
    # Files are named like scan_bhh_L0.8_500x500.npz / scan_bhh_L0.65_500x500.npz.
    # Trailing zeros are stripped in practice (L0.8 not L0.80), so format loosely.
    tag = f"{L:g}"
    return cfg.MINI_RESULTS / f"scan_bhh_L{tag}_{n}x{n}.npz"


def load_scan(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (A, C, RPF) where RPF.shape == (len(A), len(C))."""
    d = np.load(path)
    return d["row_vals"], d["col_vals"], d["rpf_map"]


def scan_to_table(path: str | Path, L: float, drop_nan: bool = True) -> pd.DataFrame:
    """Flatten a scan map to a tidy (a, c, L, rpf) DataFrame.

    `rpf` is the periodicity score -log10(d_min). For a Track-B label you would
    threshold this (e.g. rpf > 4  <=>  d_min < 1e-4  <=>  near-periodic).
    """
    A, C, RPF = load_scan(path)
    AA, CC = np.meshgrid(A, C, indexing="ij")     # match RPF[i, j] layout
    df = pd.DataFrame({
        "a": AA.ravel(),
        "c": CC.ravel(),
        "L": float(L),
        "rpf": RPF.ravel(),
    })
    if drop_nan:
        df = df[np.isfinite(df["rpf"])].reset_index(drop=True)
    return df


def load_all_scans(L_values: list[float], n: int = 500) -> pd.DataFrame:
    """Concatenate every available scan among L_values; skip missing files."""
    frames = []
    for L in L_values:
        p = scan_path(L, n)
        if not p.exists():
            print(f"  [miss] {p.name} not found — skipping")
            continue
        frames.append(scan_to_table(p, L))
        print(f"  [load] {p.name}: {len(frames[-1])} finite cells")
    if not frames:
        return pd.DataFrame(columns=["a", "c", "L", "rpf"])
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    # Quick smoke test against whatever scans exist on disk.
    found = sorted(cfg.MINI_RESULTS.glob("scan_bhh_L*_*x*.npz"))
    print(f"Found {len(found)} scan file(s) under {cfg.MINI_RESULTS}:")
    for p in found:
        A, C, RPF = load_scan(p)
        finite = int(np.isfinite(RPF).sum())
        print(f"  {p.name}: {RPF.shape}, {finite} finite cells, "
              f"rpf max={np.nanmax(RPF):.2f}")
