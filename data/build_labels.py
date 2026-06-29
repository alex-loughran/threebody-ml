"""Turn (a, c, L) points into stability labels via the physics engine.

This module owns *the one expensive call* the surrogate exists to avoid:

    newton_refine_bhh(a, c, L, T_guess)   # Newton to ~1e-12
    analyse_orbit(state0, T_r)            # one 156-component variational integration

`label_point` is the single label factory used by both the seed-table builder
here and the active-learning loop in surrogate/active_loop.py.

Run:
    python -m data.build_labels --seeds          # label the 75 Jankovic seeds
    python -m data.build_labels --seeds --csv    # force CSV instead of parquet
"""
from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

import ml_config as cfg

# --- memguard: required, but NOT in the physics package's py-modules allowlist.
# An editable install therefore does NOT expose it; it's only importable from the
# physics repo dir. So fall back to the repo path ml_config already resolves, and
# fail LOUD if even that misses — a silent miss means no OOM watchdog and a
# swapped box on the first sweep (Section 9 — this killed the machine twice).
import sys as _sys

try:
    import memguard
except ModuleNotFoundError:
    _sys.path.insert(0, str(cfg.PHYSICS_REPO))
    try:
        import memguard
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise RuntimeError(
            f"`memguard` not importable from {cfg.PHYSICS_REPO}. Install the "
            "engine editable (`pip install -e <physics-repo>`) and/or set "
            "THREEBODY_PHYSICS_REPO. Do NOT run label sweeps without it (Section 9)."
        ) from exc
memguard.install()

# --- physics API (Section 3, verified against the package) --------------------
from floquet import analyse_orbit, newton_refine_bhh
from three_body import ALL_ORBITS, initial_conditions_from_params


@dataclass
class Label:
    """One labelled, refined orbit. Mirrors the Section 5 dataset schema."""
    a: float
    c: float
    L: float
    lam_max: float          # max |Floquet multiplier|
    y_reg: float            # log10(lam_max)  — regression target
    y_cls: bool             # lam_max < threshold — stable?
    T: float                # refined period
    converged: bool
    source: str             # provenance: 'seed', 'active', 'grid', ...


def label_point(a: float, c: float, L: float, T_guess: float,
                source: str = "active") -> Label | None:
    """Refine (a,c,L) to a true orbit and classify its stability.

    Returns None if Newton refinement fails to converge — a non-orbit point
    carries no usable stability label. Callers should treat None as "spent one
    evaluation, got no label" (it still costs budget).
    """
    a_r, c_r, T_r, ok, _info = newton_refine_bhh(a, c, L, T_guess)
    if not ok:
        return None

    state0 = initial_conditions_from_params(a_r, c_r, L)
    res = analyse_orbit(state0, T_r, verbose=False)
    if not res.get("valid", True):
        return None

    lam_max = max(abs(m) for m in res["multipliers"])
    # log10 of |λ|: clamp away from 0 so a perfectly stable orbit (λ≈1) maps to
    # ~0 rather than blowing up; lam_max is always >= ~1 for a valid monodromy.
    y_reg = math.log10(max(lam_max, 1e-12))
    return Label(
        a=a_r, c=c_r, L=L,
        lam_max=lam_max, y_reg=y_reg,
        y_cls=bool(lam_max < cfg.STABLE_LAMBDA_MAX),
        T=T_r, converged=True, source=source,
    )


def build_seed_table() -> pd.DataFrame:
    """Label the 75 Jankovic BHH catalogue orbits (ALL_ORBITS) as the seed set.

    Tuple layout per Section 3: (n, L, a, c, T, k) — k = free-group word length.
    """
    rows: list[dict] = []
    for n, L, a, c, T, k in ALL_ORBITS:
        lab = label_point(a, c, L, T, source="seed")
        if lab is None:
            print(f"  [skip] seed n={n} (a={a:.4f}, c={c:.4f}, L={L:.4f}) did not refine")
            continue
        d = asdict(lab)
        d["word_len"] = k
        rows.append(d)
        flag = "stable" if lab.y_cls else "unstable"
        print(f"  [ok]  n={n:<3} L={L:.3f}  lam_max={lab.lam_max:.4g}  {flag}")
    df = pd.DataFrame(rows)
    return add_features(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Section 5 derived features. Cheap, deterministic, no leakage."""
    if df.empty:
        return df
    df = df.copy()
    df["ac"] = df["a"] * df["c"]
    df["L_minus_ac"] = df["L"] - df["ac"]
    df["sign_c"] = np.sign(df["c"])
    return df


FEATURE_COLS = ["a", "c", "L", "ac", "L_minus_ac", "sign_c"]


def save_table(df: pd.DataFrame, stem: str, prefer_parquet: bool = True) -> str:
    cfg.ensure_dirs()
    if prefer_parquet:
        try:
            path = cfg.DATASETS / f"{stem}.parquet"
            df.to_parquet(path)
            return str(path)
        except Exception:
            pass  # pyarrow not installed — fall back to CSV
    path = cfg.DATASETS / f"{stem}.csv"
    df.to_csv(path, index=False)
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", action="store_true",
                    help="Label the 75 Jankovic catalogue seeds.")
    ap.add_argument("--csv", action="store_true",
                    help="Force CSV output instead of parquet.")
    args = ap.parse_args()

    if args.seeds:
        print("Labelling Jankovic seed catalogue (one variational integration each)...")
        df = build_seed_table()
        path = save_table(df, "labels_seed", prefer_parquet=not args.csv)
        n_stable = int(df["y_cls"].sum()) if not df.empty else 0
        print(f"\nWrote {len(df)} labelled seeds ({n_stable} stable) -> {path}")
        if not df.empty:
            print("Seed parameter extent (use to tighten ml_config.BOUNDS):")
            for col in ("a", "c", "L"):
                print(f"  {col}: [{df[col].min():.4f}, {df[col].max():.4f}]")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
