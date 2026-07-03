"""Extract rotation/scale-invariant *dynamics* features for every orbit in the
continuation pool — to test whether trajectory-derived features improve
cross-family extrapolation, where raw (a,c,L) fails (Track A, Result 3).

Per orbit, all transferable across families (invariant to rotation/translation,
and mostly to scale):
  d_min, d_max, d_ratio  -- min/max pairwise distance: close-approach severity
                            (tidal instability driver) and orbit extent
  I_cv                   -- moment-of-inertia coefficient of variation (size pulsation)
  ss_zcross              -- shape-sphere equator (z=0) crossings: topological,
                            a free-group word-length proxy
  ss_{x,y,z}_std, ss_zmean, ss_pathlen -- shape-sphere path spread & length

The idea: these describe the orbit's intrinsic geometry/topology, which should
transfer across families better than the parameters that merely index it.

Reads the pool via load_families (a,c,L,T per point), integrates each orbit with
loosened tolerance (geometric features don't need 1e-12), and saves the augmented
table incrementally to datasets/pool_traj_features.csv.

Run:  python -m data.build_traj_features
"""
from __future__ import annotations

import csv
import sys

import numpy as np

import ml_config as cfg

# memguard: not in the engine's py-modules; add the repo path then install.
sys.path.insert(0, str(cfg.PHYSICS_REPO))
import memguard  # noqa: E402
memguard.install()

import three_body as tb  # noqa: E402
from data.load_families import load_families  # noqa: E402

_ROOT2, _ROOT6 = np.sqrt(2.0), np.sqrt(6.0)

TRAJ_COLS = ["d_min", "d_max", "d_ratio", "I_cv",
             "ss_zcross", "ss_x_std", "ss_y_std", "ss_z_std", "ss_zmean", "ss_pathlen"]


def traj_features(a: float, c: float, L: float, T: float) -> dict | None:
    """Integrate one orbit over its period and return invariant features."""
    try:
        s0 = tb.initial_conditions_from_params(a, c, L)
        sol = tb.integrate_orbit(s0, T, rtol=1e-9, atol=1e-11, max_step=0.02)
    except Exception:
        return None
    Y = sol.y
    r0, r1, r2 = Y[0:2], Y[2:4], Y[4:6]

    # pairwise distances over the orbit
    d = np.vstack([np.linalg.norm(r0 - r1, axis=0),
                   np.linalg.norm(r0 - r2, axis=0),
                   np.linalg.norm(r1 - r2, axis=0)])
    d_min, d_max = float(d.min()), float(d.max())

    # moment of inertia about origin (CoM ~ 0), equal masses -> size pulsation
    I = np.sum(r0 ** 2, 0) + np.sum(r1 ** 2, 0) + np.sum(r2 ** 2, 0)
    I_cv = float(I.std() / I.mean())

    # shape-sphere path (vectorised)
    rho = (r0 - r1) / _ROOT2
    lam = (r0 + r1 - 2 * r2) / _ROOT6
    R2 = np.sum(rho ** 2, 0) + np.sum(lam ** 2, 0)
    ss_x = 2 * np.sum(rho * lam, 0) / R2
    ss_y = (np.sum(lam ** 2, 0) - np.sum(rho ** 2, 0)) / R2
    ss_z = 2 * (rho[0] * lam[1] - rho[1] * lam[0]) / R2
    zcross = int(np.sum(np.abs(np.diff(np.sign(ss_z))) > 0))  # equator crossings
    path = float(np.sum(np.sqrt(np.diff(ss_x) ** 2 + np.diff(ss_y) ** 2
                                + np.diff(ss_z) ** 2)))

    return {
        "d_min": d_min, "d_max": d_max, "d_ratio": d_min / d_max, "I_cv": I_cv,
        "ss_zcross": zcross, "ss_x_std": float(ss_x.std()),
        "ss_y_std": float(ss_y.std()), "ss_z_std": float(ss_z.std()),
        "ss_zmean": float(ss_z.mean()), "ss_pathlen": path,
    }


def main() -> None:
    cfg.ensure_dirs()
    pool = load_families()
    out = cfg.DATASETS / "pool_traj_features.csv"
    keep = ["family", "a", "c", "L", "T", "y_reg", "y_cls"]

    n_ok = n_fail = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keep + TRAJ_COLS)
        for i, row in pool.iterrows():
            feat = traj_features(row["a"], row["c"], row["L"], row["T"])
            if feat is None:
                n_fail += 1
                continue
            n_ok += 1
            w.writerow([row[k] for k in keep] + [feat[k] for k in TRAJ_COLS])
            f.flush()
            if n_ok % 200 == 0:
                print(f"  {n_ok} orbits done...", flush=True)
    print(f"Wrote {n_ok} orbits ({n_fail} failed) -> {out}", flush=True)


if __name__ == "__main__":
    main()
