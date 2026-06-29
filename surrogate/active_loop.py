"""The active-learning loop — the actual deliverable (Section 5).

    seed with catalogue labels
    repeat under a fixed budget of N monodromy evaluations:
        fit GP on labelled set
        score a Sobol candidate pool over (a, c, L)
        pick argmax acquisition (uncertainty, or expected stability)
        label it: newton_refine_bhh -> analyse_orbit   # the one expensive call
        append to labelled set

Memory-safety (Section 9): the loop is *naturally serial* — one expensive label
at a time — and must stay that way. Only the candidate-pool scoring is
parallel-friendly, and it's cheap, so it stays in-process. We never spawn a
second heavy multiprocessing job here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import qmc

import ml_config as cfg
from data.build_labels import FEATURE_COLS, Label, add_features, label_point
from surrogate.gp import build_gp, predict

import pandas as pd


# --- acquisition functions ---------------------------------------------------
def acq_uncertainty(mean, std):
    """Pure exploration: label where the GP is least sure. Good for mapping."""
    return std


def acq_stable_lcb(mean, std, kappa: float = 1.0):
    """Lower-confidence bound on log10(lam_max): seek points likely *stable*
    (small lam_max => small y_reg) while still rewarding uncertainty. This is the
    triage-oriented acquisition (Section 1: rank candidates worth tracing)."""
    return -(mean - kappa * std)


ACQUISITIONS = {"uncertainty": acq_uncertainty, "stable_lcb": acq_stable_lcb}


@dataclass
class LoopResult:
    labelled: pd.DataFrame
    stable_found_curve: list[int] = field(default_factory=list)  # cumulative
    n_evaluations: int = 0


def _sobol_pool(n: int, seed: int) -> np.ndarray:
    """n quasi-random (a, c, L) candidates in the configured box."""
    lo = np.array([cfg.BOUNDS["a"][0], cfg.BOUNDS["c"][0], cfg.BOUNDS["L"][0]])
    hi = np.array([cfg.BOUNDS["a"][1], cfg.BOUNDS["c"][1], cfg.BOUNDS["L"][1]])
    sampler = qmc.Sobol(d=3, scramble=True, seed=seed)
    unit = sampler.random(n)
    return lo + unit * (hi - lo)


def _T_guess(a: float, c: float, L: float, labelled: pd.DataFrame) -> float:
    """Period seed for Newton. Nearest labelled neighbour's T is a cheap, decent
    guess; fall back to the mean. (Newton's basin for these orbits is forgiving
    but not infinite — a sane T_guess materially lifts the convergence rate.)"""
    if labelled.empty:
        return 5.0
    d2 = ((labelled[["a", "c", "L"]].to_numpy() - np.array([a, c, L])) ** 2).sum(1)
    return float(labelled.iloc[int(d2.argmin())]["T"])


def run_active_loop(seed_df: pd.DataFrame, budget: int,
                    acquisition: str = "uncertainty",
                    pool_size: int = 4096, seed: int = 0) -> LoopResult:
    """Run the loop for `budget` true-label evaluations. Returns the growing
    labelled set and the cumulative stable-found curve (the comparison metric)."""
    if acquisition not in ACQUISITIONS:
        raise ValueError(f"acquisition must be one of {list(ACQUISITIONS)}")
    acq = ACQUISITIONS[acquisition]

    labelled = add_features(seed_df.copy())
    base_stable = int(labelled["y_cls"].sum()) if not labelled.empty else 0
    curve: list[int] = []
    evals = 0

    while evals < budget:
        # 1. fit GP on what we know
        X = labelled[FEATURE_COLS].to_numpy()
        y = labelled["y_reg"].to_numpy()
        model = build_gp().fit(X, y)

        # 2. score a fresh candidate pool (cheap, in-process)
        pool = _sobol_pool(pool_size, seed=seed + evals)
        pool_df = add_features(pd.DataFrame(pool, columns=["a", "c", "L"]))
        mean, std = predict(model, pool_df[FEATURE_COLS].to_numpy())
        scores = acq(mean, std)

        # 3. pick the best candidate
        i = int(np.argmax(scores))
        a, c, L = pool[i]

        # 4. label it — THE expensive call (one variational integration)
        lab = label_point(a, c, L, _T_guess(a, c, L, labelled), source="active")
        evals += 1

        if lab is not None:
            row = add_features(pd.DataFrame([lab.__dict__]))
            labelled = pd.concat([labelled, row], ignore_index=True)
            tag = "STABLE" if lab.y_cls else "unstable"
            print(f"[{evals:>3}/{budget}] a={a:.4f} c={c:.4f} L={L:.4f}  "
                  f"lam_max={lab.lam_max:.4g}  {tag}")
        else:
            print(f"[{evals:>3}/{budget}] a={a:.4f} c={c:.4f} L={L:.4f}  "
                  f"did not refine (no label)")

        found = (int(labelled["y_cls"].sum()) - base_stable)
        curve.append(found)

    return LoopResult(labelled=labelled, stable_found_curve=curve, n_evaluations=evals)


if __name__ == "__main__":
    # Tiny self-contained demo: seed from a couple of catalogue orbits, run a few
    # iterations. For the real experiment use experiments/exp01_surrogate_vs_grid.
    from data.build_labels import build_seed_table
    print("Building a small seed set from the catalogue...")
    seeds = build_seed_table()
    res = run_active_loop(seeds, budget=10, acquisition="uncertainty")
    print(f"\nStable-found curve (cumulative): {res.stable_found_curve}")
