"""Experiment 02: the real Track A deliverable.

Pool-based active learning vs random sampling, predicting log10(lambda_max) over
the continuation-family pool. The headline result: *active learning reaches a
given prediction accuracy with fewer expensive labels than random* — the
sample-efficiency-over-an-expensive-simulator story (the ML-for-silicon pattern).

This runs entirely in-memory: every label already exists in the pool (traced by
continuation), so there are NO physics integrations here. Fast to iterate.

Protocol (standard pool-based AL):
    fix a held-out test set (for honest RMSE)
    seed with k0 random labels from the candidate pool
    repeat to a label budget:
        fit GP on the labelled set
        ACTIVE: reveal the B unlabelled candidates of highest GP std
        RANDOM: reveal B random unlabelled candidates  (the baseline arm)
        record test RMSE vs #labels
    average over repeats (AL curves are noisy); report labels-to-accuracy ratio.

Run:
    python -m experiments.exp02_active_vs_random --budget 160 --repeats 6
"""
from __future__ import annotations

import argparse
import json
import warnings

from sklearn.exceptions import ConvergenceWarning
# The along-family signal is nearly noiseless, so the GP's WhiteKernel noise
# term pins to its lower bound on every refit — a benign, expected warning here.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ml_config as cfg
from data.load_families import load_families
from surrogate.gp import build_gp, predict


def _rmse(model, X, y):
    mean, _ = predict(model, X)
    return float(np.sqrt(np.mean((mean - y) ** 2)))


def _run_one(X, y, budget, k0, batch, strategy, test_frac, rng):
    """One AL trajectory. Returns (n_labels[], rmse[])."""
    n = len(y)
    perm = rng.permutation(n)
    n_test = int(round(test_frac * n))
    test_idx, candidate_idx = perm[:n_test], perm[n_test:]
    Xte, yte = X[test_idx], y[test_idx]

    labelled = list(rng.choice(candidate_idx, size=k0, replace=False))
    pool = [i for i in candidate_idx if i not in set(labelled)]

    ns, rmses = [], []
    while True:
        model = build_gp(n_restarts=1).fit(X[labelled], y[labelled])
        ns.append(len(labelled))
        rmses.append(_rmse(model, Xte, yte))
        if len(labelled) >= budget or not pool:
            break

        b = min(batch, len(pool), budget - len(labelled))
        if strategy == "active":
            _, std = predict(model, X[pool])
            take = np.argsort(std)[-b:]            # highest predictive std
        else:  # random baseline
            take = rng.choice(len(pool), size=b, replace=False)
        chosen = [pool[i] for i in take]
        labelled += chosen
        pool = [i for i in pool if i not in set(chosen)]
    return np.array(ns), np.array(rmses)


def _avg_curve(X, y, budget, k0, batch, strategy, test_frac, repeats, seed0):
    """Average RMSE-vs-#labels over repeats on a common label grid."""
    curves = []
    grid = None
    for r in range(repeats):
        rng = np.random.default_rng(seed0 + r)
        ns, rmses = _run_one(X, y, budget, k0, batch, strategy, test_frac, rng)
        if grid is None:
            grid = ns
        curves.append(np.interp(grid, ns, rmses))
    return grid, np.mean(curves, axis=0), np.std(curves, axis=0)


def _labels_to_reach(grid, curve, target):
    """Fewest labels at which the averaged curve reaches `target` RMSE."""
    hit = np.where(curve <= target)[0]
    return int(grid[hit[0]]) if len(hit) else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=160)
    ap.add_argument("--k0", type=int, default=8, help="initial random labels")
    ap.add_argument("--batch", type=int, default=4, help="labels revealed per round")
    ap.add_argument("--repeats", type=int, default=6)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg.ensure_dirs()

    pool = load_families()
    X = pool[cfg.FEATURE_COLS].to_numpy()
    y = pool["y_reg"].to_numpy()
    print(f"Pool: {len(y)} points, log10(lam_max) in [{y.min():.2f}, {y.max():.2f}]")

    common = dict(X=X, y=y, budget=args.budget, k0=args.k0, batch=args.batch,
                  test_frac=args.test_frac, repeats=args.repeats, seed0=args.seed)
    print("Running active arm..."); g, a_mean, a_std = _avg_curve(strategy="active", **common)
    print("Running random arm..."); _, r_mean, r_std = _avg_curve(strategy="random", **common)

    # headline: labels to reach a target RMSE (use the random arm's mid-accuracy)
    target = float(np.interp(args.budget // 2, g, r_mean))
    n_active = _labels_to_reach(g, a_mean, target)
    n_random = _labels_to_reach(g, r_mean, target)
    speedup = (n_random / n_active) if (n_active and n_random) else None

    summary = {
        "pool_size": len(y), "budget": args.budget, "repeats": args.repeats,
        "target_rmse": round(target, 4),
        "labels_active": n_active, "labels_random": n_random,
        "label_efficiency_x": round(speedup, 2) if speedup else None,
        "final_rmse_active": round(float(a_mean[-1]), 4),
        "final_rmse_random": round(float(r_mean[-1]), 4),
    }
    out_json = cfg.RESULTS / "exp02_active_vs_random.json"
    out_json.write_text(json.dumps(summary, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(g, a_mean, "-o", ms=3, label="active (GP uncertainty)")
    ax.fill_between(g, a_mean - a_std, a_mean + a_std, alpha=0.15)
    ax.plot(g, r_mean, "-s", ms=3, label="random (baseline)")
    ax.fill_between(g, r_mean - r_std, r_mean + r_std, alpha=0.15)
    if target:
        ax.axhline(target, ls="--", lw=0.8, color="gray",
                   label=f"target RMSE={target:.2f}")
    ax.set_xlabel("# expensive labels")
    ax.set_ylabel("test RMSE on log10(lambda_max)")
    ax.set_title("Exp02: active learning vs random over continuation pool")
    ax.legend()
    fig.tight_layout()
    out_png = cfg.RESULTS / "exp02_active_vs_random.png"
    fig.savefig(out_png, dpi=130)

    print("\n=== Result ===")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out_json}\nwrote {out_png}")


if __name__ == "__main__":
    main()
