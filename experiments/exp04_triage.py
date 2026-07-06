"""Experiment 04: the surrogate as a triage tool.

Reframe from "predict log10(lambda_max) accurately" to the actually-useful task:
*rank* orbits by stability so expensive exact (variational-monodromy) analysis is
spent on the promising ones. "Interesting" = least unstable (bottom decile of
lambda_max — closest to the rare stable orbits).

Two questions, each in interpolation (5-fold CV) and extrapolation
(leave-one-family-out) regimes, comparing cheap (a,c,L) vs +trajectory features:
  1. Ranking quality      -> Spearman correlation between predicted & true rank.
  2. Triage efficiency    -> if you exact-analyse only the K% the surrogate ranks
                             most promising, what fraction of truly-interesting
                             orbits do you catch (recall)? Plotted vs K.

Run:  python -m experiments.exp04_triage
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold

import ml_config as cfg
from data.build_traj_features import TRAJ_COLS

INTERESTING_Q = 0.10   # bottom decile of lambda_max = "worth exact analysis"


def _rf():
    return RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1)


def lofo_predict(df, cols):
    """Out-of-family predictions: each family predicted by a model trained on the
    others (the extrapolation regime)."""
    pred = np.full(len(df), np.nan)
    for h in df.family.unique():
        tr, te = df[df.family != h], df[df.family == h]
        pred[df.family.values == h] = _rf().fit(tr[cols], tr.y_reg).predict(te[cols])
    return pred


def triage_curve(y_true, y_pred, q=INTERESTING_Q):
    """Recall of the truly-interesting (bottom-q lambda_max) orbits as a function
    of the fraction of orbits exact-analysed, when analysing in surrogate-ranked
    order (most-promising = lowest predicted lambda_max first)."""
    n = len(y_true)
    interesting = y_true <= np.quantile(y_true, q)
    total = interesting.sum()
    order = np.argsort(y_pred)                       # least-unstable predicted first
    caught = np.cumsum(interesting[order])
    frac_analysed = np.arange(1, n + 1) / n
    recall = caught / total
    return frac_analysed, recall


def main() -> None:
    cfg.ensure_dirs()
    df = cfg.add_features(pd.read_csv(cfg.DATASETS / "pool_traj_features.csv"))
    y = df.y_reg.to_numpy()
    cv = KFold(5, shuffle=True, random_state=0)
    sets = {"base (a,c,L)": cfg.FEATURE_COLS,
            "base + traj": cfg.FEATURE_COLS + TRAJ_COLS}

    print(f"Pool {len(df)} orbits. 'Interesting' = bottom {int(INTERESTING_Q*100)}% "
          f"of lambda_max (n={int((y<=np.quantile(y,INTERESTING_Q)).sum())}).\n")
    print(f"{'features':16s} {'regime':14s} {'Spearman':>9s}")

    curves = {}
    for name, cols in sets.items():
        for regime, pred in [("interpolation", cross_val_predict(_rf(), df[cols], y, cv=cv)),
                             ("extrapolation", lofo_predict(df, cols))]:
            rho = spearmanr(y, pred).correlation
            print(f"{name:16s} {regime:14s} {rho:>9.3f}")
            curves[(name, regime)] = triage_curve(y, pred)

    # plot the interpolation-regime triage curves (the deployable case)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name in sets:
        fa, rec = curves[(name, "interpolation")]
        ax.plot(fa, rec, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="random (no triage)")
    ax.axhline(1.0, color="gray", lw=0.5)
    ax.set_xlabel("fraction of orbits exact-analysed (surrogate-ranked order)")
    ax.set_ylabel(f"recall of interesting (bottom-{int(INTERESTING_Q*100)}%) orbits")
    ax.set_title("Exp04: triage efficiency (interpolation regime)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = cfg.RESULTS / "exp04_triage.png"
    fig.savefig(out, dpi=130)

    # headline: fraction analysed to reach 90% recall (interpolation)
    print("\nCompute to catch 90% of interesting orbits (interpolation):")
    for name in sets:
        fa, rec = curves[(name, "interpolation")]
        hit = np.where(rec >= 0.90)[0]
        frac = fa[hit[0]] if len(hit) else 1.0
        print(f"  {name:16s} analyse {frac*100:4.1f}%  (random baseline: 90%)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
