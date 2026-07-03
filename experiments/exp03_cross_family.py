"""Experiment 03: does the surrogate EXTRAPOLATE to unseen families?

Interpolation within known families is easy (exp02 diagnostic: R^2~0.996). The
harder, more useful question for a triage/discovery tool: train on some families,
predict an entirely held-out family. Leave-one-family-out over the pool.

A big gap between interpolation (~0.10) and extrapolation RMSE tells us the
surrogate memorizes family curves rather than learning a transferable
(a,c,L)->stability law — which bounds how it can be used.

Run:  python -m experiments.exp03_cross_family
"""
from __future__ import annotations

import warnings

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import numpy as np
from sklearn.ensemble import RandomForestRegressor

import ml_config as cfg
from data.load_families import load_families
from surrogate.gp import build_gp, predict


def _rmse(model, X, y, is_gp):
    pred = predict(model, X)[0] if is_gp else model.predict(X)
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def main() -> None:
    pool = load_families()
    fams = sorted(pool["family"].unique(), key=lambda f: -len(pool[pool.family == f]))
    print(f"Pool {len(pool)} pts, {len(fams)} families. Leave-one-family-out:\n")
    print(f"{'held-out family':22s} {'n':>5}  {'GP':>6}  {'RandomForest':>12}")

    gp_errs, rf_errs = [], []
    for held in fams:
        tr = pool[pool.family != held]
        te = pool[pool.family == held]
        Xtr, ytr = tr[cfg.FEATURE_COLS].to_numpy(), tr["y_reg"].to_numpy()
        Xte, yte = te[cfg.FEATURE_COLS].to_numpy(), te["y_reg"].to_numpy()

        gp = build_gp(1.0, n_restarts=1).fit(Xtr, ytr)
        rf = RandomForestRegressor(n_estimators=300, random_state=0,
                                   n_jobs=-1).fit(Xtr, ytr)
        eg, er = _rmse(gp, Xte, yte, True), _rmse(rf, Xte, yte, False)
        gp_errs.append(eg); rf_errs.append(er)
        print(f"{held:22s} {len(te):>5}  {eg:>6.3f}  {er:>12.3f}")

    print(f"\n{'MEAN extrapolation RMSE':22s} {'':>5}  {np.mean(gp_errs):>6.3f}  "
          f"{np.mean(rf_errs):>12.3f}")
    print("(compare interpolation ceiling ~0.10; target std ~1.63)")


if __name__ == "__main__":
    main()
