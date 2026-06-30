"""Gaussian-Process stability surrogate (the load-bearing model).

Predicts y_reg = log10(lam_max) with *calibrated uncertainty* — the std is what
the active-learning acquisition step consumes (Section 5). Inputs are 2-3D and
labels are scarce/expensive: exactly the regime where a GP beats a neural net.
"""
from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def build_gp(length_scale=1.0, nu: float = 2.5,
             n_restarts: int = 4) -> Pipeline:
    """A scaled-input Matern GP regressor.

    - Matern(nu=2.5): twice-differentiable, a sane default for smooth-ish but not
      analytic response surfaces (lam_max is bumpy — see the close-approach
      scatter note in Section 5; do not assume an RBF/infinitely-smooth prior).
    - WhiteKernel: absorbs label noise / jitter so the GP doesn't interpolate
      every point and collapse its uncertainty to zero.
    - StandardScaler on inputs so a single length_scale is comparable across the
      a, c, L axes (their raw ranges differ).
    """
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=length_scale, nu=nu,
                 length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e1))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,              # center/scale the target
        n_restarts_optimizer=n_restarts,  # robuster fit; lower for inner AL loops
        random_state=0,
    )
    # Scale inputs, then GP. predict(..., return_std=True) flows through.
    return Pipeline([("scale", StandardScaler()), ("gp", gp)])


def fit(model: Pipeline, X: np.ndarray, y: np.ndarray) -> Pipeline:
    model.fit(X, y)
    return model


def predict(model: Pipeline, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std). std is the acquisition signal."""
    # The final estimator supports return_std; Pipeline forwards **kwargs.
    mean, std = model.predict(X, return_std=True)
    return mean, std
