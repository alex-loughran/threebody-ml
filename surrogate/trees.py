"""Gradient-boosted comparators (Section 5).

Two jobs:
  1. A *comparator* for the GP on y_reg = log10(lam_max) — does a flexible tree
     model beat the GP on held-out points? (Honest-baseline discipline.)
  2. The coarse y_cls stable/unstable classifier, where calibrated uncertainty
     matters less and trees handle the bumpy response well.

XGBoost preferred, LightGBM as fallback; both are optional installs
(`pip install -e '.[trees]'`). Import is lazy so the GP path never depends on
having a booster installed.
"""
from __future__ import annotations


def _backend():
    try:
        import xgboost as xgb
        return "xgboost", xgb
    except ImportError:
        try:
            import lightgbm as lgb
            return "lightgbm", lgb
        except ImportError as exc:
            raise ImportError(
                "No gradient-boosting backend. Install one: "
                "`pip install -e '.[trees]'` (xgboost + lightgbm)."
            ) from exc


def build_regressor(**kw):
    """Regressor for y_reg. Small-data-friendly defaults; tune per dataset."""
    name, mod = _backend()
    params = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=0)
    params.update(kw)
    if name == "xgboost":
        return mod.XGBRegressor(objective="reg:squarederror", **params)
    return mod.LGBMRegressor(objective="regression", **params)


def build_classifier(**kw):
    """Classifier for y_cls (stable?). Handles class imbalance via the backend."""
    name, mod = _backend()
    params = dict(n_estimators=400, max_depth=4, learning_rate=0.05,
                  subsample=0.8, colsample_bytree=0.8, random_state=0)
    params.update(kw)
    if name == "xgboost":
        return mod.XGBClassifier(objective="binary:logistic",
                                 eval_metric="logloss", **params)
    return mod.LGBMClassifier(objective="binary", **params)
