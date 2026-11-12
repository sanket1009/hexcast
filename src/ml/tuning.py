"""
tuning.py
=========
Nested space-time validation for choosing model settings without
touching the test windows.

For an outer split (spatial fold k, test window starting at week s):

    outer train   folds != k,                weeks <= s - embargo - 1
    inner val     fold (k + 1) % n_splits,   the last val_weeks of outer train
    inner train   folds != k and != val fold, weeks <= val_start - embargo - 1

The inner validation set holds out space and time the same way the test set
does. Without the spatial hold-out, deep trees score well on validation by
remembering tracts they trained on, and tuning would favor them.

Each candidate setting is fit on inner train with early stopping on inner
val (PR-AUC). The best setting and its tree count are refit on the full
outer train and scored on the test fold once.
"""

from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

from src.utils.logger import get_logger

logger = get_logger(__name__)

XGB_GRID = {
    "max_depth": [2, 3, 4, 6],
    "min_child_weight": [1, 20, 100],
}
XGB_FIXED = {
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "aucpr",
}
MAX_TREES = 1000
EARLY_STOPPING_ROUNDS = 50


def inner_split(
    week_idx: np.ndarray,
    fold: np.ndarray,
    outer_fold: int,
    test_start: int,
    n_splits: int,
    val_weeks: int = 8,
    embargo_weeks: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Row indices of inner train and inner validation for one outer split."""
    week_idx, fold = np.asarray(week_idx), np.asarray(fold)
    val_fold = (outer_fold + 1) % n_splits
    val_end = test_start - embargo_weeks - 1  # last outer-train week
    val_start = val_end - val_weeks + 1
    train_end = val_start - embargo_weeks - 1
    inner_val = (fold == val_fold) & (week_idx >= val_start) & (week_idx <= val_end)
    inner_train = (fold != outer_fold) & (fold != val_fold) & (week_idx <= train_end)
    info = {
        "val_fold": val_fold,
        "val_weeks": (val_start, val_end),
        "inner_train_last_week": train_end,
    }
    return np.flatnonzero(inner_train), np.flatnonzero(inner_val), info


def grid(space: dict[str, list]) -> list[dict[str, Any]]:
    keys = list(space)
    return [dict(zip(keys, values, strict=True)) for values in product(*space.values())]


def tune_xgb(
    X: pd.DataFrame,
    y: np.ndarray,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    seed: int,
    space: dict[str, list] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Best setting (with n_estimators from early stopping) and all trial scores."""
    trials = []
    for params in grid(space or XGB_GRID):
        model = xgb.XGBClassifier(
            n_estimators=MAX_TREES,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            random_state=seed,
            n_jobs=-1,
            **XGB_FIXED,
            **params,
        )
        model.fit(
            X.iloc[inner_train],
            y[inner_train],
            eval_set=[(X.iloc[inner_val], y[inner_val])],
            verbose=False,
        )
        score = average_precision_score(
            y[inner_val], model.predict_proba(X.iloc[inner_val])[:, 1]
        )
        trials.append(
            {**params, "n_estimators": model.best_iteration + 1, "val_pr_auc": score}
        )
    best = max(trials, key=lambda t: t["val_pr_auc"])
    return {k: v for k, v in best.items() if k != "val_pr_auc"}, trials


def fit_xgb(
    X: pd.DataFrame, y: np.ndarray, params: dict[str, Any], seed: int
) -> xgb.XGBClassifier:
    """Refit a chosen setting on the full outer training rows."""
    model = xgb.XGBClassifier(random_state=seed, n_jobs=-1, **XGB_FIXED, **params)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Model C: LightGBM with monotone constraints
# ---------------------------------------------------------------------------

# Features whose effect on risk has a known direction. All four index
# components are signed so that higher means more burden (raw_burden is their
# sum), so every constrained feature is +1. Deltas stay free: the data shows
# both "dropped out and comes back" and "rising into CRITICAL" entries. Static
# ACS/CDC/ER fields stay free so they can interact with burden.
MONOTONE_INCREASING = {
    "burden_percentile",
    "raw_burden",
    "pm25_z",
    "lst_z",
    "access_z",
    "vulnerability_z",
    "pct_lag_1w",
    "pct_lag_2w",
    "pct_lag_4w",
    "pct_mean_4w",
    "weeks_critical_prev4",
    "pm25_1w",
    "lst_mean_c",
    "nbr_pct_mean",
    "nbr_critical_share",
}

LGBM_GRID = {
    "num_leaves": [4, 8, 16],
    "min_child_samples": [20, 100, 400],
}
LGBM_FIXED = {
    "learning_rate": 0.05,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "monotone_constraints_method": "advanced",
    "verbose": -1,
}


def monotone_constraints(features: list[str]) -> list[int]:
    return [1 if f in MONOTONE_INCREASING else 0 for f in features]


def _lgbm(features: list[str], seed: int, **params):
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        random_state=seed,
        n_jobs=-1,
        monotone_constraints=monotone_constraints(features),
        **LGBM_FIXED,
        **params,
    )


def tune_lgbm(
    X: pd.DataFrame,
    y: np.ndarray,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    seed: int,
    space: dict[str, list] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Same contract as tune_xgb, for monotone-constrained LightGBM."""
    import lightgbm as lgb

    features = list(X.columns)
    trials = []
    for params in grid(space or LGBM_GRID):
        model = _lgbm(features, seed, n_estimators=MAX_TREES, **params)
        model.fit(
            X.iloc[inner_train],
            y[inner_train],
            eval_set=[(X.iloc[inner_val], y[inner_val])],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
        )
        score = average_precision_score(
            y[inner_val], model.predict_proba(X.iloc[inner_val])[:, 1]
        )
        trials.append(
            {
                **params,
                "n_estimators": max(int(model.best_iteration_ or 0), 1),
                "val_pr_auc": score,
            }
        )
    best = max(trials, key=lambda t: t["val_pr_auc"])
    return {k: v for k, v in best.items() if k != "val_pr_auc"}, trials


def fit_lgbm(X: pd.DataFrame, y: np.ndarray, params: dict[str, Any], seed: int):
    model = _lgbm(list(X.columns), seed, **params)
    model.fit(X, y)
    return model


# ---------------------------------------------------------------------------
# Logistic regression (champion candidate)
# ---------------------------------------------------------------------------

LOGREG_C = [0.01, 0.1, 1.0, 10.0]  # inverse L2 strength: smaller = stronger shrinkage


def make_logreg(C: float = 1.0):
    """Median fill + standardize + L2 logistic regression, all fit on the rows given."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=C, max_iter=2000),
    )


def tune_logreg(
    df: pd.DataFrame,
    y: np.ndarray,
    inner_train: np.ndarray,
    inner_val: np.ndarray,
    feature_sets: dict[str, list[str]],
    c_values: list[float] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Best (C, feature set) on inner validation PR-AUC."""
    trials = []
    for name, cols in feature_sets.items():
        for C in c_values or LOGREG_C:
            model = make_logreg(C).fit(df.iloc[inner_train][cols], y[inner_train])
            score = average_precision_score(
                y[inner_val], model.predict_proba(df.iloc[inner_val][cols])[:, 1]
            )
            trials.append({"features": name, "C": C, "val_pr_auc": score})
    best = max(trials, key=lambda t: t["val_pr_auc"])
    return {"features": best["features"], "C": best["C"]}, trials
