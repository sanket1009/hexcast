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
