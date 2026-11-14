"""
contributions.py
================
Per-feature contributions to each tract's watch-list probability.

The production model is a pipeline: median fill, standardize, logistic
regression. Its log-odds for one tract are

    intercept + sum_j coef_j * scaled_j

where scaled_j = (value_j - training mean_j) / training SD_j. Each term
coef_j * scaled_j is that feature's contribution: how far it moves this tract's
log-odds away from a tract with every feature at the training average. The
terms add up exactly to the model's log-odds, so the dashboard can show
"why this tract" without SHAP or a second model.

Contributions use the scaled values inside the pipeline, not raw units: a
coefficient on raw population (thousands) and one on a z-score cannot be
compared, but after scaling every feature is in SD units.

Written by score_week as ml_contributions so the API reads a table and does
not need scikit-learn.
"""

from typing import Any

import numpy as np
import pandas as pd

CONTRIBUTIONS_TABLE = "ml_contributions"


def _pipeline_parts(model: Any) -> tuple[Any, Any, Any]:
    steps = dict(model.steps)
    try:
        return (
            steps["simpleimputer"],
            steps["standardscaler"],
            steps["logisticregression"],
        )
    except KeyError as exc:
        raise ValueError(
            "Contributions need an imputer + scaler + logistic regression "
            f"pipeline; got steps {list(steps)}"
        ) from exc


def feature_contributions(
    model: Any, rows: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    """
    Long table: one row per tract and feature.

    Columns: geoid, feature, value (raw, NaN when missing), imputed,
    contribution (log-odds units), logit (the tract's total log-odds).
    """
    imputer, scaler, logreg = _pipeline_parts(model)
    raw = rows[features].to_numpy(dtype=float)
    scaled = scaler.transform(imputer.transform(rows[features]))
    contrib = scaled * logreg.coef_[0]
    logit = logreg.intercept_[0] + contrib.sum(axis=1)

    # The terms must add up to what the model itself returns
    expected = model.decision_function(rows[features])
    if not np.allclose(logit, expected, atol=1e-6):
        raise ValueError("Contributions do not add up to the model's log-odds")

    n_rows, n_feat = raw.shape
    return pd.DataFrame(
        {
            "geoid": np.repeat(rows["geoid"].to_numpy(), n_feat),
            "feature": np.tile(features, n_rows),
            "value": raw.ravel(),
            "imputed": np.isnan(raw).ravel(),
            "contribution": contrib.ravel().round(5),
            "logit": np.repeat(logit.round(5), n_feat),
        }
    )
