"""
calibrate.py
============
Calibrate champion model probabilities using Platt scaling.

Gradient boosting models (XGBoost, LightGBM) are optimized for
ranking (AUC) not probability calibration. They tend to output
probabilities skewed toward 0 and 1 — when the model says 0.82,
the true probability might be only 0.71.

Platt scaling fixes this by fitting a logistic regression on the
model's raw outputs using a held-out calibration set. The logistic
regression learns the mapping from raw scores to true probabilities.

Why calibration matters for CHBIndex:
  Our intervention thresholds (0.75 = HIGH, 0.90 = CRITICAL) assume
  that probabilities mean what they say. A miscalibrated model saying
  0.90 when true probability is 0.65 generates too many false alarms.
  After calibration, 0.90 means 90% — thresholds become meaningful.

Method: Platt scaling (isotonic regression as alternative)
  Platt scaling: logistic regression on raw scores
  Isotonic regression: non-parametric monotone fit (more flexible
    but needs more data — 2,300 tracts × 52 weeks is sufficient)

Input:
    Trained champion model (from train.py)
    Out-of-fold predictions (from evaluate.py)
    True labels y

Output:
    CalibratedClassifierCV wrapper around champion model
    BigQuery: calibration_metrics (reliability diagram data)
    MLflow: calibration metrics logged

Usage:
    from src.ml.calibrate import calibrate_model
    calibrated_model = calibrate_model(config, champion_model, X, y, groups)
"""

from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss

from src.ml.spatial_cv import SpatialBlockCV
from src.utils.bigquery import write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def calibrate_model(
    config: dict[str, Any],
    champion_model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    week_str: str,
) -> Any:
    """
    Calibrate champion model probabilities using Platt scaling.

    Parameters
    ----------
    config : dict
        Merged pipeline config.
    champion_model : trained classifier
        Model B or C from train.py (whichever won evaluation).
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary target vector.
    groups : np.ndarray
        H3 spatial block labels for CV.
    week_str : str
        Current week string for BigQuery metadata.

    Returns
    -------
    CalibratedClassifierCV
        Calibrated wrapper around the champion model.
        Use this for all future predictions instead of raw model.

    Examples
    --------
    >>> calibrated = calibrate_model(config, model_b, X_b, y, groups)
    >>> probs = calibrated.predict_proba(X_new)[:, 1]
    >>> # probs are now calibrated — 0.82 means ~82% true probability
    """
    city = config["city_name"]
    full_fips = config["full_fips"]

    logger.info(f"Calibrating champion model for {city}")

    # Step 1 — Fit calibrated model using spatial CV
    # CalibratedClassifierCV wraps the model and fits a calibrator
    # cv parameter controls how calibration data is split
    # We pass our spatial CV to prevent geographic leakage here too
    cv = SpatialBlockCV(
        n_splits=config["ml"]["cv_folds"],
        random_seed=config["ml"]["random_seed"],
    )

    # method="sigmoid" = Platt scaling (logistic regression)
    # method="isotonic" = non-parametric alternative (more flexible)
    # Sigmoid is more robust with smaller datasets
    # Isotonic can overfit with fewer than ~1000 calibration points
    # With 2,300 tracts × 52 weeks we have enough for isotonic
    # but sigmoid is the safer default and standard in production
    calibrated_model = CalibratedClassifierCV(
        estimator=champion_model,
        method="sigmoid",
        cv=cv,
    )

    logger.info("Fitting Platt scaling calibration")
    calibrated_model.fit(X, y, groups=groups)

    # Step 2 — Evaluate calibration quality
    # Reliability diagram: compare predicted probabilities to actual rates
    # Perfect calibration = diagonal line (predicted 0.8 → actual 0.8)
    raw_probs = champion_model.predict_proba(X)[:, 1]
    cal_probs = calibrated_model.predict_proba(X)[:, 1]

    # Brier score before and after calibration
    brier_before = brier_score_loss(y, raw_probs)
    brier_after = brier_score_loss(y, cal_probs)
    brier_improvement = brier_before - brier_after

    logger.info(
        f"Calibration results: "
        f"Brier before={brier_before:.4f}, "
        f"Brier after={brier_after:.4f}, "
        f"improvement={brier_improvement:+.4f}"
    )

    # Step 3 — Generate reliability diagram data
    # Reliability diagram shows how well calibrated the model is
    # x-axis: mean predicted probability in each bin
    # y-axis: fraction of actual positives in each bin
    # Perfect calibration: x == y (diagonal line)
    reliability_df = _compute_reliability_diagram(
        y_true=y,
        y_prob_raw=raw_probs,
        y_prob_calibrated=cal_probs,
        n_bins=10,
        week_str=week_str,
        city_fips=full_fips,
    )

    # Step 4 — Write calibration metrics to BigQuery
    write_dataframe(reliability_df, "calibration_metrics", config)

    # Step 5 — Log to MLflow
    with mlflow.start_run(run_name="calibration"):
        mlflow.set_tag("city_fips", full_fips)
        mlflow.log_metric("brier_before_calibration", brier_before)
        mlflow.log_metric("brier_after_calibration", brier_after)
        mlflow.log_metric("brier_improvement", brier_improvement)
        mlflow.log_metric("calibration_method", 0)  # 0=sigmoid, 1=isotonic
        mlflow.sklearn.log_model(calibrated_model, "calibrated_champion")

    logger.info(
        f"Calibration complete. "
        f"Brier improved by {brier_improvement:+.4f}. "
        f"Calibrated model saved to MLflow."
    )

    return calibrated_model


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------

def _compute_reliability_diagram(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    y_prob_calibrated: np.ndarray,
    n_bins: int,
    week_str: str,
    city_fips: str,
) -> pd.DataFrame:
    """
    Compute reliability diagram data for before and after calibration.

    A reliability diagram plots predicted probability (x-axis) against
    actual fraction of positives (y-axis) across probability bins.

    Perfect calibration: points fall on the diagonal (y = x).
    Overconfident model: points fall below diagonal (predicts too high).
    Underconfident model: points fall above diagonal (predicts too low).

    This data powers the calibration plot in the Streamlit dashboard —
    showing analysts how trustworthy the model's probabilities are.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_prob_raw : np.ndarray
        Raw (uncalibrated) predicted probabilities.
    y_prob_calibrated : np.ndarray
        Calibrated predicted probabilities.
    n_bins : int
        Number of probability bins. Default 10 (each bin = 10% range).
    week_str : str
        Week string for BigQuery metadata.
    city_fips : str
        City FIPS for BigQuery metadata.

    Returns
    -------
    pd.DataFrame
        One row per probability bin with columns:
        bin_mean_predicted, fraction_positive_raw,
        fraction_positive_calibrated, bin_count, week_str, city_fips
    """
    # calibration_curve returns (fraction_of_positives, mean_predicted_value)
    # for each bin
    frac_pos_raw, mean_pred_raw = calibration_curve(
        y_true, y_prob_raw,
        n_bins=n_bins,
        strategy="uniform",
    )

    frac_pos_cal, mean_pred_cal = calibration_curve(
        y_true, y_prob_calibrated,
        n_bins=n_bins,
        strategy="uniform",
    )

    # Build DataFrame — use calibrated predictions as the x-axis reference
    # since both curves share the same bins
    min_len = min(len(mean_pred_raw), len(mean_pred_cal))

    df = pd.DataFrame({
        "bin_mean_predicted_raw": mean_pred_raw[:min_len].round(4),
        "bin_mean_predicted_cal": mean_pred_cal[:min_len].round(4),
        "fraction_positive_raw": frac_pos_raw[:min_len].round(4),
        "fraction_positive_calibrated": frac_pos_cal[:min_len].round(4),
        "week_str": week_str,
        "city_fips": city_fips,
    })

    # Log calibration quality — how close to diagonal?
    # Mean absolute calibration error (MACE) — lower is better
    mace_raw = np.mean(np.abs(frac_pos_raw - mean_pred_raw))
    mace_cal = np.mean(np.abs(frac_pos_cal - mean_pred_cal))

    logger.info(
        f"Mean Absolute Calibration Error: "
        f"before={mace_raw:.4f}, after={mace_cal:.4f} "
        f"(0.0 = perfect, lower is better)"
    )

    return df