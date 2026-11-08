"""
evaluate.py
===========
Evaluate and compare Models A, B, and C using rigorous statistical tests.

Three evaluations:
  1. DeLong's test: is Model B's AUC significantly better than Model A's?
     (correct test for comparing correlated AUC scores from same CV folds)
  2. Brier score: how well calibrated are the probability predictions?
     (a model saying 0.8 risk should be right 80% of the time)
  3. SHAP values: which features drive each tract's prediction?
     (used in Streamlit dashboard and equity audit)

The A/B test result — DeLong's p-value and AUC lift — is the number
you cite in interviews and on your resume.

Input:
    MLflow: trained model artifacts for Models A, B, C
    BigQuery: ml_features_model_a, ml_features_model_b

Output:
    BigQuery: model_evaluation (AUC, Brier, DeLong results)
    BigQuery: shap_values (feature importance per tract)
    MLflow: evaluation metrics logged to each run

Usage:
    from src.ml.evaluate import evaluate_all_models
    results = evaluate_all_models(config, model_a, model_b, model_c,
                                   X_a, X_b, y, groups)
"""

from typing import Any

import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score

import mlflow
from src.ml.spatial_cv import SpatialBlockCV
from src.utils.bigquery import write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def evaluate_all_models(
    config: dict[str, Any],
    model_a,
    model_b,
    model_c,
    X_a: np.ndarray,
    X_b: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names_a: list[str],
    feature_names_b: list[str],
    week_str: str,
) -> dict[str, Any]:
    """
    Run full evaluation suite on all three models.

    Parameters
    ----------
    config : dict
        Merged pipeline config.
    model_a, model_b, model_c : trained model objects
        Models from train.py.
    X_a : np.ndarray
        Model A feature matrix (tabular only).
    X_b : np.ndarray
        Model B/C feature matrix (with spatial lags).
    y : np.ndarray
        Binary target vector.
    groups : np.ndarray
        H3 spatial block labels for CV.
    feature_names_a, feature_names_b : list of str
        Feature names for SHAP interpretation.
    week_str : str
        Current week as "YYYY-MM-DD" string.

    Returns
    -------
    dict
        Evaluation results including AUC scores, DeLong p-value,
        Brier scores, and SHAP DataFrames.
    """
    full_fips = config["full_fips"]
    cv = SpatialBlockCV(
        n_splits=config["ml"]["cv_folds"],
        random_seed=config["ml"]["random_seed"],
    )

    logger.info(f"Evaluating all models for {config['city_name']}")

    # Step 1 — Get out-of-fold predictions for all models
    # These are predictions on the test fold — data the model
    # never saw during training — the honest evaluation
    logger.info("Computing out-of-fold predictions")
    oof_preds_a = _get_oof_predictions(model_a, X_a, y, cv, groups)
    oof_preds_b = _get_oof_predictions(model_b, X_b, y, cv, groups)
    oof_preds_c = _get_oof_predictions(model_c, X_b, y, cv, groups)

    # Step 2 — Compute AUC for each model
    auc_a = roc_auc_score(y, oof_preds_a)
    auc_b = roc_auc_score(y, oof_preds_b)
    auc_c = roc_auc_score(y, oof_preds_c)

    logger.info(f"AUC scores: A={auc_a:.4f}, B={auc_b:.4f}, C={auc_c:.4f}")

    # Step 3 — DeLong's test: A vs B
    # Tests whether Model B's AUC is significantly better than Model A's
    delong_stat, delong_pvalue = _delongs_test(y, oof_preds_a, oof_preds_b)
    auc_lift = auc_b - auc_a
    is_significant = delong_pvalue < config["ml"]["ab_test_significance_threshold"]

    logger.info(
        f"DeLong's test (A vs B): "
        f"AUC lift={auc_lift:+.4f}, "
        f"p={delong_pvalue:.4f}, "
        f"significant={is_significant}"
    )

    # Step 4 — Brier score for each model
    # Lower is better — 0.0 is perfect, 0.25 is random
    brier_a = brier_score_loss(y, oof_preds_a)
    brier_b = brier_score_loss(y, oof_preds_b)
    brier_c = brier_score_loss(y, oof_preds_c)

    logger.info(
        f"Brier scores: A={brier_a:.4f}, B={brier_b:.4f}, C={brier_c:.4f} "
        f"(lower is better)"
    )

    # Step 5 — SHAP values for champion model (Model B or C)
    # Use Model C if it has better Brier score (better calibration)
    # Use Model B otherwise
    champion = model_c if brier_c < brier_b else model_b
    champion_X = X_b
    champion_features = feature_names_b
    champion_name = "model_c" if brier_c < brier_b else "model_b"

    logger.info(f"Computing SHAP values for champion model: {champion_name}")
    shap_df = _compute_shap_values(
        champion,
        champion_X,
        champion_features,
        config,
        week_str,
    )

    # Step 6 — Log all results to MLflow
    _log_to_mlflow(
        config=config,
        auc_a=auc_a,
        auc_b=auc_b,
        auc_c=auc_c,
        brier_a=brier_a,
        brier_b=brier_b,
        brier_c=brier_c,
        delong_stat=delong_stat,
        delong_pvalue=delong_pvalue,
        auc_lift=auc_lift,
        is_significant=is_significant,
        champion_name=champion_name,
    )

    # Step 7 — Write evaluation results to BigQuery
    eval_df = pd.DataFrame(
        [
            {
                "week_str": week_str,
                "city_fips": full_fips,
                "auc_model_a": round(auc_a, 6),
                "auc_model_b": round(auc_b, 6),
                "auc_model_c": round(auc_c, 6),
                "auc_lift_a_to_b": round(auc_lift, 6),
                "delong_statistic": round(delong_stat, 6),
                "delong_pvalue": round(delong_pvalue, 6),
                "delong_significant": is_significant,
                "brier_model_a": round(brier_a, 6),
                "brier_model_b": round(brier_b, 6),
                "brier_model_c": round(brier_c, 6),
                "champion_model": champion_name,
            }
        ]
    )

    write_dataframe(eval_df, "model_evaluation", config)
    write_dataframe(shap_df, "shap_values", config)

    logger.info(
        f"Evaluation complete. "
        f"Champion: {champion_name}. "
        f"AUC lift A→B: {auc_lift:+.4f} "
        f"({'significant' if is_significant else 'NOT significant'}, "
        f"p={delong_pvalue:.4f})"
    )

    return {
        "auc_a": auc_a,
        "auc_b": auc_b,
        "auc_c": auc_c,
        "auc_lift": auc_lift,
        "delong_pvalue": delong_pvalue,
        "is_significant": is_significant,
        "brier_a": brier_a,
        "brier_b": brier_b,
        "brier_c": brier_c,
        "champion_model": champion_name,
        "shap_df": shap_df,
    }


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------


def _get_oof_predictions(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: SpatialBlockCV,
    groups: np.ndarray,
) -> np.ndarray:
    """
    Get out-of-fold predictions for all samples.

    For each CV fold, train on the training set and predict on
    the test set. Collect all test set predictions into one array
    the same length as y.

    Why out-of-fold predictions instead of just using CV scores?
    Because DeLong's test and Brier score need the actual predicted
    probabilities per sample — not just the fold-level AUC averages.
    OOF predictions give us one probability per tract, estimated on
    data the model never saw during that fold's training.

    Parameters
    ----------
    model : sklearn-compatible classifier
        Model to evaluate.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target vector.
    cv : SpatialBlockCV
        Spatial cross-validator.
    groups : np.ndarray
        H3 block labels.

    Returns
    -------
    np.ndarray
        Predicted probabilities, one per sample, shape (n_samples,).
    """
    oof_predictions = np.zeros(len(y))

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, groups=groups)):
        # Clone model to avoid state leakage between folds
        from sklearn.base import clone

        fold_model = clone(model)

        # Train on this fold's training set
        fold_model.fit(X[train_idx], y[train_idx])

        # Predict probabilities on the test set
        # predict_proba returns [prob_class_0, prob_class_1]
        # we want prob_class_1 = probability of being high risk
        oof_predictions[test_idx] = fold_model.predict_proba(X[test_idx])[:, 1]

        fold_auc = roc_auc_score(y[test_idx], oof_predictions[test_idx])
        logger.debug(f"Fold {fold + 1} AUC: {fold_auc:.4f}")

    return oof_predictions


def _delongs_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> tuple[float, float]:
    """
    DeLong's test for comparing two correlated AUC scores.

    Why DeLong's and not a t-test?
    Both models were tested on the same samples (same CV folds).
    Their predictions are correlated — if a tract is easy to
    classify, both models get it right. A t-test assumes the two
    AUC scores are independent, which is wrong here. DeLong's test
    accounts for this correlation using the covariance structure
    of the AUC estimators.

    Implementation based on:
    DeLong, E.R., DeLong, D.M., Clarke-Pearson, D.L. (1988).
    "Comparing the areas under two or more correlated receiver
    operating characteristic curves: a nonparametric approach."
    Biometrics, 44(3), 837-845.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_pred_a : np.ndarray
        Predicted probabilities from Model A.
    y_pred_b : np.ndarray
        Predicted probabilities from Model B.

    Returns
    -------
    tuple of (z_statistic, p_value)
        z_statistic: DeLong Z-score
        p_value: two-sided p-value (< 0.05 = significant difference)
    """
    # Separate predictions by true class
    pos_mask = y_true == 1
    neg_mask = y_true == 0

    # Predictions on positive and negative samples
    pos_a = y_pred_a[pos_mask]
    neg_a = y_pred_a[neg_mask]
    pos_b = y_pred_b[pos_mask]
    neg_b = y_pred_b[neg_mask]

    n_pos = len(pos_a)
    n_neg = len(neg_a)

    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "DeLong's test requires both positive and negative samples. "
            f"Got {n_pos} positive, {n_neg} negative."
        )

    # Compute AUC components using placement values
    # V10[i] = P(score_A(positive_i) > score_A(negative_j))
    # These are the structural components of the AUC estimator
    def placement_values_10(pos_scores, neg_scores):
        return np.array(
            [
                np.mean(pos_score > neg_scores) + 0.5 * np.mean(pos_score == neg_scores)
                for pos_score in pos_scores
            ]
        )

    def placement_values_01(pos_scores, neg_scores):
        return np.array(
            [
                np.mean(pos_scores > neg_score) + 0.5 * np.mean(pos_scores == neg_score)
                for neg_score in neg_scores
            ]
        )

    v10_a = placement_values_10(pos_a, neg_a)
    v01_a = placement_values_01(pos_a, neg_a)
    v10_b = placement_values_10(pos_b, neg_b)
    v01_b = placement_values_01(pos_b, neg_b)

    # AUC estimates
    auc_a = np.mean(v10_a)
    auc_b = np.mean(v10_b)

    # Variance-covariance matrix of the two AUC estimators
    # This is where DeLong accounts for the correlation
    # between the two models' predictions
    s10 = np.cov(np.vstack([v10_a, v10_b]))
    s01 = np.cov(np.vstack([v01_a, v01_b]))

    s = s10 / n_pos + s01 / n_neg

    # Z-test: difference in AUCs divided by standard error of difference
    # contrast = [1, -1] computes AUC_B - AUC_A
    contrast = np.array([1, -1])
    auc_diff = auc_b - auc_a
    se_diff = np.sqrt(contrast @ s @ contrast)

    if se_diff == 0:
        logger.warning(
            "Standard error of AUC difference is zero — models may be identical"
        )
        return 0.0, 1.0

    z_stat = auc_diff / se_diff

    # Two-sided p-value from standard normal distribution
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    logger.debug(
        f"DeLong's test: AUC_A={auc_a:.4f}, AUC_B={auc_b:.4f}, "
        f"diff={auc_diff:+.4f}, SE={se_diff:.4f}, "
        f"Z={z_stat:.4f}, p={p_value:.4f}"
    )

    return float(z_stat), float(p_value)


def _compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: list[str],
    config: dict[str, Any],
    week_str: str,
) -> pd.DataFrame:
    """
    Compute SHAP feature importance values for the champion model.

    SHAP (SHapley Additive exPlanations) explains each prediction
    by assigning each feature a contribution value:
      - Positive SHAP: this feature increased the predicted risk
      - Negative SHAP: this feature decreased the predicted risk
      - Magnitude: how much did this feature contribute?

    Example for one tract:
        base_value:          0.52  (average prediction across all tracts)
        + h3_lag_ring1:     +0.18  (high-risk neighbors increased risk)
        + score_current:    +0.14  (high current score increased risk)
        + asthma_pct:       +0.09  (high asthma rate increased risk)
        + median_income:    -0.02  (moderate income slightly reduced risk)
        = final prediction:  0.91

    This breakdown appears in the Streamlit dashboard when you click
    on any census tract — telling the analyst exactly WHY that tract
    is high risk, not just that it is.

    Parameters
    ----------
    model : trained classifier
        Champion model (Model B or C).
    X : np.ndarray
        Feature matrix for all tracts.
    feature_names : list of str
        Feature names in same order as X columns.
    config : dict
        Pipeline config.
    week_str : str
        Current week string for BigQuery metadata.

    Returns
    -------
    pd.DataFrame
        SHAP values per tract per feature.
        Columns: geoid (implicit via index), week_str, city_fips,
                 shap_{feature_name} for each feature.
    """
    n_background = config["ml"]["shap_background_samples"]

    logger.info(
        f"Computing SHAP values for {X.shape[0]:,} tracts, {X.shape[1]} features"
    )

    # TreeExplainer is optimized for tree-based models
    # (XGBoost and LightGBM are both tree-based)
    # Much faster than the general KernelExplainer
    explainer = shap.TreeExplainer(
        model,
        data=X[:n_background],  # background dataset for base value
        feature_names=feature_names[: X.shape[1]],
    )

    # Compute SHAP values for all tracts
    # Returns array of shape (n_tracts, n_features)
    # Each row: how much each feature contributed to that tract's prediction
    shap_values = explainer.shap_values(X)

    # Handle binary classification output format
    # Some models return list of [shap_class_0, shap_class_1]
    # We want class 1 (high risk) SHAP values
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    logger.info(
        "SHAP values computed. "
        "Mean absolute SHAP by feature:\n"
        + "\n".join(
            [
                f"  {name}: {abs_val:.4f}"
                for name, abs_val in sorted(
                    zip(
                        feature_names[: X.shape[1]],
                        np.abs(shap_values).mean(axis=0),
                        strict=False,
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
            ]
        )
    )

    # Convert to DataFrame
    shap_df = pd.DataFrame(
        shap_values,
        columns=[f"shap_{name}" for name in feature_names[: X.shape[1]]],
    )

    # Add metadata
    shap_df["week_str"] = week_str
    shap_df["city_fips"] = config["full_fips"]

    return shap_df


def _log_to_mlflow(
    config: dict[str, Any],
    auc_a: float,
    auc_b: float,
    auc_c: float,
    brier_a: float,
    brier_b: float,
    brier_c: float,
    delong_stat: float,
    delong_pvalue: float,
    auc_lift: float,
    is_significant: bool,
    champion_name: str,
) -> None:
    """
    Log all evaluation metrics to the MLflow experiment.

    Creates a dedicated evaluation run in MLflow that consolidates
    all three model comparison metrics in one place.
    """
    with mlflow.start_run(run_name="ab_test_evaluation"):
        mlflow.set_tag("city_fips", config["full_fips"])
        mlflow.set_tag("champion_model", champion_name)

        # AUC comparison
        mlflow.log_metric("auc_model_a", auc_a)
        mlflow.log_metric("auc_model_b", auc_b)
        mlflow.log_metric("auc_model_c", auc_c)
        mlflow.log_metric("auc_lift_a_to_b", auc_lift)

        # DeLong's test results
        mlflow.log_metric("delong_z_statistic", delong_stat)
        mlflow.log_metric("delong_pvalue", delong_pvalue)
        mlflow.log_metric("ab_test_significant", int(is_significant))

        # Brier scores
        mlflow.log_metric("brier_model_a", brier_a)
        mlflow.log_metric("brier_model_b", brier_b)
        mlflow.log_metric("brier_model_c", brier_c)
        mlflow.log_metric("brier_improvement_b_vs_c", brier_b - brier_c)

    logger.info("Evaluation metrics logged to MLflow")
