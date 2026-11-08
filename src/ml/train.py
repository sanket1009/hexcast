"""
train.py
========
Train Models A, B, and C for the CHBIndex compound burden prediction task.

Model A: XGBoost — tabular features only (baseline)
Model B: XGBoost — tabular + H3 spatial lag features (challenger)
Model C: LightGBM — all features + monotonicity constraints (clinical)

All three models are trained using blocked spatial cross-validation
to prevent geographic data leakage. All experiments are tracked
in MLflow with parameters, metrics, and model artifacts.

The A/B test comparing Models A and B is run in evaluate.py using
DeLong's test for statistical significance of AUC difference.

Input:
    BigQuery: ml_features_model_a (tabular features)
    BigQuery: ml_features_model_b (spatial lag features)

Output:
    MLflow: experiment runs for all three models
    MLflow: model artifacts (saved model files)
    BigQuery: model_cv_scores (CV scores per fold per model)

Usage:
    from src.ml.train import train_all_models
    results = train_all_models(config)
"""

from datetime import datetime, timedelta
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from src.ml.spatial_cv import SpatialBlockCV, build_spatial_groups, summarize_cv_folds
from src.utils.bigquery import read_query, write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Feature columns for each model
# Model A: temporal + socioeconomic only
MODEL_A_FEATURES = [
    "score_current",
    "lag_1w",
    "lag_2w",
    "lag_4w",
    "rolling_mean_4w",
    "rolling_std_4w",
    "delta_1w",
    "delta_4w",
    "weeks_high_risk_4w",
    "is_summer",
    "week_of_year",
    "vulnerability_index",
    "median_income_norm",
    "asthma_pct",
    "copd_pct",
    "heart_disease_pct",
    "population",
]

# Model B: Model A + H3 spatial lags
MODEL_B_FEATURES = MODEL_A_FEATURES + [
    "h3_lag_ring1",
    "h3_lag_ring2",
    "h3_lag_diff",
    "h3_max_neighbor",
    "h3_high_risk_neighbors",
]

# Target variable
TARGET = "is_high_risk_next_week"


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------


def train_all_models(
    config: dict[str, Any],
    week_start: datetime | None = None,
) -> dict[str, Any]:
    """
    Train Models A, B, and C with spatial CV and MLflow tracking.

    Parameters
    ----------
    config : dict
        Merged pipeline config.
    week_start : datetime, optional
        Week to train for. Defaults to most recent Monday.

    Returns
    -------
    dict
        Results dictionary with keys:
        model_a, model_b, model_c: trained model objects
        scores_a, scores_b, scores_c: CV AUC scores per fold
        feature_importance_a, feature_importance_b, feature_importance_c
    """
    if week_start is None:
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

    week_str = week_start.strftime("%Y-%m-%d")
    city = config["city_name"]
    random_seed = config["ml"]["random_seed"]

    logger.info(f"Training all models for {city} week of {week_str}")

    # Step 1 — Set up MLflow
    _setup_mlflow(config)

    # Step 2 — Load feature matrices from BigQuery
    df_a = _load_features(config, "ml_features_model_a", week_str)
    df_b = _load_features(config, "ml_features_model_b", week_str)
    logger.info(
        f"Loaded features: Model A={len(df_a):,} rows, Model B={len(df_b):,} rows"
    )

    # Step 3 — Build spatial CV groups
    # Both models use identical groups for fair comparison
    groups = build_spatial_groups(
        geoids=df_a["geoid"],
        centroids_lat=df_a.get("centroid_lat", pd.Series([None] * len(df_a))),
        centroids_lon=df_a.get("centroid_lon", pd.Series([None] * len(df_a))),
    )

    cv = SpatialBlockCV(
        n_splits=config["ml"]["cv_folds"],
        random_seed=random_seed,
    )

    # Log CV fold summary
    X_dummy = np.zeros((len(df_a), 1))
    fold_summary = summarize_cv_folds(cv, X_dummy, groups)
    logger.info(f"CV fold summary:\n{fold_summary.to_string(index=False)}")

    # Step 4 — Prepare feature matrices and targets
    X_a, y = _prepare_features(df_a, MODEL_A_FEATURES, TARGET)
    X_b, _ = _prepare_features(df_b, MODEL_B_FEATURES, TARGET)

    # Step 5 — Train Model A (XGBoost, tabular only)
    logger.info("Training Model A — XGBoost tabular baseline")
    model_a, scores_a, importance_a = _train_xgboost(
        X=X_a,
        y=y,
        feature_names=MODEL_A_FEATURES,
        groups=groups,
        cv=cv,
        config=config,
        model_name="model_a_xgboost_tabular",
        experiment_tags={
            "model_variant": "A",
            "spatial_features": "false",
            "city": city,
            "week": week_str,
        },
    )

    # Step 6 — Train Model B (XGBoost, with spatial lags)
    logger.info("Training Model B — XGBoost with H3 spatial lags")
    model_b, scores_b, importance_b = _train_xgboost(
        X=X_b,
        y=y,
        feature_names=MODEL_B_FEATURES,
        groups=groups,
        cv=cv,
        config=config,
        model_name="model_b_xgboost_spatial",
        experiment_tags={
            "model_variant": "B",
            "spatial_features": "true",
            "city": city,
            "week": week_str,
        },
    )

    # Step 7 — Train Model C (LightGBM with monotonicity constraints)
    logger.info("Training Model C — LightGBM with monotonicity constraints")
    model_c, scores_c, importance_c = _train_lightgbm(
        X=X_b,  # uses same features as Model B
        y=y,
        feature_names=MODEL_B_FEATURES,
        groups=groups,
        cv=cv,
        config=config,
        model_name="model_c_lightgbm_monotone",
        experiment_tags={
            "model_variant": "C",
            "spatial_features": "true",
            "monotonicity_constraints": "true",
            "city": city,
            "week": week_str,
        },
    )

    # Step 8 — Log comparison summary
    logger.info(
        f"Training complete:\n"
        f"  Model A AUC: {np.mean(scores_a):.4f} ± {np.std(scores_a):.4f}\n"
        f"  Model B AUC: {np.mean(scores_b):.4f} ± {np.std(scores_b):.4f}\n"
        f"  Model C AUC: {np.mean(scores_c):.4f} ± {np.std(scores_c):.4f}\n"
        f"  A→B lift: {np.mean(scores_b) - np.mean(scores_a):+.4f} AUC points"
    )

    # Step 9 — Write CV scores to BigQuery
    _write_cv_scores(
        scores_a,
        scores_b,
        scores_c,
        week_str,
        config,
    )

    return {
        "model_a": model_a,
        "model_b": model_b,
        "model_c": model_c,
        "scores_a": scores_a,
        "scores_b": scores_b,
        "scores_c": scores_c,
        "feature_importance_a": importance_a,
        "feature_importance_b": importance_b,
        "feature_importance_c": importance_c,
        "auc_lift_a_to_b": float(np.mean(scores_b) - np.mean(scores_a)),
    }


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------


def _setup_mlflow(config: dict[str, Any]) -> None:
    """
    Configure MLflow tracking URI and experiment name.

    MLflow stores all experiment data in the directory set by
    tracking URI. We use the local mlflow/mlruns/ directory —
    access the UI at localhost:5000 by running:
        mlflow ui --port 5000

    Parameters
    ----------
    config : dict
        Pipeline config with MLflow tracking URI.
    """
    tracking_uri = config["ml"]["mlflow_tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)

    # Experiment name groups related runs together in the MLflow UI
    # All CHBIndex runs go under "chbindex_{city_fips}"
    experiment_name = f"chbindex_{config['full_fips']}"
    mlflow.set_experiment(experiment_name)

    logger.debug(
        f"MLflow configured: tracking_uri={tracking_uri}, experiment={experiment_name}"
    )


def _load_features(
    config: dict[str, Any],
    table_name: str,
    week_str: str,
) -> pd.DataFrame:
    """Load feature matrix from BigQuery for the current week."""
    df = read_query(
        f"""
        SELECT *
        FROM {{dataset}}.{table_name}
        WHERE city_fips = '{config["full_fips"]}'
        ORDER BY geoid
        """,
        config,
    )

    if df.empty:
        raise ValueError(f"No features found in {table_name}. Run features.py first.")

    # Drop rows where target is null
    # These are weeks where next-week data doesn't exist yet
    before = len(df)
    df = df.dropna(subset=[TARGET])
    after = len(df)

    if before != after:
        logger.warning(
            f"Dropped {before - after} rows with null target "
            f"(next week not yet available)"
        )

    return df


def _prepare_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and target vector y from DataFrame.

    Fills any remaining NaN values with column median —
    gradient boosting can handle NaN natively but explicit
    imputation makes the pipeline more transparent.

    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame.
    feature_cols : list of str
        Which columns to use as features.
    target_col : str
        Target column name.

    Returns
    -------
    tuple of (X, y)
        X: feature matrix as numpy array
        y: target vector as numpy array
    """
    # Keep only feature columns that exist in df
    available_cols = [c for c in feature_cols if c in df.columns]
    missing_cols = set(feature_cols) - set(available_cols)

    if missing_cols:
        logger.warning(f"Missing feature columns: {missing_cols}")

    X_df = df[available_cols].copy()

    # Impute remaining NaN with column median
    for col in X_df.columns:
        if X_df[col].isna().any():
            X_df[col] = X_df[col].fillna(X_df[col].median())

    X = X_df.values
    y = df[target_col].values.astype(int)

    logger.debug(f"Feature matrix: {X.shape}, positive rate: {y.mean():.2%}")

    return X, y


def _train_xgboost(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    groups: np.ndarray,
    cv: SpatialBlockCV,
    config: dict[str, Any],
    model_name: str,
    experiment_tags: dict[str, str],
) -> tuple[xgb.XGBClassifier, np.ndarray, pd.DataFrame]:
    """
    Train XGBoost model with spatial CV and log to MLflow.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary target vector.
    feature_names : list of str
        Names for feature importance logging.
    groups : np.ndarray
        H3 parent cell group labels for spatial CV.
    cv : SpatialBlockCV
        Blocked spatial cross-validator.
    config : dict
        Pipeline config with XGBoost hyperparameters.
    model_name : str
        Name for MLflow run.
    experiment_tags : dict
        Tags to log with this MLflow run.

    Returns
    -------
    tuple of (model, cv_scores, feature_importance)
    """
    xgb_params = config["ml"]["xgboost"]
    random_seed = config["ml"]["random_seed"]

    model = xgb.XGBClassifier(
        n_estimators=xgb_params["n_estimators"],
        learning_rate=xgb_params["learning_rate"],
        max_depth=xgb_params["max_depth"],
        subsample=xgb_params["subsample"],
        colsample_bytree=xgb_params["colsample_bytree"],
        early_stopping_rounds=xgb_params["early_stopping_rounds"],
        random_state=random_seed,
        eval_metric="auc",
        use_label_encoder=False,
        verbosity=0,
    )

    with mlflow.start_run(run_name=model_name):
        # Log experiment tags
        mlflow.set_tags(experiment_tags)

        # Log hyperparameters
        mlflow.log_params(
            {
                "n_estimators": xgb_params["n_estimators"],
                "learning_rate": xgb_params["learning_rate"],
                "max_depth": xgb_params["max_depth"],
                "subsample": xgb_params["subsample"],
                "n_features": X.shape[1],
                "n_samples": X.shape[0],
                "cv_folds": config["ml"]["cv_folds"],
                "h3_resolution": config["spatial"]["h3_resolution"],
            }
        )

        # Run spatial cross-validation
        # scoring="roc_auc" uses AUC as the evaluation metric
        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            groups=groups,
            scoring="roc_auc",
        )

        # Log CV metrics to MLflow
        mlflow.log_metric("cv_auc_mean", float(np.mean(cv_scores)))
        mlflow.log_metric("cv_auc_std", float(np.std(cv_scores)))
        mlflow.log_metric("cv_auc_min", float(np.min(cv_scores)))
        mlflow.log_metric("cv_auc_max", float(np.max(cv_scores)))

        for i, score in enumerate(cv_scores):
            mlflow.log_metric(f"cv_auc_fold_{i + 1}", float(score))

        # Fit final model on ALL data
        # CV gives us the performance estimate
        # Final model trained on all data is what we deploy
        model.fit(X, y)

        # Log model artifact to MLflow
        mlflow.xgboost.log_model(model, model_name)

        # Compute and log feature importance
        importance_df = pd.DataFrame(
            {
                "feature": feature_names[: X.shape[1]],
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        logger.info(
            f"{model_name} CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}"
        )
        logger.info(f"Top 5 features:\n{importance_df.head(5).to_string(index=False)}")

    return model, cv_scores, importance_df


def _train_lightgbm(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    groups: np.ndarray,
    cv: SpatialBlockCV,
    config: dict[str, Any],
    model_name: str,
    experiment_tags: dict[str, str],
) -> tuple[lgb.LGBMClassifier, np.ndarray, pd.DataFrame]:
    """
    Train LightGBM Model C with monotonicity constraints.

    Monotonicity constraints enforce that:
        higher PM2.5    → higher risk (constraint: +1)
        higher LST      → higher risk (constraint: +1)
        higher access   → lower risk  (constraint: -1)
        higher income   → lower risk  (constraint: -1)

    These make the model clinically defensible — it cannot learn
    spurious relationships where more pollution correlates with
    less risk in some data subset.

    Monotonicity constraint values:
        +1 = feature must increase monotonically with prediction
        -1 = feature must decrease monotonically with prediction
         0 = no constraint (most features)

    Parameters
    ----------
    X, y, feature_names, groups, cv, config, model_name, experiment_tags
        Same as _train_xgboost().

    Returns
    -------
    tuple of (model, cv_scores, feature_importance)
    """
    lgb_params = config["ml"]["lightgbm"]
    random_seed = config["ml"]["random_seed"]

    # Build monotonicity constraints array
    # One value per feature — must match feature order exactly
    # 0 = no constraint (default for most features)
    # +1 = higher feature value must increase predicted risk
    # -1 = higher feature value must decrease predicted risk
    monotone_constraints = _build_monotone_constraints(feature_names[: X.shape[1]])

    model = lgb.LGBMClassifier(
        n_estimators=lgb_params["n_estimators"],
        learning_rate=lgb_params["learning_rate"],
        num_leaves=lgb_params["num_leaves"],
        subsample=lgb_params["subsample"],
        colsample_bytree=lgb_params["colsample_bytree"],
        monotone_constraints=monotone_constraints,
        monotone_constraints_method="advanced",
        random_state=random_seed,
        verbose=-1,
    )

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tags(experiment_tags)

        mlflow.log_params(
            {
                "n_estimators": lgb_params["n_estimators"],
                "learning_rate": lgb_params["learning_rate"],
                "num_leaves": lgb_params["num_leaves"],
                "n_features": X.shape[1],
                "n_monotone_constraints": sum(
                    1 for c in monotone_constraints if c != 0
                ),
            }
        )

        # Log which features have constraints
        constrained_features = [
            f"{name}:{c}"
            for name, c in zip(
                feature_names[: X.shape[1]], monotone_constraints, strict=False
            )
            if c != 0
        ]
        mlflow.log_param(
            "monotone_constrained_features",
            str(constrained_features),
        )

        cv_scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            groups=groups,
            scoring="roc_auc",
        )

        mlflow.log_metric("cv_auc_mean", float(np.mean(cv_scores)))
        mlflow.log_metric("cv_auc_std", float(np.std(cv_scores)))

        for i, score in enumerate(cv_scores):
            mlflow.log_metric(f"cv_auc_fold_{i + 1}", float(score))

        model.fit(X, y)
        mlflow.sklearn.log_model(model, model_name)

        importance_df = pd.DataFrame(
            {
                "feature": feature_names[: X.shape[1]],
                "importance": model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        logger.info(
            f"{model_name} CV AUC: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}"
        )

    return model, cv_scores, importance_df


def _build_monotone_constraints(
    feature_names: list[str],
) -> list[int]:
    """
    Build monotonicity constraint array for LightGBM Model C.

    Constraints enforce clinically sensible relationships:
        More pollution → more risk (+1)
        More heat → more risk (+1)
        More healthcare access → less risk (-1)
        More income → less risk (-1)

    All other features get constraint 0 (unconstrained).

    Parameters
    ----------
    feature_names : list of str
        Feature names in the exact order they appear in X.

    Returns
    -------
    list of int
        One constraint value per feature: +1, -1, or 0.
    """
    # Map feature name patterns to constraint direction
    # Pattern matching handles feature names robustly
    positive_features = [
        # Higher = more risk (+1)
        "pm25",
        "lst",
        "lag_",
        "score_current",
        "rolling_mean",
        "delta_",
        "weeks_high_risk",
        "h3_lag",
        "h3_max",
        "h3_high_risk",
        "asthma",
        "copd",
        "heart_disease",
        "vulnerability",
    ]
    negative_features = [
        # Higher = less risk (-1)
        "access_score",
        "median_income_norm",
    ]

    constraints = []
    for feature in feature_names:
        feature_lower = feature.lower()

        if any(p in feature_lower for p in positive_features):
            constraints.append(1)
        elif any(p in feature_lower for p in negative_features):
            constraints.append(-1)
        else:
            constraints.append(0)

    constrained_count = sum(1 for c in constraints if c != 0)
    logger.debug(
        f"Monotonicity constraints: {constrained_count}/{len(constraints)} "
        f"features constrained"
    )

    return constraints


def _write_cv_scores(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    scores_c: np.ndarray,
    week_str: str,
    config: dict[str, Any],
) -> None:
    """Write CV scores for all three models to BigQuery."""
    records = []
    for fold, (sa, sb, sc) in enumerate(
        zip(scores_a, scores_b, scores_c, strict=False)
    ):
        records.append(
            {
                "fold": fold + 1,
                "model_a_auc": round(float(sa), 6),
                "model_b_auc": round(float(sb), 6),
                "model_c_auc": round(float(sc), 6),
                "week_str": week_str,
                "city_fips": config["full_fips"],
            }
        )

    df = pd.DataFrame(records)
    write_dataframe(df, "model_cv_scores", config)
    logger.info("CV scores written to BigQuery")
