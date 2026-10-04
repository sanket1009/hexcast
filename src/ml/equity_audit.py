"""
equity_audit.py
===============
Audit model performance across demographic subgroups to identify
disparities in prediction accuracy.

Health ML models that perform well overall can systematically
underperform for the most vulnerable populations — often due to
data quality gaps rather than model bias per se.

For CHBIndex, the likely disparity source is sparse EPA monitor
coverage in low-income minority neighborhoods — Kriging uncertainty
is higher there, making PM2.5 features less reliable, which
degrades model performance for those tracts specifically.

Documenting this honestly — with root cause explanation and
proposed fixes — is more valuable than hiding it.

Stratification dimensions:
  1. Majority race/ethnicity (from ACS B03002)
  2. Income quartile (from ACS B19013 median household income)

Metrics computed per subgroup:
  - AUC: ranking performance
  - Precision at top decile: accuracy when flagging highest-risk tracts
  - Recall: fraction of truly high-risk tracts correctly flagged
  - Mean Kriging variance: data quality proxy (high variance = less reliable)

Input:
    BigQuery: ml_features_model_b (features + demographics)
    BigQuery: pm25_kriged (Kriging variance per tract)
    Out-of-fold predictions from evaluate.py
    True labels y

Output:
    BigQuery: equity_audit_results (one row per subgroup)
    MLflow: equity metrics logged

Usage:
    from src.ml.equity_audit import run_equity_audit
    audit_df = run_equity_audit(config, oof_predictions, y, df_features)
"""

from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils.bigquery import read_query, write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Minimum subgroup size to compute reliable metrics
# Groups smaller than this get flagged as unreliable
MIN_SUBGROUP_SIZE = 50


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def run_equity_audit(
    config: dict[str, Any],
    oof_predictions: np.ndarray,
    y: np.ndarray,
    df_features: pd.DataFrame,
    week_str: str,
) -> pd.DataFrame:
    """
    Audit model performance across demographic subgroups.

    Parameters
    ----------
    config : dict
        Merged pipeline config.
    oof_predictions : np.ndarray
        Out-of-fold predicted probabilities from evaluate.py.
        Shape: (n_tracts,) — one calibrated probability per tract.
    y : np.ndarray
        True binary labels (is_high_risk_next_week).
    df_features : pd.DataFrame
        Feature DataFrame containing geoid, income_quartile,
        and demographic columns.
    week_str : str
        Current week string for BigQuery metadata.

    Returns
    -------
    pd.DataFrame
        One row per subgroup with performance metrics.
        Includes disparity flags where performance gaps are found.
    """
    city = config["city_name"]
    full_fips = config["full_fips"]

    logger.info(f"Running equity audit for {city}")

    # Step 1 — Load race/ethnicity data from ACS
    race_df = _load_race_data(config)

    # Step 2 — Merge demographics onto features
    audit_df = df_features[["geoid", "income_quartile"]].copy()
    audit_df["y_true"] = y
    audit_df["y_pred_prob"] = oof_predictions
    audit_df["y_pred_binary"] = (oof_predictions >= 0.75).astype(int)

    audit_df = audit_df.merge(race_df, on="geoid", how="left")

    # Step 3 — Load Kriging variance as data quality proxy
    kriging_var_df = _load_kriging_variance(config, week_str)
    audit_df = audit_df.merge(kriging_var_df, on="geoid", how="left")

    # Step 4 — Compute metrics by income quartile
    logger.info("Computing metrics by income quartile")
    income_results = _compute_subgroup_metrics(
        audit_df,
        group_col="income_quartile",
        group_label="income_quartile",
        config=config,
    )

    # Step 5 — Compute metrics by majority race/ethnicity
    logger.info("Computing metrics by majority race/ethnicity")
    race_results = _compute_subgroup_metrics(
        audit_df,
        group_col="majority_race",
        group_label="majority_race",
        config=config,
    )

    # Step 6 — Combine results
    results_df = pd.concat(
        [income_results, race_results],
        ignore_index=True,
    )

    # Step 7 — Flag disparities
    results_df = _flag_disparities(results_df)

    # Step 8 — Add metadata
    results_df["week_str"] = week_str
    results_df["city_fips"] = full_fips

    # Step 9 — Log to MLflow and BigQuery
    _log_to_mlflow(results_df, config)
    write_dataframe(results_df, "equity_audit_results", config)

    # Step 10 — Log summary
    _log_summary(results_df)

    return results_df


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------

def _load_race_data(config: dict[str, Any]) -> pd.DataFrame:
    """
    Load majority race/ethnicity per tract from ACS B03002.

    ACS table B03002 provides population by race and Hispanic origin.
    We compute the majority group — the largest single group —
    for each tract.

    If no group has >50% population, the tract is classified as
    "mixed" — no single majority.

    Returns
    -------
    pd.DataFrame
        Columns: geoid, majority_race
        Values: "white", "hispanic", "black", "asian", "mixed"
    """
    df = read_query(
        f"""
        SELECT
            geoid,
            -- ACS B03002 race/ethnicity columns
            -- These were pulled during census_tiger ingestion
            -- as additional ACS variables
            white_alone_pct,
            hispanic_pct,
            black_alone_pct,
            asian_alone_pct,
            total_population
        FROM {{dataset}}.tract_geometries
        WHERE city_fips = '{config["full_fips"]}'
        ORDER BY geoid
        """,
        config,
    )

    if df.empty:
        logger.warning(
            "No race/ethnicity data found. "
            "equity audit will skip race stratification."
        )
        return pd.DataFrame(columns=["geoid", "majority_race"])

    # Determine majority group for each tract
    race_cols = {
        "white": "white_alone_pct",
        "hispanic": "hispanic_pct",
        "black": "black_alone_pct",
        "asian": "asian_alone_pct",
    }

    def get_majority(row: pd.Series) -> str:
        """Return majority race if any group > 50%, else 'mixed'."""
        max_group = None
        max_pct = 0.0

        for group, col in race_cols.items():
            pct = row.get(col, 0) or 0
            if pct > max_pct:
                max_pct = pct
                max_group = group

        # Only call it a majority if group exceeds 50%
        if max_pct >= 50.0:
            return max_group
        return "mixed"

    df["majority_race"] = df.apply(get_majority, axis=1)

    logger.info(
        f"Race distribution: "
        + ", ".join([
            f"{race}={count}"
            for race, count in df["majority_race"].value_counts().items()
        ])
    )

    return df[["geoid", "majority_race"]]


def _load_kriging_variance(
    config: dict[str, Any],
    week_str: str,
) -> pd.DataFrame:
    """
    Load Kriging PM2.5 prediction variance per tract.

    High Kriging variance = fewer nearby monitors = less reliable
    PM2.5 estimate = model features less reliable for this tract.

    This is the data quality proxy that explains performance gaps —
    if low-income minority tracts have higher Kriging variance,
    that explains lower model performance there.

    Returns
    -------
    pd.DataFrame
        Columns: geoid, pm25_kriged_variance
    """
    df = read_query(
        f"""
        SELECT geoid, pm25_kriged_variance
        FROM {{dataset}}.pm25_kriged
        WHERE
            city_fips = '{config["full_fips"]}'
            AND week_start_date = '{week_str}'
        ORDER BY geoid
        """,
        config,
    )

    if df.empty:
        logger.warning("No Kriging variance data found for this week.")
        return pd.DataFrame(columns=["geoid", "pm25_kriged_variance"])

    return df


def _compute_subgroup_metrics(
    audit_df: pd.DataFrame,
    group_col: str,
    group_label: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Compute model performance metrics for each subgroup.

    Metrics:
        auc: AUC-ROC (ranking performance)
        precision_top_decile: precision when flagging top 10% riskiest tracts
            — most relevant for operational decisions (we don't alert everyone,
            just the highest-risk tracts)
        recall: fraction of truly high-risk tracts correctly flagged
        avg_kriging_variance: mean PM2.5 prediction uncertainty
            — proxy for data quality in this subgroup
        n_tracts: subgroup size
        n_high_risk: number of truly high-risk tracts in subgroup

    Parameters
    ----------
    audit_df : pd.DataFrame
        Full audit DataFrame with predictions and demographics.
    group_col : str
        Column to stratify by (income_quartile or majority_race).
    group_label : str
        Label for the stratification dimension.
    config : dict
        Pipeline config.

    Returns
    -------
    pd.DataFrame
        One row per subgroup with all metrics.
    """
    records = []
    groups = audit_df[group_col].dropna().unique()

    for group_value in sorted(groups, key=str):
        mask = audit_df[group_col] == group_value
        subgroup = audit_df[mask]

        n_tracts = len(subgroup)
        n_high_risk = subgroup["y_true"].sum()

        if n_tracts < MIN_SUBGROUP_SIZE:
            logger.warning(
                f"Subgroup {group_label}={group_value} has only "
                f"{n_tracts} tracts — below minimum {MIN_SUBGROUP_SIZE}. "
                f"Metrics may be unreliable."
            )
            reliable = False
        else:
            reliable = True

        # Skip subgroups with no positive cases — AUC undefined
        if n_high_risk == 0 or n_high_risk == n_tracts:
            logger.warning(
                f"Subgroup {group_label}={group_value}: "
                f"no variation in y_true — skipping AUC"
            )
            auc = float("nan")
        else:
            try:
                auc = roc_auc_score(
                    subgroup["y_true"],
                    subgroup["y_pred_prob"],
                )
            except Exception as e:
                logger.warning(f"AUC failed for {group_label}={group_value}: {e}")
                auc = float("nan")

        # Precision at top decile
        # Flag the top 10% highest predicted risk tracts
        # What fraction of those are truly high risk?
        n_top_decile = max(1, int(n_tracts * 0.10))
        top_decile_mask = (
            subgroup["y_pred_prob"].rank(ascending=False) <= n_top_decile
        )
        precision_top_10 = subgroup.loc[
            top_decile_mask, "y_true"
        ].mean()

        # Recall — fraction of true positives caught
        if n_high_risk > 0:
            recall = recall_score(
                subgroup["y_true"],
                subgroup["y_pred_binary"],
                zero_division=0,
            )
        else:
            recall = float("nan")

        # Mean Kriging variance — data quality in this subgroup
        avg_kriging_var = subgroup["pm25_kriged_variance"].mean() \
            if "pm25_kriged_variance" in subgroup.columns else float("nan")

        records.append({
            "stratification_dim": group_label,
            "subgroup": str(group_value),
            "n_tracts": n_tracts,
            "n_high_risk": int(n_high_risk),
            "high_risk_rate": round(n_high_risk / n_tracts, 4),
            "auc": round(auc, 4) if not np.isnan(auc) else None,
            "precision_top_decile": round(precision_top_10, 4),
            "recall": round(recall, 4) if not np.isnan(recall) else None,
            "avg_kriging_variance": round(avg_kriging_var, 6)
                if not np.isnan(avg_kriging_var) else None,
            "metrics_reliable": reliable,
        })

        logger.info(
            f"  {group_label}={group_value}: "
            f"n={n_tracts}, AUC={auc:.3f}, "
            f"precision@10%={precision_top_10:.3f}, "
            f"recall={recall:.3f}"
        )

    return pd.DataFrame(records)


def _flag_disparities(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag subgroups where model performance is significantly below average.

    Disparity threshold: AUC more than 0.05 below the overall mean AUC
    for that stratification dimension.

    0.05 AUC points is a clinically meaningful gap — it means the model
    misranks ~5% more tracts in that subgroup than average.

    Also flags subgroups with high Kriging variance — these are likely
    data quality issues (sparse monitors) rather than model bias.

    Parameters
    ----------
    results_df : pd.DataFrame
        Combined equity audit results.

    Returns
    -------
    pd.DataFrame
        Same DataFrame with disparity_flag and disparity_likely_cause columns.
    """
    results_df["disparity_flag"] = False
    results_df["disparity_likely_cause"] = ""

    for dim in results_df["stratification_dim"].unique():
        dim_mask = results_df["stratification_dim"] == dim
        dim_df = results_df[dim_mask].copy()

        valid_auc = dim_df["auc"].dropna()
        if len(valid_auc) < 2:
            continue

        mean_auc = valid_auc.mean()
        disparity_threshold = 0.05

        for idx in dim_df.index:
            row_auc = results_df.loc[idx, "auc"]
            if pd.isna(row_auc):
                continue

            auc_gap = mean_auc - row_auc

            if auc_gap > disparity_threshold:
                results_df.loc[idx, "disparity_flag"] = True

                # Diagnose likely cause
                kriging_var = results_df.loc[idx, "avg_kriging_variance"]
                mean_kriging_var = dim_df["avg_kriging_variance"].mean()

                if (
                    not pd.isna(kriging_var)
                    and not pd.isna(mean_kriging_var)
                    and kriging_var > mean_kriging_var * 1.3
                ):
                    results_df.loc[idx, "disparity_likely_cause"] = (
                        "sparse_monitor_coverage"
                    )
                else:
                    results_df.loc[idx, "disparity_likely_cause"] = (
                        "investigate_further"
                    )

    flagged = results_df["disparity_flag"].sum()
    if flagged > 0:
        logger.warning(
            f"Equity audit: {flagged} subgroups flagged with performance disparities. "
            f"See equity_audit_results table for details."
        )
    else:
        logger.info("Equity audit: no significant disparities detected.")

    return results_df


def _log_to_mlflow(
    results_df: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    """Log equity audit summary metrics to MLflow."""
    with mlflow.start_run(run_name="equity_audit"):
        mlflow.set_tag("city_fips", config["full_fips"])

        # Log AUC range across income quartiles
        income_df = results_df[
            results_df["stratification_dim"] == "income_quartile"
        ]
        if not income_df.empty and income_df["auc"].notna().any():
            auc_range = income_df["auc"].max() - income_df["auc"].min()
            mlflow.log_metric("income_auc_range", round(float(auc_range), 4))
            mlflow.log_metric("n_disparities_income", int(
                income_df["disparity_flag"].sum()
            ))

        # Log AUC range across race groups
        race_df = results_df[
            results_df["stratification_dim"] == "majority_race"
        ]
        if not race_df.empty and race_df["auc"].notna().any():
            auc_range = race_df["auc"].max() - race_df["auc"].min()
            mlflow.log_metric("race_auc_range", round(float(auc_range), 4))
            mlflow.log_metric("n_disparities_race", int(
                race_df["disparity_flag"].sum()
            ))

        mlflow.log_metric(
            "total_disparities_flagged",
            int(results_df["disparity_flag"].sum()),
        )


def _log_summary(results_df: pd.DataFrame) -> None:
    """Log a human-readable equity audit summary to the pipeline log."""
    logger.info("=" * 60)
    logger.info("EQUITY AUDIT SUMMARY")
    logger.info("=" * 60)

    for dim in results_df["stratification_dim"].unique():
        dim_df = results_df[results_df["stratification_dim"] == dim]
        logger.info(f"\n{dim.upper()}:")

        for _, row in dim_df.iterrows():
            flag = " ← DISPARITY FLAGGED" if row["disparity_flag"] else ""
            cause = f" ({row['disparity_likely_cause']})" \
                if row["disparity_likely_cause"] else ""
            logger.info(
                f"  {row['subgroup']:20s} | "
                f"AUC={row['auc'] or 'N/A':.3f} | "
                f"n={row['n_tracts']:4d} | "
                f"precision@10%={row['precision_top_decile']:.3f}"
                f"{flag}{cause}"
            )

    logger.info("=" * 60)