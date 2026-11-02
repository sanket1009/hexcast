"""
compound_score.py
=================
Compound Climate-Health Burden Score for each census tract.

The index combines three signals. Each is signed so that a higher
contribution means more burden:
  - PM2.5 (Kriged annual concentration)          higher = more burden
  - LST (land surface temperature z-score)        higher = more burden
  - Healthcare access (beds per 1,000 residents)  higher = LESS burden

Method: signed z-score composite
  1. Load the three signals for the week and city.
  2. Impute missing values with the city median. A median is neutral:
     it adds nothing to the burden. Affects water-only tracts (LST) and
     unscored tracts (access: unreachable or no residents).
  3. Standardize each signal across city tracts: z = (x - mean) / std.
  4. Sign and sum: raw_burden = z_pm25 + z_lst - z_access.
  5. Min-max scale raw_burden to 0-1 for display (compound_score).
  6. Rank tracts by city percentile and classify with burden_percentiles
     from the city config.

Why not Spatial PCA:
  PCA finds the direction of greatest shared variance. It cannot be told
  that access reduces burden. In LA, access rises with PM2.5 and LST, so
  PC1 scored high-access tracts as more burdened. The sign here is set
  by the formula, not by the data.

Spatial lags were dropped along with PCA. They can return as a
sensitivity check.

Input (per city, per week):
    pm25_kriged, lst_tract_stats, healthcare_access

Output:
    compound_scores: one row per tract per week
    Columns: geoid, compound_score, raw_burden, burden_percentile,
             burden_class, pm25_z, lst_z, access_z,
             week_start_date, city_fips

    pm25_z, lst_z, access_z are signed contributions. access_z is
    negative when access is above the city average.

Usage:
    from src.spatial.compound_score import compute_compound_score
    df = compute_compound_score(config)
"""

from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.utils.bigquery import REPLACE_PARTITION, read_query, write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Column in the input table -> sign of its contribution to burden.
# +1: higher value means more burden. -1: higher value means less burden.
SIGNED_SIGNALS = {
    "pm25_kriged": +1,
    "lst_zscore": +1,
    "access_score_er": -1,
}

# Output column name for each signed contribution
COMPONENT_COLUMNS = {
    "pm25_kriged": "pm25_z",
    "lst_zscore": "lst_z",
    "access_score_er": "access_z",
}

VALID_CLASSES = {"LOW", "ELEVATED", "HIGH", "CRITICAL"}


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def compute_compound_score(
    config: dict[str, Any],
    week_start: datetime | None = None,
) -> pd.DataFrame:
    """
    Compute the weekly compound burden score for each tract in the city.

    Parameters
    ----------
    config : dict
        Merged pipeline config (pipeline.yaml + city yaml).
    week_start : datetime, optional
        Week to score. Defaults to the most recent Monday.

    Returns
    -------
    pd.DataFrame
        One row per tract with the composite score, percentile, class,
        and signed component contributions.
    """
    if week_start is None:
        today = datetime.now()
        week_start = today - timedelta(days=today.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)

    week_str = week_start.strftime("%Y-%m-%d")
    logger.info(
        f"Computing compound burden score for {config['city_name']} "
        f"week of {week_str}"
    )

    # Step 1 — Load the three signals
    signals_df = _load_signals(config, week_str)
    logger.info(f"Loaded signals for {len(signals_df):,} tracts")

    # Step 2 — Impute missing values with the city median
    signals_df, imputed_count = _impute_missing(signals_df)
    if imputed_count > 0:
        logger.warning(f"Imputed {imputed_count} missing values with city median")

    # Steps 3-6 — Standardize, sign, sum, scale, classify
    scores_df = _build_composite(signals_df, config)

    # Step 7 — Metadata
    scores_df["week_start_date"] = week_start.date()
    scores_df["city_fips"] = config["full_fips"]

    # Step 8 — Validate and write
    _validate_output(scores_df, signals_df)
    # Replace this week's rows for this city; re-runs do not duplicate
    write_dataframe(
        scores_df, "compound_scores", config,
        write_disposition=REPLACE_PARTITION,
    )

    high_risk = scores_df["burden_class"].isin(["HIGH", "CRITICAL"]).sum()
    critical = (scores_df["burden_class"] == "CRITICAL").sum()
    logger.info(
        f"Compound scores complete: {len(scores_df):,} tracts, "
        f"{high_risk} high/critical, {critical} critical. "
        f"Mean score: {scores_df['compound_score'].mean():.3f}"
    )

    return scores_df


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------

def _load_signals(config: dict[str, Any], week_str: str) -> pd.DataFrame:
    """
    Load PM2.5, LST z-score and healthcare access for one week.

    Returns
    -------
    pd.DataFrame
        One row per city tract with columns:
        geoid, pm25_kriged, lst_zscore, access_score_er
    """
    full_fips = config["full_fips"]

    query = f"""
        SELECT
            k.geoid,
            k.pm25_kriged,
            l.lst_zscore,
            a.access_score_er
        FROM
            {{dataset}}.pm25_kriged k
        LEFT JOIN
            {{dataset}}.lst_tract_stats l
            ON k.geoid = l.geoid
            AND l.week_start_date = '{week_str}'
        LEFT JOIN
            {{dataset}}.healthcare_access a
            ON k.geoid = a.geoid
        WHERE
            k.week_start_date = '{week_str}'
            AND k.city_fips = '{full_fips}'
        ORDER BY k.geoid
    """

    df = read_query(query, config)

    if df.empty:
        raise ValueError(
            f"No signals found for week {week_str}, FIPS {full_fips}. "
            f"Run ingestion and spatial transform DAGs first."
        )

    return df


def _impute_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Fill missing signal values with the city median.

    Missing LST: water-only tracts with no valid land pixels.
    Missing access: unreachable tracts or tracts with no residents.
    Dropping them would leave holes in the city map, so they stay in
    with a neutral (median) value.

    Returns
    -------
    tuple of (DataFrame with signals filled, number of values filled)
    """
    df = df.copy()
    imputed_count = 0

    for col in SIGNED_SIGNALS:
        null_mask = df[col].isna()
        n_null = int(null_mask.sum())
        if n_null > 0:
            df.loc[null_mask, col] = df[col].median()
            imputed_count += n_null

    return df, imputed_count


def _build_composite(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """
    Build the signed z-score composite and classify each tract.

    For each signal x:
        z = (x - mean) / std          across city tracts this week
        contribution = sign * z       sign from SIGNED_SIGNALS

    raw_burden = sum of contributions
    compound_score = min-max scaled raw_burden (0-1, display only)
    burden_percentile = city percentile rank of raw_burden (0-100)

    Classes come from percentiles, not from the 0-1 score. Min-max
    scores depend on the single most extreme tract, so one outlier
    would shift every class cutoff.

    Returns
    -------
    pd.DataFrame
        geoid, compound_score, raw_burden, burden_percentile,
        burden_class, and one signed component column per signal.
    """
    out = pd.DataFrame({"geoid": df["geoid"].values})
    raw = np.zeros(len(df))

    for col, sign in SIGNED_SIGNALS.items():
        x = df[col].astype(float).values
        std = x.std(ddof=0)
        z = (x - x.mean()) / std if std > 0 else np.zeros(len(x))
        contribution = sign * z
        out[COMPONENT_COLUMNS[col]] = np.round(contribution, 4)
        raw += contribution

    out["raw_burden"] = np.round(raw, 4)

    lo, hi = raw.min(), raw.max()
    if hi > lo:
        out["compound_score"] = np.round((raw - lo) / (hi - lo), 4)
    else:
        logger.warning("All raw burden values identical; compound_score set to 0.5")
        out["compound_score"] = 0.5

    pct = pd.Series(raw).rank(pct=True).mul(100).round(1).values
    out["burden_percentile"] = pct

    pcts = config["burden_percentiles"]
    conditions = [
        pct >= pcts["critical"],
        pct >= pcts["high_risk"],
        pct >= pcts["elevated"],
    ]
    out["burden_class"] = np.select(
        conditions, ["CRITICAL", "HIGH", "ELEVATED"], default="LOW"
    )

    for label in ["CRITICAL", "HIGH", "ELEVATED", "LOW"]:
        logger.info(f"  {label}: {(out['burden_class'] == label).sum():,} tracts")

    return out


def _validate_output(df: pd.DataFrame, signals_df: pd.DataFrame) -> None:
    """Validate compound scores before the write."""
    if df.empty:
        raise ValueError("Compound scores output is empty")

    if df["geoid"].duplicated().any():
        raise ValueError("Duplicate geoid in compound scores")

    if len(df) < 100:
        raise ValueError(f"Only {len(df)} tracts, expected at least 100")

    out_of_range = ((df["compound_score"] < 0) | (df["compound_score"] > 1)).sum()
    if out_of_range > 0:
        raise ValueError(f"{out_of_range} tracts have compound_score outside [0, 1]")

    invalid = set(df["burden_class"].unique()) - VALID_CLASSES
    if invalid:
        raise ValueError(f"Invalid burden classes: {invalid}")

    # Direction check: a higher score must mean less access. This is the
    # check that would have caught the PCA sign problem.
    rho = df["compound_score"].corr(signals_df["access_score_er"], method="spearman")
    if rho >= 0:
        raise ValueError(
            f"Burden rises with access (Spearman rho = {rho:.3f}). "
            f"Check the access sign in SIGNED_SIGNALS."
        )

    logger.info(
        f"Compound score validation passed: {len(df):,} tracts, "
        f"mean={df['compound_score'].mean():.3f}, "
        f"access rho={rho:.3f}"
    )
