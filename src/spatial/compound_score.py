"""
compound_score.py
=================
Compound Climate-Health Burden Score for each census tract.

The index combines four signals. Each is signed so that a higher
contribution means more burden:
  - PM2.5 (Kriged, mean of the last N weeks)     higher = more burden
  - LST (land surface temperature z-score)        higher = more burden
  - Healthcare access (beds per 1,000 residents)  higher = LESS burden
  - Population vulnerability (Census ACS)         higher = more burden

The first three are hazards; vulnerability is who lives there. It is one
score built from seven ACS shares (age 65+, under 5, income below 200% of
poverty, no vehicle, limited English, uninsured, disability):
  a. Shrink each share toward the county mean by its sampling noise
     (empirical Bayes). B = real variance / (real variance + the tract's
     sampling variance), shrunk = mean + B * (share - mean). A precise
     share keeps its value; a noisy one moves toward the mean. Unlike a
     CV cutoff, small but precise shares are kept.
  b. z-score each shrunk share; a missing share counts as the mean (0).
  c. Average the seven z-scores. The weekly composite z-scores this
     average again, so vulnerability has the same weight as each hazard.
ACS changes once a year, so vulnerability is the same every week.

Method: signed z-score composite
  1. Load the three signals for the week and city. PM2.5 is the mean
     of the Kriged weekly values over the last pm25_window_weeks weeks
     (city config, default 4). Weekly PM2.5 from ~21 monitors is noisy:
     with weekly values only 59% of CRITICAL tracts stayed CRITICAL the
     next week; with a 4-week mean, 81%. The pm25_kriged table keeps the
     weekly values for drift checks.
     LST comes from a 32-day cloud-free composite. In cloudy weeks it can
     cover only a few tracts (2025-11-24: 223 of 2,493). If fewer than
     gee.min_coverage of city tracts have LST, the week reuses the LST
     z-scores of the latest week within LST_FALLBACK_MAX_WEEKS that
     passes. Otherwise the median fill below would give ~90% of tracts
     an average heat score and rank heat on the few tracts that remain.
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

from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.utils.bigquery import REPLACE_PARTITION, read_query, write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

# How far back to look for a week with enough LST coverage
LST_FALLBACK_MAX_WEEKS = 8

# Column in the input table -> sign of its contribution to burden.
# +1: higher value means more burden. -1: higher value means less burden.
SIGNED_SIGNALS = {
    "pm25_kriged": +1,
    "lst_zscore": +1,
    "access_score_er": -1,
    "vulnerability": +1,
}

# Output column name for each signed contribution
COMPONENT_COLUMNS = {
    "pm25_kriged": "pm25_z",
    "lst_zscore": "lst_z",
    "access_score_er": "access_z",
    "vulnerability": "vulnerability_z",
}

# acs_vulnerability columns (<name>_pct, <name>_moe) averaged into vulnerability
VULNERABILITY_INDICATORS = [
    "age_65_plus",
    "under_5",
    "low_income",
    "no_vehicle",
    "limited_english",
    "uninsured",
    "disability",
]
Z90 = 1.645  # ACS MOEs are 90% intervals

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
        f"Computing compound burden score for {config['city_name']} week of {week_str}"
    )

    # Step 1 — Load the three signals
    window = int(config.get("pm25_window_weeks", 1))
    signals_df = _load_signals(config, week_str, pm25_window_weeks=window)
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
        scores_df,
        "compound_scores",
        config,
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


def _window_start(week_str: str, weeks: int) -> str:
    """First week of a window of `weeks` weeks ending at week_str."""
    end = date.fromisoformat(week_str)
    return (end - timedelta(weeks=weeks - 1)).isoformat()


def _load_signals(
    config: dict[str, Any],
    week_str: str,
    pm25_window_weeks: int = 1,
) -> pd.DataFrame:
    """
    Load PM2.5, LST z-score and healthcare access for one week.

    PM2.5 is the per-tract mean of pm25_kriged over the weeks from
    _window_start(week_str, pm25_window_weeks) to week_str. Weeks missing
    from the table (e.g. the first weeks of the record) are skipped, and
    the mean uses the weeks that exist.

    Returns
    -------
    pd.DataFrame
        One row per city tract with columns:
        geoid, pm25_kriged, lst_zscore, access_score_er
    """
    full_fips = config["full_fips"]
    start_str = _window_start(week_str, pm25_window_weeks)

    query = f"""
        WITH pm AS (
            SELECT
                geoid,
                AVG(pm25_kriged) AS pm25_kriged,
                COUNT(*) AS pm25_weeks
            FROM
                {{dataset}}.pm25_kriged
            WHERE
                week_start_date BETWEEN '{start_str}' AND '{week_str}'
                AND city_fips = '{full_fips}'
            GROUP BY geoid
        )
        SELECT
            k.geoid,
            pm.pm25_kriged,
            pm.pm25_weeks,
            l.lst_zscore,
            a.access_score_er
        FROM
            {{dataset}}.pm25_kriged k
        JOIN
            pm ON k.geoid = pm.geoid
        LEFT JOIN
            {{dataset}}.lst_tract_stats l
            ON k.geoid = l.geoid
            AND l.week_start_date = '{week_str}'
        LEFT JOIN
            {{dataset}}.healthcare_access a
            ON k.geoid = a.geoid
            AND a.week_start_date = '{week_str}'
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

    weeks_used = int(df["pm25_weeks"].min())
    if weeks_used < pm25_window_weeks:
        _check_window_complete(config, start_str, week_str)
        logger.warning(
            f"PM2.5 window: {weeks_used} of {pm25_window_weeks} weeks available "
            f"from {start_str} to {week_str}"
        )
    df = df.drop(columns="pm25_weeks")
    df = _fill_low_coverage_lst(df, config, week_str)
    return df.merge(_load_vulnerability(config), on="geoid", how="left")


def _check_window_complete(
    config: dict[str, Any], start_str: str, week_str: str
) -> None:
    """
    Refuse to score while a window week has monitor data but no Kriging yet.

    A week can be missing from pm25_kriged for two reasons: no monitor data
    (start of the record; the mean then uses the weeks that exist) or its
    Kriging has not run yet. The second silently shortened the window for
    30 backfill weeks (2024-07-01 to 2025-06-09) when spatial runs were
    triggered out of week order. Score weeks oldest first.
    """
    fips = config["full_fips"]
    late = read_query(
        f"""
        SELECT DISTINCT CAST(r.week_start_date AS VARCHAR) AS week
        FROM {{dataset}}.epa_pm25_raw r
        WHERE r.city_fips = '{fips}'
          AND r.week_start_date BETWEEN '{start_str}' AND '{week_str}'
          AND r.week_start_date NOT IN (
              SELECT DISTINCT week_start_date FROM {{dataset}}.pm25_kriged
              WHERE city_fips = '{fips}'
                AND week_start_date BETWEEN '{start_str}' AND '{week_str}'
          )
        ORDER BY week
        """,
        config,
    )
    if not late.empty:
        weeks = ", ".join(late["week"].str[:10])
        raise ValueError(
            f"PM2.5 window for {week_str} is missing Kriging for {weeks}, which "
            f"have monitor data. Score weeks oldest first (scripts/rescore_weeks.py)."
        )


def _load_vulnerability(config: dict[str, Any]) -> pd.DataFrame:
    """geoid, vulnerability for the city (yearly ACS, same every week)."""
    acs = read_query(
        f"""
        SELECT *
        FROM {{dataset}}.acs_vulnerability
        WHERE city_fips = '{config["full_fips"]}'
        """,
        config,
    )
    if acs.empty:
        raise ValueError(
            "acs_vulnerability is empty. Run: "
            "PYTHONPATH=. python -m src.ingestion.acs_vulnerability"
        )
    return vulnerability_score(acs)


def shrink_share(pct: pd.Series, moe: pd.Series) -> pd.Series:
    """
    Empirical Bayes shrinkage of ACS shares toward their mean.

    Observed share = true share + sampling error (SE = MOE / 1.645).
    Real (between-tract) variance = observed variance - mean sampling
    variance. Each share moves toward the mean by its own noise:
        B = real / (real + SE^2), shrunk = mean + B * (share - mean)
    Missing shares stay missing.
    """
    se2 = (moe / Z90) ** 2
    ok = pct.notna() & se2.notna()
    if ok.sum() < 2:
        return pct
    mean = pct[ok].mean()
    real = max(float(pct[ok].var(ddof=0) - se2[ok].mean()), 0.0)
    total = real + se2
    b = (real / total).where(total > 0, 1.0)
    return mean + b * (pct - mean)


def vulnerability_score(acs: pd.DataFrame) -> pd.DataFrame:
    """
    One vulnerability value per tract: mean z-score of the shrunk shares.

    Tracts with every share missing (no residents) get NaN; the weekly
    median fill then treats them as neutral.
    """
    zs = []
    for name in VULNERABILITY_INDICATORS:
        shrunk = shrink_share(acs[f"{name}_pct"], acs[f"{name}_moe"])
        std = shrunk.std(ddof=0)
        z = (shrunk - shrunk.mean()) / std if std > 0 else shrunk * 0
        zs.append(z)
    z = pd.concat(zs, axis=1)
    score = z.fillna(0.0).mean(axis=1).where(z.notna().any(axis=1))
    return pd.DataFrame({"geoid": acs["geoid"], "vulnerability": score})


def _latest_good_week(
    counts: pd.DataFrame,
    n_tracts: int,
    min_coverage: float,
) -> str | None:
    """
    Latest week whose LST covers at least min_coverage of city tracts.

    counts: columns week (ISO date string) and n (tracts with LST).
    Returns None when no week passes.
    """
    good = counts[counts["n"] / n_tracts >= min_coverage]
    if good.empty:
        return None
    return str(good["week"].max())


def _fill_low_coverage_lst(
    df: pd.DataFrame,
    config: dict[str, Any],
    week_str: str,
) -> pd.DataFrame:
    """
    Reuse an earlier week's LST z-scores when this week's composite is
    mostly cloud gaps.

    Carries the z-score, not the temperature: the z-score is a tract's
    position within its own week, so it does not bring the earlier week's
    seasonal level along.
    """
    min_coverage = float(config.get("gee", {}).get("min_coverage", 0.8))
    coverage = float(df["lst_zscore"].notna().mean())
    if coverage >= min_coverage:
        return df

    full_fips = config["full_fips"]
    start = (
        date.fromisoformat(week_str) - timedelta(weeks=LST_FALLBACK_MAX_WEEKS)
    ).isoformat()
    counts = read_query(
        f"""
        SELECT
            CAST(week_start_date AS STRING) AS week,
            COUNT(lst_zscore) AS n
        FROM {{dataset}}.lst_tract_stats
        WHERE
            city_fips = '{full_fips}'
            AND week_start_date >= '{start}'
            AND week_start_date < '{week_str}'
        GROUP BY week_start_date
        """,
        config,
    )
    source = _latest_good_week(counts, len(df), min_coverage)
    if source is None:
        logger.warning(
            f"LST covers {coverage:.0%} of tracts for {week_str} and no week in "
            f"the previous {LST_FALLBACK_MAX_WEEKS} passes {min_coverage:.0%}; "
            f"missing tracts get the city median"
        )
        return df

    prev = read_query(
        f"""
        SELECT geoid, lst_zscore
        FROM {{dataset}}.lst_tract_stats
        WHERE city_fips = '{full_fips}' AND week_start_date = '{source}'
        """,
        config,
    )
    logger.warning(
        f"LST covers {coverage:.0%} of tracts for {week_str} "
        f"(min {min_coverage:.0%}); using LST z-scores from {source}"
    )
    df = df.copy()
    df["lst_zscore"] = df["geoid"].map(prev.set_index("geoid")["lst_zscore"])
    return df


def _impute_missing(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Fill missing signal values with the city median.

    Missing LST: water-only tracts with no valid land pixels.
    Missing access: unreachable tracts or tracts with no residents.
    Missing vulnerability: tracts with no residents in the ACS.
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


def _build_composite(
    df: pd.DataFrame,
    config: dict[str, Any],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Build the signed z-score composite and classify each tract.

    weights: optional multiplier per signal column (default 1 each). Used
    only by the weight sensitivity check (src/evals/index_validation.py);
    the weekly pipeline uses equal weights.

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
        contribution = sign * z * (weights or {}).get(col, 1.0)
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

    # Direction check: better access must lower burden. This is the check
    # that would have caught the PCA sign problem (access rho +0.86). It tests
    # the access component itself, not the total score: in LA the best-served
    # tracts are also the hottest and most polluted, so in some weeks total
    # burden is uncorrelated with access even though access enters negatively
    # (week of 2025-10-27: total rho +0.007).
    rho = df["access_z"].corr(signals_df["access_score_er"], method="spearman")
    if not rho <= -0.5:
        raise ValueError(
            f"Access component does not fall as access rises (Spearman rho = "
            f"{rho:.3f}). Check the access sign in SIGNED_SIGNALS."
        )
    total_rho = df["compound_score"].corr(
        signals_df["access_score_er"], method="spearman"
    )

    logger.info(
        f"Compound score validation passed: {len(df):,} tracts, "
        f"mean={df['compound_score'].mean():.3f}, "
        f"access component rho={rho:.3f}, total burden vs access rho={total_rho:.3f}"
    )
