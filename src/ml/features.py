"""
features.py
===========
Build the tract-week panel the CHBIndex models train on.

Target (Phase 6, decided 2026-11-11)
------------------------------------
enters_critical_4w = 1 if a tract that is NOT CRITICAL in week t is CRITICAL
in any of weeks t+1 .. t+4, else 0. Only non-CRITICAL tracts are eligible.

Why not "CRITICAL next week": next week's class is almost a copy of this
week's. PM2.5 enters the index as a 4-week mean and heat as a 32-day
composite (next week's window shares 25 of 32 days with this week's), so
this week's burden percentile alone scores ROC-AUC 0.972 on it. A 4-week
horizon reaches a heat window that shares no days with week t, and the
question it answers is the useful one: which tracts not on the list yet
will be on it within a month. Index-alone baseline on this target:
ROC-AUC 0.894, PR-AUC 0.409 at a 7.2% base rate.

Every feature is known at the end of week t
--------------------------------------------
- Index history: percentile now, lags 1/2/4, deltas, 4-week mean/std,
  weeks CRITICAL among the previous 4.
- Components now: pm25_z, lst_z, access_z, vulnerability_z, plus the latest
  single-week Kriged PM2.5 and its gap to the 4-week mean the index uses
  (the single week moves first).
- Static tract data: ACS groups, CDC PLACES, population, income, nearest ER.
- Spatial lags (Model B): the 8 nearest tract centroids' mean percentile,
  CRITICAL share and 1-week change in week t. k-nearest instead of H3 rings
  because ring membership swings from many tracts downtown to none in the
  desert; 8 neighbors gives every tract the same-sized neighborhood.

Left out on purpose
-------------------
- week_of_year / is_summer: one year of data makes them a week ID, which
  leaks how many tracts enter CRITICAL later in that week under CV.
- lst_yoy_anomaly: needs a prior year we don't have.
- No imputation: XGBoost and LightGBM route missing values natively, so no
  medians are computed over rows that end up in a test fold.

Usage
-----
    from src.ml.features import engineer_features, MODEL_B_FEATURES, TARGET
    panel = engineer_features(config)          # eligible rows only
"""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from src.utils.bigquery import read_query
from src.utils.logger import get_logger

logger = get_logger(__name__)

TARGET = "enters_critical_4w"
HORIZON_WEEKS = 4
LAG_WEEKS = (1, 2, 4)
HISTORY_WEEKS = max(LAG_WEEKS)
NEIGHBOR_K = 8
EARTH_RADIUS_KM = 6371.0

TEMPORAL_FEATURES = [
    "burden_percentile",
    "raw_burden",
    "pm25_z",
    "lst_z",
    "access_z",
    "vulnerability_z",
    "pct_lag_1w",
    "pct_lag_2w",
    "pct_lag_4w",
    "pct_delta_1w",
    "pct_delta_4w",
    "pct_mean_4w",
    "pct_std_4w",
    "weeks_critical_prev4",
    "pm25_z_delta_1w",
    "lst_z_delta_1w",
    "pm25_1w",
    "pm25_1w_minus_4w",
    "lst_mean_c",
]

STATIC_FEATURES = [
    "population",
    "median_income",
    "age_65_plus_pct",
    "under_5_pct",
    "low_income_pct",
    "no_vehicle_pct",
    "limited_english_pct",
    "uninsured_pct",
    "disability_pct",
    "asthma_pct",
    "copd_pct",
    "heart_disease_pct",
    "nearest_er_minutes",
    "er_count_30min",
]

SPATIAL_LAG_FEATURES = [
    "nbr_pct_mean",
    "nbr_critical_share",
    "nbr_pct_delta_1w",
    "pct_minus_nbr",
]

MODEL_A_FEATURES = TEMPORAL_FEATURES + STATIC_FEATURES
MODEL_B_FEATURES = MODEL_A_FEATURES + SPATIAL_LAG_FEATURES

ID_COLUMNS = ["geoid", "week_start_date", "week_idx", "burden_class"]

# Names that would mean a column carries week t+1.. information.
FORBIDDEN_NAME_PARTS = ("next", "future", "lead", "target", "enters_", "label")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_inputs(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weekly tract signals and static tract attributes from the warehouse."""
    fips = config["full_fips"]
    weekly = read_query(
        f"""
        SELECT
            c.geoid, c.week_start_date, c.burden_percentile, c.burden_class,
            c.raw_burden, c.pm25_z, c.lst_z, c.access_z, c.vulnerability_z,
            p.pm25_kriged AS pm25_1w,
            l.lst_mean_celsius AS lst_mean_c
        FROM {{dataset}}.compound_scores c
        LEFT JOIN {{dataset}}.pm25_kriged p
            ON p.geoid = c.geoid AND p.week_start_date = c.week_start_date
            AND p.city_fips = c.city_fips
        LEFT JOIN {{dataset}}.lst_tract_stats l
            ON l.geoid = c.geoid AND l.week_start_date = c.week_start_date
            AND l.city_fips = c.city_fips
        WHERE c.city_fips = '{fips}'
        """,
        config,
    )
    static = read_query(
        f"""
        WITH access AS (
            SELECT geoid, nearest_er_minutes, er_count_30min
            FROM {{dataset}}.healthcare_access
            WHERE city_fips = '{fips}'
              AND week_start_date = (
                  SELECT MAX(week_start_date) FROM {{dataset}}.healthcare_access
                  WHERE city_fips = '{fips}')
        )
        SELECT
            t.geoid, t.centroid_lat, t.centroid_lon, t.population, t.median_income,
            a.age_65_plus_pct, a.under_5_pct, a.low_income_pct, a.no_vehicle_pct,
            a.limited_english_pct, a.uninsured_pct, a.disability_pct,
            h.asthma_pct, h.copd_pct, h.heart_disease_pct,
            x.nearest_er_minutes, x.er_count_30min
        FROM {{dataset}}.tract_geometries t
        LEFT JOIN {{dataset}}.acs_vulnerability a ON a.geoid = t.geoid
        LEFT JOIN {{dataset}}.cdc_places_health h ON h.geoid = t.geoid
        LEFT JOIN access x ON x.geoid = t.geoid
        WHERE t.city_fips = '{fips}'
        """,
        config,
    )
    if weekly.empty or static.empty:
        raise ValueError(
            f"No ML inputs for FIPS {fips}; run the spatial pipeline first."
        )
    return weekly, static


# ---------------------------------------------------------------------------
# Panel construction (pure functions, testable without a warehouse)
# ---------------------------------------------------------------------------


def complete_grid(weekly: pd.DataFrame) -> pd.DataFrame:
    """One row per tract x week, sorted, with a 0-based consecutive week_idx.

    Shifts below are by row within a tract, so a missing tract-week would
    silently turn a 1-week lag into a 2-week lag. Reindexing to the full grid
    (every tract x every Monday from the first to the last stored week) makes a
    missing tract-week, or a whole week the backfill skipped, show up as NaN.
    A NaN week is never eligible itself, and add_target leaves the label NaN
    for any row whose 4-week window touches it, so those rows drop out too.
    """
    df = weekly.copy()
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    stored = np.sort(df["week_start_date"].unique())
    gaps = np.diff(stored).astype("timedelta64[D]").astype(int)
    if len(gaps) and (gaps % 7 != 0).any():
        raise ValueError(
            f"Week starts are not 7 days apart (gaps in days: {sorted(set(gaps))})"
        )
    weeks = pd.date_range(stored[0], stored[-1], freq="7D")
    missing = weeks.difference(pd.DatetimeIndex(stored))
    if len(missing):
        logger.warning(
            f"{len(missing)} week(s) missing from the record, kept as NaN rows: "
            f"{[d.strftime('%Y-%m-%d') for d in missing]}"
        )
    grid = pd.MultiIndex.from_product(
        [np.sort(df["geoid"].unique()), weeks], names=["geoid", "week_start_date"]
    )
    df = df.set_index(["geoid", "week_start_date"]).reindex(grid).reset_index()
    week_idx = {w: i for i, w in enumerate(weeks)}
    df["week_idx"] = df["week_start_date"].map(week_idx).astype(int)
    df["is_critical"] = (df["burden_class"] == "CRITICAL").astype(float)
    df.loc[df["burden_class"].isna(), "is_critical"] = np.nan
    return df.sort_values(["geoid", "week_idx"]).reset_index(drop=True)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lags, deltas and rolling stats that use weeks <= t only."""
    df = df.copy()
    g = df.groupby("geoid", sort=False)
    pct = df["burden_percentile"]
    for k in LAG_WEEKS:
        df[f"pct_lag_{k}w"] = g["burden_percentile"].shift(k)
    df["pct_delta_1w"] = pct - df["pct_lag_1w"]
    df["pct_delta_4w"] = pct - df["pct_lag_4w"]
    # Rolling windows end at week t (current row included), never after it.
    df["pct_mean_4w"] = g["burden_percentile"].transform(
        lambda s: s.rolling(4, min_periods=4).mean()
    )
    df["pct_std_4w"] = g["burden_percentile"].transform(
        lambda s: s.rolling(4, min_periods=4).std()
    )
    prev = sum(g["is_critical"].shift(k) for k in range(1, 5))
    df["weeks_critical_prev4"] = prev
    df["pm25_z_delta_1w"] = df["pm25_z"] - g["pm25_z"].shift(1)
    df["lst_z_delta_1w"] = df["lst_z"] - g["lst_z"].shift(1)
    # Same window as the index (compound_score.pm25_window_weeks = 4, trailing).
    pm25_4w = g["pm25_1w"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    df["pm25_1w_minus_4w"] = df["pm25_1w"] - pm25_4w
    return df


def add_spatial_lags(
    df: pd.DataFrame, static: pd.DataFrame, k: int = NEIGHBOR_K
) -> pd.DataFrame:
    """Week-t values of each tract's k nearest neighbors (by centroid)."""
    df = df.copy()
    tracts = static.dropna(subset=["centroid_lat", "centroid_lon"]).sort_values("geoid")
    coords = np.radians(tracts[["centroid_lat", "centroid_lon"]].to_numpy())
    tree = BallTree(coords, metric="haversine")
    _, idx = tree.query(coords, k=k + 1)
    nbr = idx[:, 1:]  # drop self
    geoids = tracts["geoid"].to_numpy()
    weeks = np.sort(df["week_idx"].unique())

    def neighbor_mean(col: str) -> pd.DataFrame:
        wide = df.pivot(index="geoid", columns="week_idx", values=col).reindex(
            index=geoids, columns=weeks
        )
        vals = wide.to_numpy()[nbr]  # tracts x k x weeks
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN neighbors -> NaN
            means = np.nanmean(vals, axis=1)
        return pd.DataFrame(
            {
                "geoid": np.repeat(geoids, len(weeks)),
                "week_idx": np.tile(weeks, len(geoids)),
                col: means.ravel(),
            }
        )

    for src, dst in [
        ("burden_percentile", "nbr_pct_mean"),
        ("is_critical", "nbr_critical_share"),
        ("pct_delta_1w", "nbr_pct_delta_1w"),
    ]:
        m = neighbor_mean(src).rename(columns={src: dst})
        df = df.merge(m, on=["geoid", "week_idx"], how="left")
    df["pct_minus_nbr"] = df["burden_percentile"] - df["nbr_pct_mean"]
    return df


def add_target(df: pd.DataFrame, horizon: int = HORIZON_WEEKS) -> pd.DataFrame:
    """enters_critical_4w and the eligibility flag.

    Eligible rows: not CRITICAL in week t, full lag history (t >= 4 weeks into
    the record) and all `horizon` future weeks observed.
    """
    df = df.copy()
    g = df.groupby("geoid", sort=False)["is_critical"]
    future = pd.concat([g.shift(-k) for k in range(1, horizon + 1)], axis=1)
    df[TARGET] = future.max(axis=1, skipna=False)
    last = df["week_idx"].max()
    df["eligible"] = (
        (df["is_critical"] == 0)
        & (df["week_idx"] >= HISTORY_WEEKS)
        & (df["week_idx"] <= last - horizon)
        & df[TARGET].notna()
    )
    return df


def build_panel(weekly: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    """Full tract-week panel with features, target and eligibility (all rows)."""
    df = complete_grid(weekly)
    df = add_temporal_features(df)
    df = add_spatial_lags(df, static)
    df = df.merge(static, on="geoid", how="left")
    df = add_target(df)
    return df


def check_no_leakage(feature_cols: list[str]) -> None:
    """Fail if a feature name suggests future information or the label."""
    bad = [c for c in feature_cols if any(p in c.lower() for p in FORBIDDEN_NAME_PARTS)]
    if bad:
        raise ValueError(f"Feature columns look like future/label columns: {bad}")


def engineer_features(config: dict[str, Any]) -> pd.DataFrame:
    """Eligible training rows with ids, Model B features (a superset of A) and target."""
    check_no_leakage(MODEL_B_FEATURES)
    weekly, static = load_inputs(config)
    panel = build_panel(weekly, static)
    out = panel.loc[
        panel["eligible"],
        ID_COLUMNS + ["centroid_lat", "centroid_lon"] + MODEL_B_FEATURES + [TARGET],
    ].reset_index(drop=True)
    out[TARGET] = out[TARGET].astype(int)
    logger.info(
        f"ML panel: {len(out):,} eligible rows, {out['week_idx'].nunique()} weeks "
        f"({out['week_start_date'].min():%Y-%m-%d} to {out['week_start_date'].max():%Y-%m-%d}), "
        f"base rate {out[TARGET].mean():.3f}"
    )
    return out
