"""
drift.py
========
Weekly data-drift checks, run by spatial_transform after the compound score.

For week W against the trailing TRAILING_WEEKS weeks:

    pm25_level, lst_level   county mean this week, in standard deviations of
                            the trailing weekly means (a citywide jump: smoke,
                            a heat wave, a season turning)
    pm25_shape, lst_shape   PSI of the within-week standardized tract values:
                            has the spatial pattern changed (a local plume, a
                            cloud mask over one area)
    lst_coverage            share of tracts with a valid LST value
    monitors_ratio          PM2.5 monitor count / trailing median
    kriging_mae_ratio       Kriging LOO MAE / trailing median
    critical_jaccard        overlap of this week's CRITICAL set with last week's

PSI (population stability index) bins the trailing values into deciles and
compares the share of this week's tracts in each bin:
    PSI = sum((actual - expected) * ln(actual / expected))

Why level and shape are split: PSI on raw PM2.5 or LST flagged every one of
the 97 checkable weeks as a major shift, because the county-wide level moves
with the weather every week. That is alert fatigue, not monitoring. Splitting
"did the whole county move" from "did the pattern change" keeps both signals
and lets each have a threshold that fires rarely. The thresholds were set from
the 2024-2026 history (2026-11-11) so each flags roughly the most unusual
5-10% of weeks; on that history the major flags land on real events
(2024-12-02 and 2025-10-20 smoke, cloud-masked LST weeks 2025-11-24 and
2025-12-01, July 4 and New Year Kriging errors).

The within-week z components of the index are not checked: they are
N(0, 1) every week by construction.

Checks warn; they never fail the DAG. Results go to the drift_checks table.
"""

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.utils.bigquery import REPLACE_PARTITION, read_query, write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

TRAILING_WEEKS = 8
PSI_BINS = 10
EPS = 1e-4

# metric -> (moderate threshold, major threshold, direction)
# "abs": |value| larger is worse; "high": larger is worse; "low": smaller is worse;
# "ratio": far from 1 either way; "ratio_up": only increases above 1 matter
THRESHOLDS = {
    "pm25_level": (2.0, 3.0, "abs"),
    "lst_level": (2.0, 3.0, "abs"),
    "pm25_shape": (0.5, 1.0, "high"),
    "lst_shape": (0.10, 0.25, "high"),
    "lst_coverage": (0.95, 0.80, "low"),
    "monitors_ratio": (0.10, 0.25, "ratio"),
    "kriging_mae_ratio": (0.5, 1.0, "ratio_up"),
    "critical_jaccard": (0.45, 0.30, "low"),
}


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = PSI_BINS) -> float:
    """Population stability index of `actual` against `expected` (decile bins)."""
    expected = np.asarray(expected, float)
    actual = np.asarray(actual, float)
    expected, actual = expected[np.isfinite(expected)], actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, EPS, None), np.clip(a, EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def status(metric: str, value: float) -> str:
    if value != value:  # NaN
        return "missing"
    moderate, major, direction = THRESHOLDS[metric]
    if direction == "abs":
        bad = abs(value)
    elif direction == "high":
        bad = value
    elif direction == "low":
        return "major" if value < major else "moderate" if value < moderate else "ok"
    elif direction == "ratio":
        bad = abs(value - 1)
    else:  # ratio_up: only increases matter
        bad = value - 1
    return "major" if bad > major else "moderate" if bad > moderate else "ok"


def _within_week_z(df: pd.DataFrame, col: str) -> pd.Series:
    g = df.groupby("week_start_date")[col]
    return (df[col] - g.transform("mean")) / g.transform("std")


def _level_shift(current: pd.Series, trailing: pd.DataFrame, col: str) -> float:
    weekly = trailing.groupby("week_start_date")[col].mean()
    sd = weekly.std()
    return (
        float((current.mean() - weekly.mean()) / sd)
        if sd and sd == sd
        else float("nan")
    )


def drift_metrics(
    current: pd.DataFrame,
    trailing: pd.DataFrame,
    critical_now: set,
    critical_prev: set,
) -> dict[str, float]:
    """current/trailing: tract rows with pm25, lst, n_monitors, cv_mae, week_start_date."""
    both = pd.concat([trailing, current], ignore_index=True)
    for col in ("pm25", "lst"):
        both[f"{col}_z"] = _within_week_z(both, col)
    is_cur = both["week_start_date"] == current["week_start_date"].iloc[0]
    trail_weekly = trailing.groupby("week_start_date").agg(
        n_monitors=("n_monitors", "first"), cv_mae=("cv_mae", "first")
    )
    union = critical_now | critical_prev
    return {
        "pm25_level": _level_shift(current["pm25"], trailing, "pm25"),
        "lst_level": _level_shift(current["lst"], trailing, "lst"),
        "pm25_shape": psi(both.loc[~is_cur, "pm25_z"], both.loc[is_cur, "pm25_z"]),
        "lst_shape": psi(both.loc[~is_cur, "lst_z"], both.loc[is_cur, "lst_z"]),
        "lst_coverage": float(current["lst"].notna().mean()),
        "monitors_ratio": float(
            current["n_monitors"].iloc[0] / trail_weekly["n_monitors"].median()
        ),
        "kriging_mae_ratio": float(
            current["cv_mae"].iloc[0] / trail_weekly["cv_mae"].median()
        ),
        "critical_jaccard": (
            len(critical_now & critical_prev) / len(union) if union else float("nan")
        ),
    }


def drift_report(config: dict[str, Any], week: Any) -> pd.DataFrame:
    week = pd.Timestamp(week)
    first = week - timedelta(weeks=TRAILING_WEEKS)
    fips = config["full_fips"]
    df = read_query(
        f"""
        SELECT p.geoid, p.week_start_date, p.pm25_kriged AS pm25, p.n_monitors, p.cv_mae,
               l.lst_mean_celsius AS lst, c.burden_class
        FROM {{dataset}}.pm25_kriged p
        LEFT JOIN {{dataset}}.lst_tract_stats l
            ON l.geoid = p.geoid AND l.week_start_date = p.week_start_date
            AND l.city_fips = p.city_fips
        LEFT JOIN {{dataset}}.compound_scores c
            ON c.geoid = p.geoid AND c.week_start_date = p.week_start_date
            AND c.city_fips = p.city_fips
        WHERE p.city_fips = '{fips}'
          AND p.week_start_date BETWEEN '{first:%Y-%m-%d}' AND '{week:%Y-%m-%d}'
        """,
        config,
    )
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    current = df[df["week_start_date"] == week]
    trailing = df[df["week_start_date"] < week]
    if current.empty or trailing.empty:
        raise ValueError(f"Need week {week:%Y-%m-%d} and at least one earlier week")
    prev_week = trailing["week_start_date"].max()

    def crit(w: pd.Timestamp) -> set:
        rows = df[(df["week_start_date"] == w) & (df["burden_class"] == "CRITICAL")]
        return set(rows["geoid"])

    metrics = drift_metrics(current, trailing, crit(week), crit(prev_week))
    out = pd.DataFrame(
        [
            {
                "metric": m,
                "value": round(v, 4) if v == v else None,
                "status": status(m, v),
                "moderate_threshold": THRESHOLDS[m][0],
                "major_threshold": THRESHOLDS[m][1],
            }
            for m, v in metrics.items()
        ]
    )
    out["trailing_weeks"] = int(trailing["week_start_date"].nunique())
    out["week_start_date"] = week.date()
    out["city_fips"] = fips
    return out


def check_drift(config: dict[str, Any], week: Any) -> dict[str, str]:
    """Compute, store and log the week's drift checks. Never raises on drift itself."""
    report = drift_report(config, week)
    write_dataframe(report, "drift_checks", config, write_disposition=REPLACE_PARTITION)
    for row in report.itertuples():
        msg = f"Drift {row.metric} = {row.value} ({row.status})"
        if row.status in ("moderate", "major"):
            logger.warning(msg)
        else:
            logger.info(msg)
    return dict(zip(report["metric"], report["status"], strict=True))
