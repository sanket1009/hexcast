"""
acs_race_ethnicity.py
=====================
Pull Census ACS race and ethnicity counts (table B03002) for each tract.

Race is not an index input. It is used only to audit outcomes: who lives
in CRITICAL tracts, and whether the ML watch list misses some groups more
than others (Phase 6 audited income, vehicle access and English only).

Groups (B03002, Hispanic or Latino origin by race):
    latino   Hispanic or Latino, any race         B03002_012
    white    not Hispanic, White alone            B03002_003
    black    not Hispanic, Black alone            B03002_004
    asian    not Hispanic, Asian alone            B03002_006
    other    not Hispanic: American Indian and    B03002_005, _007,
             Alaska Native, Native Hawaiian and   _008, _009
             Pacific Islander, some other race,
             two or more races
The four smallest groups are combined: at tract level their counts are so
small that the margin of error is usually larger than the estimate. The
five groups add up to the total (B03002_001).

Counts, not just shares: the equity chart asks what share of the county's
residents in each group live in CRITICAL tracts, which is a sum of counts
across tracts. Shares are kept for the tract panel.

Margins of error (90%): a combined count uses sqrt(sum of MOE_i^2); a
share uses the ratio formula from acs_vulnerability (controlled totals
have MOE 0).

Data source:
    ACS 5-year detailed tables, 2020-2024 release (January 29, 2026), same
    vintage and endpoint as acs_vulnerability. Needs CENSUS_API_KEY.

Output:
    acs_race_ethnicity table, one row per tract (WRITE_TRUNCATE; yearly)
        geoid, total, total_moe, <group>, <group>_moe, <group>_pct,
        <group>_pct_moe, city_fips, acs_vintage

Usage:
    PYTHONPATH=. python -m src.ingestion.acs_race_ethnicity
"""

from typing import Any

import numpy as np
import pandas as pd

from src.ingestion.acs_vulnerability import ACS_VINTAGE, CONTROLLED_MOE, _fetch
from src.utils.bigquery import write_dataframe
from src.utils.logger import get_logger

logger = get_logger(__name__)

TABLE = "acs_race_ethnicity"
TOTAL = "B03002_001"
GROUPS: dict[str, list[str]] = {
    "latino": ["B03002_012"],
    "white": ["B03002_003"],
    "black": ["B03002_004"],
    "asian": ["B03002_006"],
    "other": ["B03002_005", "B03002_007", "B03002_008", "B03002_009"],
}


def pull_acs_race_ethnicity(config: dict[str, Any]) -> pd.DataFrame:
    """Fetch B03002 for the city's county and replace the table."""
    api_key = config["secrets"].get("CENSUS_API_KEY", "")
    if not api_key:
        raise ValueError("CENSUS_API_KEY is not set (needed for ACS B03002)")

    raw = _fetch(config["state_fips"], config["county_fips"], api_key, rows=variables())
    df = compute_groups(raw)
    df["city_fips"] = config["full_fips"]
    df["acs_vintage"] = ACS_VINTAGE
    _validate(df, config)
    write_dataframe(df, TABLE, config, write_disposition="WRITE_TRUNCATE")

    county = {g: round(100 * df[g].sum() / df["total"].sum(), 1) for g in GROUPS}
    logger.info(f"ACS race/ethnicity: {len(df):,} tracts; county shares {county}")
    return df


def variables() -> list[str]:
    return sorted({TOTAL, *(r for rows in GROUPS.values() for r in rows)})


def compute_groups(raw: pd.DataFrame) -> pd.DataFrame:
    """Counts, MOEs and shares per group from raw <row>E / <row>M columns."""
    est = raw.filter(regex="E$").where(lambda d: d >= 0)
    moe = raw.filter(regex="M$")
    moe = moe.mask(moe == CONTROLLED_MOE, 0).where(lambda d: d >= 0)

    total, total_moe = est[f"{TOTAL}E"], moe[f"{TOTAL}M"]
    out = pd.DataFrame({"geoid": raw["geoid"], "total": total, "total_moe": total_moe})
    for name, rows in GROUPS.items():
        n = sum(est[f"{r}E"] for r in rows)
        n_moe = np.sqrt(sum(moe[f"{r}M"] ** 2 for r in rows))
        p = (n / total).where(total > 0)
        radicand = n_moe**2 - p**2 * total_moe**2
        radicand = radicand.where(radicand >= 0, n_moe**2 + p**2 * total_moe**2)
        out[name] = n
        out[f"{name}_moe"] = n_moe.round(1)
        out[f"{name}_pct"] = (100 * p).round(2)
        out[f"{name}_pct_moe"] = (
            (100 * np.sqrt(radicand) / total).where(p.notna()).round(2)
        )
    return out


def _validate(df: pd.DataFrame, config: dict[str, Any]) -> None:
    if df.empty:
        raise ValueError("ACS returned no tracts")
    if df["geoid"].duplicated().any():
        raise ValueError("Duplicate tracts in ACS output")
    expected = config.get("census_tract_count")
    if expected and abs(len(df) - expected) > 0.02 * expected:
        raise ValueError(f"ACS returned {len(df)} tracts, expected about {expected}")
    parts = df[list(GROUPS)].sum(axis=1)
    bad = (parts - df["total"]).abs() > 0.5
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} tracts where groups do not add up to the total"
        )


if __name__ == "__main__":
    from src.utils.config_loader import load_config

    pull_acs_race_ethnicity(load_config("los_angeles"))
