"""
risk_inputs.py
==============
Build the TractRisk records that briefing.py consumes, from the weekly
warehouse tables (compound_scores, hotspot_detection, and the inputs).

Who gets a briefing:
    HIGH and CRITICAL burden classes only. Each briefing is a paid model
    call, so lower tiers are skipped.

Urgency:
    Set from burden_class, so there is one source of truth.
        CRITICAL -> CRITICAL
        HIGH     -> ELEVATED

Drivers:
    A component is a driver when the tract is in the city's top quarter for
    that component. If no component qualifies, the single highest one is the
    driver, so every briefing has at least one driver.
    access_z is signed so that positive means worse access, so the same
    top-quarter rule flags the worst-access tracts, including deserts.

    Drivers are listed strongest first (largest signed z), so drivers[0] is
    the tract's main hazard.

Vulnerable groups (Census ACS, Phase 3):
    Up to MAX_GROUPS groups per tract: the ACS indicators where the tract is
    in the county's top quarter, highest first. Ranks use the shrunk shares
    from compound_score (noisy estimates pulled toward the mean), so a noisy
    under-5 estimate does not win by chance. Each group's share and the
    county median are added to the indicators, so the numbers in a briefing
    come from code. Tracts with no group in the top quarter get the default
    group from config.
"""

from typing import Any

import pandas as pd

from src.rag.briefing import TractRisk
from src.spatial.compound_score import VULNERABILITY_INDICATORS, shrink_share
from src.utils.bigquery import read_query
from src.utils.logger import get_logger

logger = get_logger(__name__)

URGENCY_BY_CLASS = {"CRITICAL": "CRITICAL", "HIGH": "ELEVATED"}

# driver name (key of driver_queries) -> compound_scores column with its signed z
DRIVER_COMPONENTS = {
    "heat": "lst_z",
    "air_quality": "pm25_z",
    "healthcare_access": "access_z",
}

TOP_QUARTILE = 0.75
MAX_GROUPS = 2

# ACS indicator -> plain-language group and the unit its share counts
GROUP_LABELS = {
    "age_65_plus": "older adults (65 and over)",
    "under_5": "young children (under 5)",
    "low_income": "low-income residents",
    "no_vehicle": "households without a car",
    "limited_english": "limited-English households",
    "uninsured": "people without health insurance",
    "disability": "people with disabilities",
}
GROUP_KEYS = {label: key for key, label in GROUP_LABELS.items()}
# What each share counts (ACS universe): households for these, residents otherwise
HOUSEHOLD_GROUPS = {"no_vehicle", "limited_english"}

# plain-language indicator name -> source column in the query below
INDICATORS = {
    "PM2.5, annual mean (µg/m³)": "pm25_kriged",
    "Land surface temperature, mean (°C)": "lst_mean_celsius",
    "Hospital beds per 1,000 residents, 30-minute drive": "access_score_er",
    "Drive time to nearest ER (minutes)": "nearest_er_minutes",
    "Asthma prevalence, adults (%)": "asthma_pct",
}


def build_tract_risks(config: dict[str, Any], week_str: str) -> list[TractRisk]:
    """Return one TractRisk per HIGH or CRITICAL tract for the week."""
    scores = _load_week(config, week_str)
    if scores.empty:
        raise ValueError(f"No compound scores for week {week_str}")
    groups = _load_group_ranks(config)
    scores = scores.merge(groups, on="geoid", how="left")

    # Percentiles are city-wide, so compute them before filtering
    for col in DRIVER_COMPONENTS.values():
        scores[f"{col}_pct"] = scores[col].rank(pct=True)

    # Rank 1 = highest burden. The prompt gets a rank, not a percentile, so the
    # model cannot write "100th percentile" for a tract at 99.8. Ranked on raw
    # burden, not the percentile (rounded to 0.1), so ties only happen on
    # identical scores; the weekly report ranks the same way.
    scores["burden_rank"] = (
        scores["raw_burden"].rank(ascending=False, method="min").astype(int)
    )
    tract_count = len(scores)

    medians = {k: float(groups[f"{k}_share"].median()) for k in GROUP_LABELS}
    targets = scores[scores["burden_class"].isin(URGENCY_BY_CLASS)]
    risks = [
        _to_tract_risk(row, config, tract_count, medians)
        for _, row in targets.iterrows()
    ]

    logger.info(
        f"Built {len(risks)} briefing inputs "
        f"({(targets['burden_class'] == 'CRITICAL').sum()} CRITICAL, "
        f"{(targets['burden_class'] == 'HIGH').sum()} HIGH)"
    )
    return risks


def _load_week(config: dict[str, Any], week_str: str) -> pd.DataFrame:
    """One row per city tract with scores, hotspot class and indicator inputs."""
    query = f"""
        SELECT
            s.geoid,
            s.burden_class,
            s.burden_percentile,
            s.raw_burden,
            s.pm25_z,
            s.lst_z,
            s.access_z,
            h.hotspot_class,
            k.pm25_kriged,
            l.lst_mean_celsius,
            a.access_score_er,
            a.nearest_er_minutes,
            c.asthma_pct
        FROM {{dataset}}.compound_scores s
        LEFT JOIN {{dataset}}.hotspot_detection h
            ON h.geoid = s.geoid AND h.week_start_date = s.week_start_date
        LEFT JOIN {{dataset}}.pm25_kriged k
            ON k.geoid = s.geoid AND k.week_start_date = s.week_start_date
        LEFT JOIN {{dataset}}.lst_tract_stats l
            ON l.geoid = s.geoid AND l.week_start_date = s.week_start_date
        LEFT JOIN {{dataset}}.healthcare_access a
            ON a.geoid = s.geoid AND a.week_start_date = s.week_start_date
        LEFT JOIN {{dataset}}.cdc_places_health c
            ON c.geoid = s.geoid
        WHERE s.week_start_date = '{week_str}'
          AND s.city_fips = '{config["full_fips"]}'
        ORDER BY s.geoid
    """
    return read_query(query, config)


def _load_group_ranks(config: dict[str, Any]) -> pd.DataFrame:
    """
    geoid plus, per ACS indicator, the published share (<key>_share, %) and
    the county percentile of the shrunk share (<key>_rank, 0-1).
    """
    acs = read_query(
        f"""
        SELECT * FROM {{dataset}}.acs_vulnerability
        WHERE city_fips = '{config["full_fips"]}'
        """,
        config,
    )
    out = pd.DataFrame({"geoid": acs["geoid"]})
    for key in VULNERABILITY_INDICATORS:
        shrunk = shrink_share(acs[f"{key}_pct"], acs[f"{key}_moe"])
        out[f"{key}_share"] = acs[f"{key}_pct"]
        out[f"{key}_rank"] = shrunk.rank(pct=True)
    return out


def top_groups(row: pd.Series, k: int = MAX_GROUPS) -> list[str]:
    """ACS indicator keys where the tract is in the county top quarter, highest first."""
    ranked = [
        (row[f"{key}_rank"], key)
        for key in GROUP_LABELS
        if pd.notna(row.get(f"{key}_rank")) and row[f"{key}_rank"] >= TOP_QUARTILE
    ]
    return [key for _, key in sorted(ranked, reverse=True)[:k]]


def _to_tract_risk(
    row: pd.Series,
    config: dict[str, Any],
    tract_count: int,
    medians: dict[str, float] | None = None,
) -> TractRisk:
    drivers = [
        name
        for name, col in DRIVER_COMPONENTS.items()
        if row[f"{col}_pct"] >= TOP_QUARTILE
    ]
    if not drivers:
        # Fallback: the single highest component, so every briefing has a driver
        drivers = [max(DRIVER_COMPONENTS, key=lambda n: row[DRIVER_COMPONENTS[n]])]
    # Strongest first: drivers[0] picks the group actions
    drivers.sort(key=lambda n: row[DRIVER_COMPONENTS[n]], reverse=True)

    # Round for the prompt: raw floats like 55.610474406949436 read badly
    indicators = {
        label: round(float(row[col]), 2)
        for label, col in INDICATORS.items()
        if pd.notna(row[col])
    }

    groups = top_groups(row)
    for key in groups:
        share = row.get(f"{key}_share")
        if pd.notna(share):
            indicators[group_share_label(key)] = round(float(share), 1)
            if medians and key in medians:
                indicators[group_median_label(key)] = round(medians[key], 1)

    return TractRisk(
        geoid=row["geoid"],
        risk_rank=int(row["burden_rank"]),
        tract_count=tract_count,
        urgency=URGENCY_BY_CLASS[row["burden_class"]],
        drivers=drivers,
        indicators=indicators,
        # empty -> briefing.py uses default_vulnerable_group
        vulnerable_groups=[GROUP_LABELS[k] for k in groups],
    )


def group_share_label(key: str) -> str:
    """e.g. 'Older adults (65 and over), share of tract (%)'."""
    label = GROUP_LABELS[key]
    return f"{label[0].upper()}{label[1:]}, share of tract (%)"


def group_median_label(key: str) -> str:
    label = GROUP_LABELS[key]
    return f"{label[0].upper()}{label[1:]}, county median (%)"
