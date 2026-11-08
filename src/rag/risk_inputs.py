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

Vulnerable groups:
    The default group from config. The Census ACS module will replace this.
"""

from typing import Any

import pandas as pd

from src.rag.briefing import TractRisk
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

# plain-language indicator name -> source column in the query below
INDICATORS = {
    "PM2.5, Kriged annual mean (µg/m³)": "pm25_kriged",
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

    targets = scores[scores["burden_class"].isin(URGENCY_BY_CLASS)]
    risks = [_to_tract_risk(row, config, tract_count) for _, row in targets.iterrows()]

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


def _to_tract_risk(
    row: pd.Series, config: dict[str, Any], tract_count: int
) -> TractRisk:
    drivers = [
        name
        for name, col in DRIVER_COMPONENTS.items()
        if row[f"{col}_pct"] >= TOP_QUARTILE
    ]
    if not drivers:
        # Fallback: the single highest component, so every briefing has a driver
        drivers = [max(DRIVER_COMPONENTS, key=lambda n: row[DRIVER_COMPONENTS[n]])]

    # Round for the prompt: raw floats like 55.610474406949436 read badly
    indicators = {
        label: round(float(row[col]), 2)
        for label, col in INDICATORS.items()
        if pd.notna(row[col])
    }

    return TractRisk(
        geoid=row["geoid"],
        risk_rank=int(row["burden_rank"]),
        tract_count=tract_count,
        urgency=URGENCY_BY_CLASS[row["burden_class"]],
        drivers=drivers,
        indicators=indicators,
        vulnerable_groups=[],  # empty -> briefing.py uses default_vulnerable_group
    )
