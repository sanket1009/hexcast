"""
schemas.py
==========
Response models for the CHBIndex API.

FastAPI checks every response against these models and builds the /docs page
from them, so the dashboard (and anyone else) gets a written contract: field
names, types, and which fields can be null. Nulls are real here: a tract with
no ML probability is CRITICAL already, a tract with no access score has no
residents or no road connection, and chronic counts are null until 52 weeks
of history exist.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

BurdenClass = Literal["LOW", "ELEVATED", "HIGH", "CRITICAL"]
Component = Literal["heat", "air_quality", "healthcare_access", "vulnerability"]
Confidence = Literal["HIGH", "MEDIUM", "LOW"]


class Health(BaseModel):
    status: Literal["ok"]
    backend: str
    latest_week: date


class Meta(BaseModel):
    city_fips: str
    city_name: str
    weeks: list[date]
    latest_week: date
    classes: list[BurdenClass]
    components: dict[Component, str] = Field(
        description="Component key -> plain-language name"
    )
    watchlist_threshold: float


class Tract(BaseModel):
    """One tract in one week, with every map layer."""

    geoid: str
    name: str | None = None
    population: int | None = None
    median_income: float | None = None
    centroid_lat: float | None = None
    centroid_lon: float | None = None
    region_name: str | None = Field(None, description="Service Planning Area")
    region_id: str | None = None
    community_name: str | None = None
    city: str | None = None
    neighborhood: str | None = None

    burden_class: BurdenClass
    burden_percentile: float
    raw_burden: float
    pm25_z: float | None = None
    lst_z: float | None = None
    access_z: float | None = None
    vulnerability_z: float | None = None
    main_driver: Component | None = None
    drivers: list[Component] = Field(
        description="Components in the city's top quarter, strongest first"
    )

    hotspot_class: str | None = None
    gi_star_zscore: float | None = None
    weeks_critical: int | None = Field(
        None, description="Weeks CRITICAL in the last 52"
    )
    weeks_observed: int | None = None
    is_chronic: bool | None = Field(None, description="Null until 52 weeks exist")

    p_critical: float | None = Field(
        None, description="Share of 200 plausible versions where the tract is CRITICAL"
    )
    agreement: float | None = Field(
        None,
        description="Share of versions that agree with this week's CRITICAL-or-not call",
    )
    confidence: Confidence | None = None
    confidence_source: Literal["pm25_interpolation", "component_weights"] | None = None
    percentile_p05: float | None = None
    percentile_p95: float | None = None
    retention_pm25: float | None = None
    retention_weights: float | None = None

    prob_critical_4w: float | None = Field(
        None,
        description="Probability of becoming CRITICAL within 4 weeks; null if CRITICAL now",
    )
    ml_rank: int | None = None

    pm25_kriged: float | None = Field(None, description="PM2.5 this week, ug/m3")
    lst_mean_celsius: float | None = None
    access_score_er: float | None = Field(
        None, description="Hospital beds per 1,000 residents within a 30-minute drive"
    )
    er_count_30min: int | None = None
    nearest_er_minutes: float | None = None
    access_scored: bool | None = None

    prev_class: BurdenClass | None = None
    prev_percentile: float | None = None
    percentile_change: float | None = None


class WeekTracts(BaseModel):
    week: date
    model_version: str | None
    tracts: list[Tract]


class Headline(BaseModel):
    title: str
    detail: str


class Chronic(BaseModel):
    tracts: int
    people: int


class Watch(BaseModel):
    threshold: float
    tracts: int
    expected_entering: float = Field(
        description="Sum of calibrated probabilities: expected tracts entering CRITICAL"
    )
    model_version: str


class Kpis(BaseModel):
    people_critical: int
    people_critical_share: float | None
    people_critical_change: int | None
    tracts_critical: int
    chronic: Chronic | None
    desert_people: int = Field(description="People with no ER within a 30-minute drive")
    desert_tracts: int
    watchlist: Watch | None
    critical_confidence: dict[Confidence, int]
    critical_main_driver: dict[Component, int]


class DataSource(BaseModel):
    source: str
    covers: str
    note: str


class Summary(BaseModel):
    week: date
    compare_week: date | None
    headline: Headline
    kpis: Kpis
    population_total: int
    data_sources: list[DataSource]


class Contribution(BaseModel):
    feature: str
    label: str
    value: float | None
    imputed: bool
    contribution: float = Field(description="Log-odds term: coefficient x scaled value")


class Contributions(BaseModel):
    logit: float
    top: list[Contribution]
    other_features: float
    n_other: int


class Briefing(BaseModel):
    urgency: str | None
    summary: str | None
    key_risks: list | None
    actions: list | dict | None
    vulnerable_groups: list | None
    data_gaps: list | None
    sources: list | None
    model: str | None
    generated_at: str | None


class TractDetail(Tract):
    week_start_date: date
    model_version: str | None = None
    ml_contributions: Contributions | None
    acs: dict | None
    briefing: Briefing | None


class TrendPoint(BaseModel):
    geoid: str
    week_start_date: date
    burden_class: BurdenClass
    burden_percentile: float
    pm25_z: float | None
    lst_z: float | None
    access_z: float | None
    vulnerability_z: float | None


class WatchItem(BaseModel):
    geoid: str
    name: str | None
    city: str | None
    neighborhood: str | None
    region_name: str | None
    population: int | None
    burden_class: BurdenClass
    burden_percentile: float
    percentile_change: float | None
    prob_critical_4w: float
    ml_rank: int | None
    main_driver: Component | None
    drivers: list[Component]
    weeks_critical: int | None
    confidence: Confidence | None
    ml_contributions: Contributions | None = Field(
        description="The 3 largest terms pushing the probability up"
    )


class Watchlist(BaseModel):
    week: date
    tracts: list[WatchItem]


class EquityGroup(BaseModel):
    group: str
    label: str
    unit: Literal["people", "households"]
    total: int
    in_critical: int
    share_in_critical: float | None
    ratio_to_all: float | None = Field(
        description="Share in CRITICAL relative to all residents (1.0 = same)"
    )


class Equity(BaseModel):
    week: date
    groups: list[EquityGroup]


class DriftCheck(BaseModel):
    metric: str
    value: float | None
    status: str
    moderate_threshold: float | None
    major_threshold: float | None


class CesAgreement(BaseModel):
    tracts: int
    spearman: float
    critical_in_ces_top_quarter: float | None


class WeekHealth(BaseModel):
    week_start_date: date
    kriging_loo_mae: float | None
    n_monitors: int | None
    lst_coverage: float | None
    critical_high: int | None = None
    critical_medium: int | None = None
    critical_low: int | None = None


class Validation(BaseModel):
    week: date
    drift_checks: list[DriftCheck]
    ces_agreement: CesAgreement | None
    weekly: list[WeekHealth]


class Hexes(BaseModel):
    """Columnar to keep 24,000 cells small: index i of each list is one cell."""

    h3_index: list[str]
    geoid: list[str]
    population: list[int]


class SearchHit(BaseModel):
    kind: Literal["tract", "city", "region", "community"]
    label: str
    geoids: list[str]


class GeocodeHit(BaseModel):
    geoid: str
    matched_address: str
    lat: float
    lon: float
