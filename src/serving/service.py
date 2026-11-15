"""
service.py
==========
Everything the API returns, as plain Python: queries, joins and the small
amount of logic on top (drivers, KPIs, headline, equity shares). api.py only
turns these results into HTTP responses, so this layer is tested directly.

Data flow
    Weekly data is small (2,498 tracts), so each week's tract table is built
    once with one joined query and kept in memory for cache_seconds (default
    600). Summary, watch list, equity and the tract panel all read that
    table. Static tables (tracts, hexes, outlines, ACS) are cached the same
    way. A new or rescored week shows up within cache_seconds.

    Precomputed upstream, never here: scores, classes, hotspots, chronic
    weeks, confidence draws, ML probabilities and per-feature contributions.
    The API needs no GeoPandas, scikit-learn or Kriging code.

Population is ACS 2020-2024 (tract_geometries.population, 9.8M) everywhere
people are counted; the hex map uses Census 2020 blocks only to place cells.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from collections.abc import Callable
from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.serving.store import Store
from src.utils.logger import get_logger

logger = get_logger(__name__)

CLASSES = ["LOW", "ELEVATED", "HIGH", "CRITICAL"]

# component -> signed z column in compound_scores (higher = worse for all four;
# access_z already carries the minus sign)
COMPONENTS = {
    "heat": "lst_z",
    "air_quality": "pm25_z",
    "healthcare_access": "access_z",
    "vulnerability": "vulnerability_z",
}
COMPONENT_LABELS = {
    "heat": "heat",
    "air_quality": "air quality",
    "healthcare_access": "healthcare access",
    "vulnerability": "population vulnerability",
}
# Same rule as the briefings (src/rag/risk_inputs.py): a component drives a
# tract when it is in the city's top quarter for that component
DRIVER_PERCENTILE = 0.75

FEATURE_LABELS = {
    "burden_percentile": "Burden percentile this week",
    "raw_burden": "Burden score this week",
    "pm25_z": "Air quality component",
    "lst_z": "Heat component",
    "access_z": "Healthcare access component",
    "vulnerability_z": "Vulnerability component",
    "pct_lag_1w": "Burden percentile 1 week ago",
    "pct_lag_2w": "Burden percentile 2 weeks ago",
    "pct_lag_4w": "Burden percentile 4 weeks ago",
    "pct_delta_1w": "Percentile change over 1 week",
    "pct_delta_4w": "Percentile change over 4 weeks",
    "pct_mean_4w": "Mean percentile, last 4 weeks",
    "pct_std_4w": "Percentile swings, last 4 weeks",
    "weeks_critical_prev4": "Weeks CRITICAL, last 4",
    "pm25_z_delta_1w": "Air quality change over 1 week",
    "lst_z_delta_1w": "Heat change over 1 week",
    "pm25_1w": "PM2.5 this week",
    "pm25_1w_minus_4w": "PM2.5 this week minus 4-week mean",
    "lst_mean_c": "Land surface temperature",
    "population": "Population",
    "median_income": "Median household income",
    "age_65_plus_pct": "Residents 65 and over",
    "under_5_pct": "Children under 5",
    "low_income_pct": "Low-income residents",
    "no_vehicle_pct": "Households without a car",
    "limited_english_pct": "Limited-English households",
    "uninsured_pct": "Residents without health insurance",
    "disability_pct": "Residents with a disability",
    "asthma_pct": "Adult asthma",
    "copd_pct": "Adult COPD",
    "heart_disease_pct": "Adult heart disease",
    "nearest_er_minutes": "Drive time to nearest ER",
    "er_count_30min": "ERs within 30 minutes",
}

# Equity groups: ACS share column, its universe column, and the unit counted
EQUITY_ACS = {
    "no_vehicle": ("no_vehicle_pct", "no_vehicle_base", "households"),
    "limited_english": ("limited_english_pct", "limited_english_base", "households"),
}
EQUITY_RACE = {
    "latino": "Latino",
    "white": "White, not Latino",
    "black": "Black, not Latino",
    "asian": "Asian, not Latino",
    "other": "Other or multiple races, not Latino",
}

ROUNDING = {
    "centroid_lat": 5,
    "centroid_lon": 5,
    "percentile_change": 1,
    "lst_mean_celsius": 1,
    "pm25_kriged": 2,
    "access_score_er": 2,
    "nearest_er_minutes": 1,
    "agreement": 3,
    "gi_star_zscore": 3,
}

GEOID = re.compile(r"^\d{11}$")
CENSUS_GEOCODER = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"


class NotFoundError(Exception):
    """A week, tract or address the warehouse does not have."""


class UpstreamError(Exception):
    """An outside service (the Census Geocoder) failed."""


class _TTLCache:
    """Per-key results that expire after `seconds`; thread-safe for uvicorn."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._items: dict[Any, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: Any, build: Callable[[], Any]) -> Any:
        now = time.monotonic()
        with self._lock:
            hit = self._items.get(key)
            if hit and now - hit[0] < self.seconds:
                return hit[1]
        value = build()  # outside the lock: a slow query must not block others
        with self._lock:
            self._items[key] = (now, value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


def _none(value: Any) -> Any:
    """NaN/NA -> None so JSON gets null; numpy scalars -> Python."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float | np.floating) and math.isnan(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {k: _none(v) for k, v in row.items()} for row in df.to_dict(orient="records")
    ]


def format_people(n: float) -> str:
    """Round to what the data supports: 1.1M, 812K, 9,400."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{round(n / 1000):,}K"
    if n >= 1_000:
        return f"{round(n, -2):,.0f}"
    return f"{n:,.0f}"


def split_community(name: Any) -> tuple[str | None, str | None]:
    """'City of Glendale' -> (Glendale, None); 'Los Angeles - Watts' -> (Los Angeles, Watts)."""
    if not isinstance(name, str) or not name:
        return None, None
    if " - " in name:
        city, hood = name.split(" - ", 1)
        return city, hood
    return name.removeprefix("City of "), None


def add_drivers(df: pd.DataFrame) -> pd.DataFrame:
    """main_driver: the largest component; drivers: those in the top quarter."""
    names = list(COMPONENTS)
    z = df[list(COMPONENTS.values())].to_numpy(dtype=float)
    pct = pd.DataFrame(z).rank(pct=True).to_numpy()
    # Strongest component first in every row; missing components sort last
    order = np.argsort(-np.nan_to_num(z, nan=-np.inf), axis=1)
    has_any = ~np.isnan(z).all(axis=1)

    main, drivers = [], []
    for i in range(len(df)):
        if not has_any[i]:
            main.append(None)
            drivers.append([])
            continue
        top = [names[j] for j in order[i] if pct[i, j] >= DRIVER_PERCENTILE]
        main.append(names[order[i, 0]])
        drivers.append(top or [names[order[i, 0]]])
    df["main_driver"] = main
    df["drivers"] = drivers
    return df


class Service:
    def __init__(self, store: Store, config: dict[str, Any]) -> None:
        self.store = store
        self.config = config
        self.city = config["full_fips"]
        product = config.get("product", {})
        self.watchlist_threshold = float(product.get("watchlist_threshold", 0.5))
        self.cache = _TTLCache(float(product.get("api_cache_seconds", 600)))

    def _q(self, sql: str, **params: Any) -> pd.DataFrame:
        return self.store.query(sql, {"city": self.city, **params})

    # ------------------------------------------------------------------ weeks

    def weeks(self) -> list[date]:
        def build() -> list[date]:
            df = self._q(
                """
                SELECT DISTINCT week_start_date FROM {dataset}.compound_scores
                WHERE city_fips = @city ORDER BY week_start_date
                """
            )
            return [pd.Timestamp(w).date() for w in df["week_start_date"]]

        return self.cache.get("weeks", build)

    def resolve_week(self, week: str | date | None) -> date:
        weeks = self.weeks()
        if not weeks:
            raise NotFoundError("The warehouse has no scored weeks")
        if week in (None, "", "latest"):
            return weeks[-1]
        try:
            day = week if isinstance(week, date) else date.fromisoformat(str(week))
        except ValueError as exc:
            raise NotFoundError(f"Not a date: {week!r} (use YYYY-MM-DD)") from exc
        if day not in weeks:
            raise NotFoundError(
                f"No scores for week {day}; weeks run Monday to Monday "
                f"from {weeks[0]} to {weeks[-1]}"
            )
        return day

    def previous_week(self, week: date) -> date | None:
        weeks = self.weeks()
        i = weeks.index(week)
        return weeks[i - 1] if i > 0 else None

    # ----------------------------------------------------------------- static

    def static_tracts(self) -> pd.DataFrame:
        def build() -> pd.DataFrame:
            df = self._q(
                """
                SELECT g.geoid, g.name, g.population, g.median_income,
                       g.centroid_lat, g.centroid_lon,
                       ra.region_name, ra.region_id, ra.community_name
                FROM {dataset}.tract_geometries g
                LEFT JOIN {dataset}.tract_reporting_areas ra
                  ON ra.geoid = g.geoid AND ra.city_fips = g.city_fips
                WHERE g.city_fips = @city
                """
            )
            parts = df["community_name"].map(split_community)
            df["city"] = [p[0] for p in parts]
            df["neighborhood"] = [p[1] for p in parts]
            return df

        return self.cache.get("static", build)

    def acs(self) -> pd.DataFrame:
        def build() -> pd.DataFrame:
            vuln = self._q(
                "SELECT * FROM {dataset}.acs_vulnerability WHERE city_fips = @city"
            )
            race = self._q(
                """
                SELECT geoid, total, latino, white, black, asian, other,
                       latino_pct, white_pct, black_pct, asian_pct, other_pct
                FROM {dataset}.acs_race_ethnicity WHERE city_fips = @city
                """
            )
            return vuln.drop(columns=["city_fips", "population"]).merge(
                race, on="geoid", how="outer"
            )

        return self.cache.get("acs", build)

    def hexes(self) -> pd.DataFrame:
        return self.cache.get(
            "hexes",
            lambda: self._q(
                """
                SELECT h3_index, geoid, population FROM {dataset}.h3_hexes
                WHERE city_fips = @city ORDER BY h3_index
                """
            ),
        )

    def outlines_geojson(self) -> str:
        """FeatureCollection text, assembled from stored geometries (no parsing)."""

        def build() -> str:
            df = self._q(
                """
                SELECT geoid, geometry_geojson FROM {dataset}.tract_outlines
                WHERE city_fips = @city ORDER BY geoid
                """
            )
            features = ",".join(
                f'{{"type":"Feature","id":"{g}","properties":{{"geoid":"{g}"}},'
                f'"geometry":{geom}}}'
                for g, geom in zip(df["geoid"], df["geometry_geojson"], strict=False)
            )
            return f'{{"type":"FeatureCollection","features":[{features}]}}'

        return self.cache.get("outlines", build)

    # ------------------------------------------------------------- week table

    def tracts(self, week: date) -> pd.DataFrame:
        """One row per tract for the week, every map layer joined."""
        return self.cache.get(("tracts", week), lambda: self._build_tracts(week))

    def _build_tracts(self, week: date) -> pd.DataFrame:
        same = "ON {t}.geoid = cs.geoid AND {t}.week_start_date = cs.week_start_date AND {t}.city_fips = cs.city_fips"
        joins = "\n".join(
            f"LEFT JOIN {{dataset}}.{table} {alias} " + same.format(t=alias)
            for table, alias in [
                ("hotspot_detection", "hs"),
                ("tract_chronic", "ch"),
                ("tract_confidence", "cf"),
                ("ml_predictions", "ml"),
                ("pm25_kriged", "pk"),
                ("lst_tract_stats", "ls"),
                ("healthcare_access", "ha"),
            ]
        )
        df = self._q(
            f"""
            SELECT cs.geoid, cs.burden_class, cs.burden_percentile, cs.raw_burden,
                   cs.pm25_z, cs.lst_z, cs.access_z, cs.vulnerability_z,
                   hs.hotspot_class, hs.gi_star_zscore,
                   ch.weeks_critical, ch.weeks_observed, ch.is_chronic,
                   cf.p_critical, cf.percentile_p05, cf.percentile_p95,
                   cf.confidence, cf.retention_pm25, cf.retention_weights,
                   ml.prob_critical_4w, ml.rank AS ml_rank, ml.model_version,
                   pk.pm25_kriged, ls.lst_mean_celsius,
                   ha.access_score_er, ha.er_count_30min, ha.nearest_er_minutes,
                   ha.scored AS access_scored
            FROM {{dataset}}.compound_scores cs
            {joins}
            WHERE cs.week_start_date = @week AND cs.city_fips = @city
            """,
            week=week,
        )
        if df.empty:
            raise NotFoundError(f"No scores for week {week}")

        df = self.static_tracts().merge(df, on="geoid", how="right")
        df = add_drivers(df)
        crit = df["burden_class"] == "CRITICAL"
        # Confidence badge is about the CRITICAL-or-not call
        df["agreement"] = np.where(crit, df["p_critical"], 1 - df["p_critical"])
        df["confidence_source"] = np.where(
            df["retention_pm25"] <= df["retention_weights"],
            "pm25_interpolation",
            "component_weights",
        )
        df.loc[df["p_critical"].isna(), "confidence_source"] = None

        prev = self.previous_week(week)
        if prev is not None:
            before = self._q(
                """
                SELECT geoid, burden_class AS prev_class,
                       burden_percentile AS prev_percentile
                FROM {dataset}.compound_scores
                WHERE week_start_date = @week AND city_fips = @city
                """,
                week=prev,
            )
            df = df.merge(before, on="geoid", how="left")
            df["percentile_change"] = df["burden_percentile"] - df["prev_percentile"]
        else:
            df["prev_class"] = None
            df["prev_percentile"] = np.nan
            df["percentile_change"] = np.nan
        df["week_start_date"] = week
        # Precision the data supports, and no float noise like 1.7999999999999972
        df = df.round(ROUNDING)
        return df.sort_values("geoid").reset_index(drop=True)

    # ---------------------------------------------------------------- summary

    def data_sources(self, week: date) -> list[dict[str, str]]:
        def build() -> list[dict[str, str]]:
            acs = self._q(
                "SELECT acs_vintage FROM {dataset}.acs_vulnerability "
                "WHERE city_fips = @city LIMIT 1"
            )
            hosp = self._q(
                "SELECT data_date FROM {dataset}.hospitals "
                "WHERE city_fips = @city LIMIT 1"
            )
            return [
                acs.iloc[0, 0] if len(acs) else None,
                hosp.iloc[0, 0] if len(hosp) else None,
            ]

        acs_vintage, hosp_date = self.cache.get("source_dates", build)
        lag = self.config.get("airflow", {}).get("ingest_lag_weeks")
        pm_weeks = self.config.get("pm25_window_weeks", 4)
        lst_days = self.config.get("gee", {}).get("composite_window_days", 32)
        week_label = f"week of {week:%b} {week.day}, {week:%Y}"
        return [
            {
                "source": "EPA AQS PM2.5 monitors",
                "covers": week_label,
                "note": f"Kriged to tracts; the index uses the last {pm_weeks} weeks. "
                + (
                    f"AQS publishes months late, so each run scores the week {lag} weeks back."
                    if lag
                    else "AQS publishes months late."
                ),
            },
            {
                "source": "Landsat 8 and 9 surface temperature",
                "covers": week_label,
                "note": f"Median of clear pixels over the {lst_days} days ending that week.",
            },
            {
                "source": "CDPH emergency departments",
                "covers": f"hospital list dated {hosp_date}"
                if hosp_date
                else "hospital list",
                "note": "30-minute drive times on the OpenStreetMap road network, no traffic.",
            },
            {
                "source": "Census ACS",
                "covers": acs_vintage or "ACS 5-year",
                "note": "Population, income and vulnerable groups.",
            },
        ]

    def summary(self, week: date, compare: date | None = None) -> dict[str, Any]:
        df = self.tracts(week)
        compare = compare if compare is not None else self.previous_week(week)
        other = self.tracts(compare) if compare is not None else None

        total = float(df["population"].sum())
        crit = df["burden_class"] == "CRITICAL"
        people_crit = float(df.loc[crit, "population"].sum())
        change = None
        if other is not None:
            change = people_crit - float(
                other.loc[other["burden_class"] == "CRITICAL", "population"].sum()
            )

        chronic = None
        if df["is_chronic"].notna().any():
            is_chronic = df["is_chronic"].fillna(False).astype(bool)
            chronic = {
                "tracts": int(is_chronic.sum()),
                "people": int(df.loc[is_chronic, "population"].sum()),
            }

        desert = (df["er_count_30min"] == 0) & df["access_scored"].fillna(False).astype(
            bool
        )

        watch = None
        probs = df["prob_critical_4w"].dropna()
        if len(probs):
            watch = {
                "threshold": self.watchlist_threshold,
                "tracts": int((probs >= self.watchlist_threshold).sum()),
                # The model is calibrated (ECE 0.006), so the sum of
                # probabilities is the expected number of tracts entering
                "expected_entering": round(float(probs.sum()), 1),
                "model_version": df["model_version"].dropna().iloc[0],
            }

        mix = df.loc[crit, "confidence"].value_counts()
        drivers = df.loc[crit, "main_driver"].value_counts()
        top_driver = drivers.index[0] if len(drivers) else None

        kpis = {
            "people_critical": int(people_crit),
            "people_critical_share": round(people_crit / total, 4) if total else None,
            "people_critical_change": None if change is None else int(change),
            "tracts_critical": int(crit.sum()),
            "chronic": chronic,
            "desert_people": int(df.loc[desert, "population"].sum()),
            "desert_tracts": int(desert.sum()),
            "watchlist": watch,
            "critical_confidence": {
                k: int(mix.get(k, 0)) for k in ["HIGH", "MEDIUM", "LOW"]
            },
            "critical_main_driver": {k: int(v) for k, v in drivers.items()},
        }
        return {
            "week": week,
            "compare_week": compare,
            "headline": self._headline(week, compare, kpis, top_driver),
            "kpis": kpis,
            "population_total": int(total),
            "data_sources": self.data_sources(week),
        }

    def _headline(
        self, week: date, compare: date | None, kpis: dict, top_driver: str | None
    ) -> dict[str, str]:
        people = format_people(kpis["people_critical"])
        title = f"{people} people live in CRITICAL tracts in the week of {week:%B} {week.day}, {week:%Y}"
        change = kpis["people_critical_change"]
        if change is not None and compare is not None:
            when = f"{compare:%B} {compare.day}"
            if compare.year != week.year:
                when += f", {compare:%Y}"
            if change == 0:
                title += f", the same as {when}"
            else:
                word = "more" if change > 0 else "fewer"
                title += f", {format_people(abs(change))} {word} than {when}"
        detail = ""
        if top_driver:
            n = kpis["critical_main_driver"][top_driver]
            detail = (
                f"{COMPONENT_LABELS[top_driver].capitalize()} is the largest component "
                f"in {n} of {kpis['tracts_critical']} CRITICAL tracts."
            )
        return {"title": title, "detail": detail}

    # ------------------------------------------------------------ tract panel

    def contributions(self, week: date) -> pd.DataFrame:
        return self.cache.get(
            ("contrib", week),
            lambda: self._q(
                """
                SELECT geoid, feature, value, imputed, contribution, logit
                FROM {dataset}.ml_contributions
                WHERE week_start_date = @week AND city_fips = @city
                """,
                week=week,
            ),
        )

    def _contribution_list(self, rows: pd.DataFrame, top: int) -> dict[str, Any] | None:
        if rows.empty:
            return None
        rows = rows.assign(size=rows["contribution"].abs()).sort_values(
            "size", ascending=False
        )
        head, rest = rows.head(top), rows.iloc[top:]
        items = [
            {
                "feature": r.feature,
                "label": FEATURE_LABELS.get(r.feature, r.feature),
                "value": _none(r.value),
                "imputed": bool(r.imputed),
                "contribution": float(r.contribution),
            }
            for r in head.itertuples()
        ]
        return {
            "logit": float(rows["logit"].iloc[0]),
            "top": items,
            "other_features": round(float(rest["contribution"].sum()), 5),
            "n_other": len(rest),
        }

    def tract(self, geoid: str, week: date, top_features: int = 8) -> dict[str, Any]:
        if not GEOID.match(geoid):
            raise NotFoundError(f"Not a tract GEOID: {geoid!r} (11 digits)")
        df = self.tracts(week)
        row = df[df["geoid"] == geoid]
        if row.empty:
            raise NotFoundError(f"Tract {geoid} is not in this city")
        out = records(row)[0]

        contrib = self.contributions(week)
        out["ml_contributions"] = self._contribution_list(
            contrib[contrib["geoid"] == geoid], top_features
        )

        acs = self.acs()
        acs_row = acs[acs["geoid"] == geoid]
        out["acs"] = (
            records(acs_row.drop(columns=["geoid"]))[0] if len(acs_row) else None
        )
        out["briefing"] = self._briefing(geoid, week)
        return out

    def _briefing(self, geoid: str, week: date) -> dict[str, Any] | None:
        df = self._q(
            """
            SELECT urgency, summary, key_risks_json, actions_json,
                   vulnerable_groups_json, data_gaps_json, sources_json,
                   model, generated_at
            FROM {dataset}.briefings
            WHERE geoid = @geoid AND week_start_date = @week AND city_fips = @city
            """,
            geoid=geoid,
            week=week,
        )
        if df.empty:
            return None
        r = df.iloc[0]

        def parse(text: Any) -> Any:
            return json.loads(text) if isinstance(text, str) and text else None

        return {
            "urgency": r["urgency"],
            "summary": r["summary"],
            "key_risks": parse(r["key_risks_json"]),
            "actions": parse(r["actions_json"]),
            "vulnerable_groups": parse(r["vulnerable_groups_json"]),
            "data_gaps": parse(r["data_gaps_json"]),
            "sources": parse(r["sources_json"]),
            "model": r["model"],
            "generated_at": _none(pd.Timestamp(r["generated_at"]).isoformat()),
        }

    def trends(
        self, geoids: list[str], first: date | None = None, last: date | None = None
    ) -> pd.DataFrame:
        bad = [g for g in geoids if not GEOID.match(g)]
        if bad:
            raise NotFoundError(f"Not tract GEOIDs: {bad}")
        # GEOIDs are checked as 11 digits above, so listing them in the SQL is
        # safe; an array parameter is spelled differently in DuckDB and BigQuery
        in_list = ", ".join(f"'{g}'" for g in sorted(set(geoids)))
        df = self._q(
            f"""
            SELECT geoid, week_start_date, burden_class, burden_percentile,
                   pm25_z, lst_z, access_z, vulnerability_z
            FROM {{dataset}}.compound_scores
            WHERE city_fips = @city AND geoid IN ({in_list})
            ORDER BY geoid, week_start_date
            """
        )
        if df.empty:
            raise NotFoundError(f"No scores for {geoids}")
        df["week_start_date"] = pd.to_datetime(df["week_start_date"]).dt.date
        if first:
            df = df[df["week_start_date"] >= first]
        if last:
            df = df[df["week_start_date"] <= last]
        return df

    # -------------------------------------------------------------- watch list

    def watchlist(
        self, week: date, limit: int = 50, min_prob: float | None = None
    ) -> list[dict[str, Any]]:
        df = self.tracts(week)
        df = df[df["prob_critical_4w"].notna()]
        if min_prob is not None:
            df = df[df["prob_critical_4w"] >= min_prob]
        df = df.sort_values("prob_critical_4w", ascending=False).head(limit)
        contrib = self.contributions(week)
        contrib = contrib[contrib["geoid"].isin(df["geoid"])]
        by_geoid = {g: rows for g, rows in contrib.groupby("geoid")}
        cols = [
            "geoid", "name", "city", "neighborhood", "region_name", "population",
            "burden_class", "burden_percentile", "percentile_change",
            "prob_critical_4w", "ml_rank", "main_driver", "drivers",
            "weeks_critical", "confidence",
        ]  # fmt: skip
        out = records(df[cols])
        for item in out:
            rows = by_geoid.get(item["geoid"])
            if rows is None:
                item["ml_contributions"] = None
                continue
            # The strongest reasons this tract is flagged: largest positive terms
            ups = rows[rows["contribution"] > 0]
            item["ml_contributions"] = self._contribution_list(ups, 3)
        return out

    # ------------------------------------------------------------------ equity

    def equity(self, week: date) -> list[dict[str, Any]]:
        df = self.tracts(week)[["geoid", "population", "median_income", "burden_class"]]
        df = df.merge(self.acs(), on="geoid", how="left")
        crit = df["burden_class"] == "CRITICAL"
        groups: list[tuple[str, str, str, pd.Series]] = [
            ("all", "All residents", "people", df["population"])
        ]
        terc = pd.qcut(df["median_income"], 3, labels=["low", "middle", "high"])
        names = {
            "low": "Lowest-income third of tracts",
            "middle": "Middle-income third of tracts",
            "high": "Highest-income third of tracts",
        }
        for t in ["low", "middle", "high"]:
            groups.append(
                (
                    f"income_{t}",
                    names[t],
                    "people",
                    df["population"].where(terc == t, 0),
                )
            )
        labels = {
            "no_vehicle": "Households without a car",
            "limited_english": "Limited-English households",
        }
        for key, (pct, base, unit) in EQUITY_ACS.items():
            groups.append((key, labels[key], unit, df[pct] / 100 * df[base]))
        for key, label in EQUITY_RACE.items():
            groups.append((key, label, "people", df[key]))

        overall = float(df.loc[crit, "population"].sum() / df["population"].sum())
        out = []
        for key, label, unit, count in groups:
            count = count.fillna(0)
            total = float(count.sum())
            inside = float(count[crit].sum())
            share = inside / total if total else None
            out.append(
                {
                    "group": key,
                    "label": label,
                    "unit": unit,
                    "total": round(total),
                    "in_critical": round(inside),
                    "share_in_critical": None if share is None else round(share, 4),
                    "ratio_to_all": None
                    if share is None
                    else round(share / overall, 2),
                }
            )
        return out

    # -------------------------------------------------------------- validation

    def validation(self, week: date) -> dict[str, Any]:
        drift = self._q(
            """
            SELECT metric, value, status, moderate_threshold, major_threshold
            FROM {dataset}.drift_checks
            WHERE week_start_date = @week AND city_fips = @city ORDER BY metric
            """,
            week=week,
        )
        series = self.cache.get("validation_series", self._validation_series)

        ces = self._q(
            """
            SELECT cs.geoid, cs.burden_percentile, cs.burden_class, ce.ces_percentile
            FROM {dataset}.compound_scores cs
            JOIN {dataset}.calenviroscreen ce
              ON ce.geoid = cs.geoid AND ce.city_fips = cs.city_fips
            WHERE cs.week_start_date = @week AND cs.city_fips = @city
            """,
            week=week,
        ).dropna()
        ces_out = None
        if len(ces) > 10:
            crit = ces["burden_class"] == "CRITICAL"
            ces_out = {
                "tracts": len(ces),
                # Spearman = Pearson on ranks (average ranks for ties, as
                # pandas does). Written out because method="spearman" imports
                # scipy, which the API image does not install.
                "spearman": round(
                    float(
                        ces["burden_percentile"]
                        .rank()
                        .corr(ces["ces_percentile"].rank())
                    ),
                    3,
                ),
                "critical_in_ces_top_quarter": round(
                    float((ces.loc[crit, "ces_percentile"] >= 75).mean()), 3
                )
                if crit.any()
                else None,
            }
        return {
            "week": week,
            "drift_checks": records(drift),
            "ces_agreement": ces_out,
            "weekly": series,
        }

    def _validation_series(self) -> list[dict[str, Any]]:
        krig = self._q(
            """
            SELECT week_start_date, AVG(cv_mae) AS kriging_loo_mae,
                   MAX(n_monitors) AS n_monitors
            FROM {dataset}.pm25_kriged WHERE city_fips = @city
            GROUP BY week_start_date
            """
        )
        lst = self._q(
            """
            SELECT week_start_date,
                   AVG(CASE WHEN lst_zscore IS NULL THEN 0.0 ELSE 1.0 END) AS lst_coverage
            FROM {dataset}.lst_tract_stats WHERE city_fips = @city
            GROUP BY week_start_date
            """
        )
        mix = self._q(
            """
            SELECT cf.week_start_date, cf.confidence, COUNT(*) AS n
            FROM {dataset}.tract_confidence cf
            JOIN {dataset}.compound_scores cs
              ON cs.geoid = cf.geoid AND cs.week_start_date = cf.week_start_date
             AND cs.city_fips = cf.city_fips
            WHERE cf.city_fips = @city AND cs.burden_class = 'CRITICAL'
            GROUP BY cf.week_start_date, cf.confidence
            """
        )
        wide = (
            mix.pivot(index="week_start_date", columns="confidence", values="n")
            .reindex(columns=["HIGH", "MEDIUM", "LOW"])
            .fillna(0)
            .astype(int)
            .rename(columns=lambda c: f"critical_{c.lower()}")
            .reset_index()
        )
        out = krig.merge(lst, on="week_start_date", how="outer").merge(
            wide, on="week_start_date", how="left"
        )
        # A week with no CRITICAL tracts has zero of each, not unknown
        counts = [c for c in out if c.startswith("critical_")]
        out[counts] = out[counts].fillna(0).astype(int)
        out["week_start_date"] = pd.to_datetime(out["week_start_date"]).dt.date
        out = out.round({"kriging_loo_mae": 3, "lst_coverage": 4})
        return records(out.sort_values("week_start_date"))

    # -------------------------------------------------------------- evaluation

    def evaluation(self) -> dict[str, Any]:
        """
        Model and RAG evaluation results from eval_summary (src/evals/publish.py).

        Not tied to a week: the watch-list model was evaluated once, out of
        fold over three test windows, and the briefing and retrieval scores
        are the accepted baselines. Each suite is None when not published.
        """
        df = self.cache.get(
            "evaluation",
            lambda: self._q(
                "SELECT * FROM {dataset}.eval_summary WHERE city_fips = @city"
            ),
        )
        if df.empty:
            raise NotFoundError(
                "No evaluation results published; run scripts/publish_evals.py"
            )
        return {
            "ml": self._ml_eval(df[df["suite"] == "ml"]),
            "briefings": self._baseline_eval(df[df["suite"] == "briefings"]),
            "retrieval": self._baseline_eval(df[df["suite"] == "retrieval"]),
        }

    @staticmethod
    def _baseline_eval(df: pd.DataFrame) -> dict[str, Any] | None:
        if df.empty:
            return None
        first = df.iloc[0]
        return {
            "model": _none(first["model"]),
            "detail": _none(first["detail"]),
            "git_sha": _none(first["git_sha"]),
            "evaluated_at": _none(first["evaluated_at"]),
            "metrics": {r.metric: float(r.value) for r in df.itertuples()},
        }

    @staticmethod
    def _ml_eval(df: pd.DataFrame) -> dict[str, Any] | None:
        if df.empty:
            return None

        def part(section: str) -> pd.DataFrame:
            return df[df["section"] == section]

        def wide(section: str, index: list[str]) -> pd.DataFrame:
            return (
                part(section)
                .pivot_table(
                    index=index, columns="metric", values="value", aggfunc="first"
                )
                .reset_index()
            )

        panel = part("panel").set_index("metric")
        ref = panel.loc["reference"]

        pooled = wide("pooled", ["model"]).sort_values("pr_auc", ascending=False)

        windows = []
        win = part("window")
        for sub, rows in win.groupby("subgroup", sort=True):
            base = rows[rows["metric"] == "base_rate"]["value"]
            scores = rows[rows["metric"] == "pr_auc"]
            windows.append(
                {
                    "window": int(sub),
                    "base_rate": float(base.iloc[0]),
                    "pr_auc": dict(
                        zip(scores["model"], scores["value"].astype(float), strict=True)
                    ),
                }
            )

        comp = wide("comparison", ["model"])
        n_boot = part("comparison").groupby("model")["n"].max()
        comp["n_boot"] = comp["model"].map(n_boot).astype("Int64")

        cal = part("calibration").set_index("metric")["value"]
        rel = wide("reliability", ["subgroup"])
        rel["n"] = rel["subgroup"].map(
            part("reliability").groupby("subgroup")["n"].first()
        )
        rel["bin"] = rel["subgroup"].astype(int)

        eq = wide("equity", ["subgroup"])
        eq[["attribute", "tercile"]] = eq["subgroup"].str.split(":", n=1, expand=True)
        eq["tracts"] = eq["subgroup"].map(
            part("equity").groupby("subgroup")["n"].first()
        )
        summ = part("equity_summary")
        ratio = summ[summ["metric"] == "recall_ratio_min_max"].set_index("subgroup")
        flag = summ[summ["metric"] == "flag"].set_index("subgroup")["value"]
        equity_summary = [
            {
                "attribute": attr,
                "recall_ratio_min_max": float(r["value"]),
                "lowest_recall_tercile": _none(r["detail"]),
                "flag": bool(flag.get(attr, 0)),
            }
            for attr, r in ratio.iterrows()
        ]

        return {
            "evaluated_at": _none(df["evaluated_at"].iloc[0]),
            "git_sha": _none(df["git_sha"].iloc[0]),
            "reference": ref["model"],
            "target": _none(ref["detail"]),
            "weeks": _none(panel.loc["rows", "detail"]),
            "rows": int(panel.loc["rows", "value"]),
            "test_rows": int(panel.loc["test_rows", "value"]),
            "base_rate": float(panel.loc["base_rate", "value"]),
            "test_windows": int(panel.loc["test_windows", "value"]),
            "top_k": int(panel.loc["top_k_per_week", "value"])
            if "top_k_per_week" in panel.index
            else None,
            "scorers": records(pooled),
            "windows": windows,
            "comparisons": records(comp),
            "calibration": {
                "brier": float(cal["brier"]),
                "brier_base_rate_only": float(cal["brier_base_rate_only"]),
                "ece": float(cal["ece"]),
                "reliability": records(
                    rel[["bin", "predicted", "observed", "n"]].sort_values("bin")
                ),
            },
            "equity": records(
                eq[
                    [
                        "attribute",
                        "tercile",
                        "tracts",
                        "base_rate",
                        "pr_auc",
                        "topk_recall",
                        "calibration_gap",
                    ]
                ]
            ),  # fmt: skip
            "equity_summary": equity_summary,
        }

    # ------------------------------------------------------------------ search

    def search(self, q: str, limit: int = 10) -> list[dict[str, Any]]:
        q = q.strip()
        static = self.static_tracts()
        out: list[dict[str, Any]] = []
        if q.isdigit():
            hits = static[
                static["geoid"].str.startswith(q)
                | static["geoid"].str[5:].str.startswith(q)
            ]
            for r in hits.head(limit).itertuples():
                out.append(
                    {
                        "kind": "tract",
                        "label": f"{r.geoid} {r.name or ''}".strip(),
                        "geoids": [r.geoid],
                    }
                )
            return out

        needle = q.lower()
        seen: set[frozenset[str]] = set()  # "City of Glendale" is also city Glendale
        for kind, col in [
            ("city", "city"),
            ("region", "region_name"),
            ("community", "community_name"),
        ]:
            names = static[col].dropna()
            matched = names[
                names.str.lower().str.contains(needle, regex=False)
            ].unique()
            # Names that start with the query first, then the rest, A to Z
            matched = sorted(
                matched, key=lambda n: (not n.lower().startswith(needle), n)
            )
            for name in matched:
                geoids = sorted(static.loc[static[col] == name, "geoid"])
                if frozenset(geoids) in seen:
                    continue
                seen.add(frozenset(geoids))
                label = name if kind != "region" else f"{name} (Service Planning Area)"
                if kind == "city" and name in ("Los Angeles", "Unincorporated"):
                    label = f"{name} (all areas)"
                out.append({"kind": kind, "label": label, "geoids": geoids})
        return out[:limit]

    def geocode(self, address: str) -> dict[str, Any]:
        try:
            resp = requests.get(
                CENSUS_GEOCODER,
                params={
                    "address": address,
                    "benchmark": "Public_AR_Current",
                    "vintage": "Current_Current",
                    "layers": "Census Tracts",
                    "format": "json",
                },
                timeout=15,
            )
            resp.raise_for_status()
            matches = resp.json()["result"]["addressMatches"]
        except (requests.RequestException, KeyError, ValueError) as exc:
            raise UpstreamError(f"Census Geocoder failed: {exc}") from exc
        if not matches:
            raise NotFoundError(f"The Census Geocoder found no match for {address!r}")
        m = matches[0]
        tracts = m.get("geographies", {}).get("Census Tracts", [])
        if not tracts:
            raise NotFoundError("The matched address has no census tract")
        geoid = tracts[0]["GEOID"]
        if geoid not in set(self.static_tracts()["geoid"]):
            raise NotFoundError(
                f"{m['matchedAddress']} is in tract {geoid}, outside this county"
            )
        return {
            "geoid": geoid,
            "matched_address": m["matchedAddress"],
            "lat": m["coordinates"]["y"],
            "lon": m["coordinates"]["x"],
        }
