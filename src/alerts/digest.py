"""
digest.py
=========
The weekly burden report for one city: what changed since last week, where
burden stays high, and what the policy briefings recommend.

Structure (read top to bottom in under a minute):
    Summary          three to five sentences on what changed
    What changed     communities with tracts that became critical, left
                     critical, or jumped in rank, by reporting area
    Where it stays high  one line per region (SPA in LA)
    Recommended actions  the guidance most often cited in this week's
                     briefings, when briefings exist
    About this report    which week the data describes and why it lags

Tracts are named by community and region (see ingestion/reporting_areas.py)
and counted by residents, not polygons. Drivers come from
risk_inputs.build_tract_risks, the same rule the briefings use, so the
report needs no model call. Rank is by raw burden (1 = highest), so ties
only happen on identical scores.

This is a report on conditions for a past week (EPA AQS lags by months),
not a real-time warning.

Usage:
    from src.alerts.digest import build_digest, render_html, render_subject, write_csv
    digest = build_digest(config, "2026-06-22")
"""

from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.rag.briefings_table import BRIEFINGS_TABLE
from src.rag.risk_inputs import build_tract_risks
from src.utils.bigquery import read_query, table_exists
from src.utils.logger import get_logger

logger = get_logger(__name__)

DRIVER_LABELS = {
    "heat": "heat",
    "air_quality": "air quality",
    "healthcare_access": "healthcare access",
}

LST = "Land surface temperature, mean (°C)"
PM25 = "PM2.5, Kriged annual mean (µg/m³)"
BEDS = "Hospital beds per 1,000 residents, 30-minute drive"
ER = "Drive time to nearest ER (minutes)"

MOVER_RANKS = (
    100  # a critical tract that rose at least this many ranks is listed as a mover
)
TOP_ACTIONS = 5
UNASSIGNED = "Unassigned area"

ACRONYMS = {"la": "LA", "ca": "CA", "carb": "CARB", "cdph": "CDPH", "pm2": "PM2"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TractRow:
    geoid: str
    rank: int
    prev_rank: int | None
    burden_class: str
    prev_class: str | None
    population: int
    region: str
    community: str
    drivers: list[str] = field(default_factory=list)
    indicators: dict[str, float] = field(default_factory=dict)
    summary: str | None = None

    @property
    def is_critical(self) -> bool:
        return self.burden_class == "CRITICAL"

    @property
    def was_critical(self) -> bool | None:
        return None if self.prev_class is None else self.prev_class == "CRITICAL"

    @property
    def rank_change(self) -> int | None:
        """Positive = moved toward rank 1 (more burden)."""
        return None if self.prev_rank is None else self.prev_rank - self.rank


@dataclass
class ActionGroup:
    citation: str
    action: str
    tracts: int


@dataclass
class Digest:
    city_name: str
    week: str
    tract_count: int
    class_counts: dict[str, int]
    rows: list[TractRow]  # every tract this week
    has_previous_week: bool
    has_briefings: bool
    actions: list[ActionGroup] = field(default_factory=list)

    @property
    def critical(self) -> list[TractRow]:
        return sorted((r for r in self.rows if r.is_critical), key=lambda r: r.rank)

    @property
    def newly_critical(self) -> list[TractRow]:
        return [r for r in self.critical if r.was_critical is False]

    @property
    def left_critical(self) -> list[TractRow]:
        return sorted(
            (r for r in self.rows if r.was_critical and not r.is_critical),
            key=lambda r: r.rank,
        )

    @property
    def movers(self) -> list[TractRow]:
        return [
            r
            for r in self.critical
            if r.was_critical
            and r.rank_change is not None
            and r.rank_change >= MOVER_RANKS
        ]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_digest(config: dict[str, Any], week_str: str) -> Digest:
    """Collect one week, the week before, names, residents, and briefings."""
    fips = config["full_fips"]
    this_week = _scores(config, week_str)
    if this_week.empty:
        raise ValueError(f"No compound scores for week {week_str}")
    last_week = _scores(config, _previous_week(week_str))
    places = _places(config)
    risks = {
        r.geoid: r for r in build_tract_risks(config, week_str)
    }  # HIGH and CRITICAL only
    briefings = _briefings(config, week_str)

    prev = last_week.set_index("geoid") if not last_week.empty else None
    rows = []
    for s in this_week.itertuples():
        place = places.get(s.geoid, {})
        risk = risks.get(s.geoid)
        rows.append(
            TractRow(
                geoid=s.geoid,
                rank=int(s.rank),
                prev_rank=int(prev.at[s.geoid, "rank"])
                if prev is not None and s.geoid in prev.index
                else None,
                burden_class=s.burden_class,
                prev_class=prev.at[s.geoid, "burden_class"]
                if prev is not None and s.geoid in prev.index
                else None,
                population=int(place.get("population") or 0),
                region=place.get("region") or UNASSIGNED,
                community=place.get("community") or UNASSIGNED,
                drivers=list(risk.drivers) if risk else [],
                indicators=dict(risk.indicators) if risk else {},
                summary=briefings.get(s.geoid, {}).get("summary"),
            )
        )

    critical_ids = {r.geoid for r in rows if r.is_critical}
    digest = Digest(
        city_name=config["city_name"],
        week=week_str,
        tract_count=len(rows),
        class_counts=this_week["burden_class"].value_counts().to_dict(),
        rows=rows,
        has_previous_week=prev is not None,
        has_briefings=bool(briefings),
        actions=group_actions(
            {g: b["actions"] for g, b in briefings.items() if g in critical_ids}
        ),
    )
    logger.info(
        f"Report for {week_str} ({fips}): {len(digest.critical)} critical, "
        f"{len(digest.newly_critical)} new, {len(digest.left_critical)} left, "
        f"previous week: {digest.has_previous_week}, briefings: {digest.has_briefings}"
    )
    return digest


def group_actions(actions_by_tract: dict[str, list[dict]]) -> list[ActionGroup]:
    """
    The guidance cited for the most critical tracts. Actions are reworded per
    tract, so they are grouped by their first citation; the shortest wording
    stands for the group.
    """
    tracts: dict[str, set[str]] = defaultdict(set)
    wordings: dict[str, list[str]] = defaultdict(list)
    for geoid, actions in actions_by_tract.items():
        for a in actions:
            if not a.get("citations"):
                continue
            key = a["citations"][0]
            tracts[key].add(geoid)
            wordings[key].append(a["action"])
    groups = [
        ActionGroup(citation=k, action=min(wordings[k], key=len), tracts=len(v))
        for k, v in tracts.items()
    ]
    return sorted(groups, key=lambda g: (-g.tracts, g.citation))[:TOP_ACTIONS]


def _previous_week(week_str: str) -> str:
    return (date.fromisoformat(week_str) - timedelta(weeks=1)).isoformat()


def _scores(config: dict[str, Any], week_str: str) -> pd.DataFrame:
    df = read_query(
        f"""
        SELECT geoid, burden_class, raw_burden
        FROM {{dataset}}.compound_scores
        WHERE week_start_date = '{week_str}' AND city_fips = '{config["full_fips"]}'
        """,
        config,
    )
    if not df.empty:
        df["rank"] = df["raw_burden"].rank(ascending=False, method="min").astype(int)
    return df


def _places(config: dict[str, Any]) -> dict[str, dict]:
    """geoid -> population, region, community. Names are empty if the crosswalk is missing."""
    fips = config["full_fips"]
    pop = read_query(
        f"SELECT geoid, population FROM {{dataset}}.tract_geometries WHERE city_fips = '{fips}'",
        config,
    )
    out = {
        g: {"population": p}
        for g, p in zip(pop["geoid"], pop["population"], strict=True)
    }
    if table_exists("tract_reporting_areas", config):
        areas = read_query(
            f"""
            SELECT geoid, region_name, community_name
            FROM {{dataset}}.tract_reporting_areas WHERE city_fips = '{fips}'
            """,
            config,
        )
        for g, region, community in zip(
            areas["geoid"], areas["region_name"], areas["community_name"], strict=True
        ):
            out.setdefault(g, {}).update(region=region, community=community)
    else:
        logger.warning("tract_reporting_areas missing; the report will not name places")
    return out


def _briefings(config: dict[str, Any], week_str: str) -> dict[str, dict]:
    if not table_exists(BRIEFINGS_TABLE, config):
        return {}
    df = read_query(
        f"""
        SELECT geoid, summary, actions_json
        FROM {{dataset}}.{BRIEFINGS_TABLE}
        WHERE week_start_date = '{week_str}' AND city_fips = '{config["full_fips"]}'
        """,
        config,
    )
    return {
        g: {"summary": s, "actions": json.loads(a)}
        for g, s, a in zip(df["geoid"], df["summary"], df["actions_json"], strict=True)
    }


# ---------------------------------------------------------------------------
# Summaries used by the renderers
# ---------------------------------------------------------------------------


def by_community(rows: list[TractRow]) -> list[dict]:
    """Group tracts by (region, community): tracts, residents, drivers, key numbers."""
    groups: dict[tuple[str, str], list[TractRow]] = defaultdict(list)
    for r in rows:
        groups[(r.region, r.community)].append(r)
    out = []
    for (region, community), members in groups.items():
        drivers = Counter(d for m in members for d in m.drivers)
        out.append(
            {
                "region": region,
                "community": community,
                "tracts": len(members),
                "residents": sum(m.population for m in members),
                "drivers": [d for d, _ in drivers.most_common()],
                "key_numbers": key_numbers(members),
                "best_rank": min(m.rank for m in members),
            }
        )
    return sorted(out, key=lambda g: (-g["residents"], g["best_rank"]))


def by_region(rows: list[TractRow]) -> list[dict]:
    """One line per region: critical tracts, residents, driver mix, top communities."""
    groups: dict[str, list[TractRow]] = defaultdict(list)
    for r in rows:
        groups[r.region].append(r)
    out = []
    for region, members in groups.items():
        n = len(members)
        drivers = Counter(d for m in members for d in m.drivers)
        communities = Counter(m.community for m in members)
        out.append(
            {
                "region": region,
                "tracts": n,
                "residents": sum(m.population for m in members),
                "driver_share": {d: c / n for d, c in drivers.most_common()},
                "top_communities": [c for c, _ in communities.most_common(3)],
            }
        )
    return sorted(out, key=lambda g: -g["residents"])


def key_numbers(members: list[TractRow]) -> str:
    """The worst value of each driver's main number across the tracts."""
    parts = []
    drivers = {d for m in members for d in m.drivers}

    def values(label):
        return [m.indicators[label] for m in members if label in m.indicators]

    if "heat" in drivers and values(LST):
        parts.append(f"LST up to {max(values(LST)):.1f} °C")
    if "air_quality" in drivers and values(PM25):
        parts.append(f"PM2.5 up to {max(values(PM25)):.1f} µg/m³")
    if "healthcare_access" in drivers and values(BEDS):
        parts.append(f"as few as {min(values(BEDS)):.2f} beds per 1,000")
    return "; ".join(parts)


def summary_lines(d: Digest) -> list[str]:
    crit = d.critical
    lines = []
    if d.has_previous_week:
        new, left = d.newly_critical, d.left_critical
        if new:
            top = by_community(new)[0]
            lines.append(
                f"{_n(len(new), 'tract')} became critical this week ({_people(new)}), "
                f"the most in {top['community']} ({top['region']}), mainly {_drivers(top['drivers'][:2])}."
            )
        else:
            lines.append("No tracts became critical this week.")
        lines.append(
            f"{_n(len(left), 'tract')} left critical status ({_people(left)})."
            if left
            else "No tracts left critical status."
        )
        if d.movers:
            lines.append(
                f"{_n(len(d.movers), 'critical tract')} rose {MOVER_RANKS} or more places in rank."
            )
    else:
        lines.append(
            "First week on record, so there is nothing to compare against yet."
        )

    regions = by_region(crit)
    if regions:
        r = regions[0]
        lines.append(
            f"{_n(len(crit), 'tract')} are critical in total ({_people(crit)}). "
            f"The most are in {r['region']} ({_n(r['tracts'], 'tract')}, {r['residents']:,} residents)."
        )
    return lines


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render_subject(d: Digest) -> str:
    base = f"CHBIndex {d.city_name} weekly burden report, conditions week of {d.week}"
    if not d.has_previous_week:
        return f"{base}: first week on record"
    new = len(d.newly_critical)
    if new == 0 and not d.left_critical:
        return f"{base}: no change in critical areas"
    return f"{base}: {_n(new, 'tract')} newly critical, {len(d.left_critical)} left"


def render_html(d: Digest, top_n: int = 5) -> str:
    """Email body. top_n caps each change type in What changed; the CSV has everything."""
    out = [_STYLE_OPEN]
    out.append(
        f'<h2 style="margin:0 0 4px">{_e(d.city_name)} weekly burden report</h2>'
    )
    out.append(
        f'<p style="margin:0 0 12px;color:#6b6a66">Conditions for the week of {d.week}. '
        "Retrospective report, not a real-time warning (see About this report).</p>"
    )

    out.append(_h("Summary"))
    out.append(
        "<ul>" + "".join(f"<li>{_e(s)}</li>" for s in summary_lines(d)) + "</ul>"
    )

    if d.has_previous_week:
        out.append(_h("What changed"))
        sections = [
            ("Newly critical", by_community(d.newly_critical)),
            ("Rose in rank", by_community(d.movers)),
            ("Left critical", by_community(d.left_critical)),
        ]
        shown = [(label, g) for label, groups in sections for g in groups[:top_n]]
        hidden = sum(max(len(groups) - top_n, 0) for _, groups in sections)
        if shown:
            out.append(
                _table(
                    [
                        "Change",
                        "Community",
                        "Region",
                        "Tracts",
                        "Residents",
                        "Drivers",
                        "Key numbers",
                    ],
                    [
                        [
                            c,
                            g["community"],
                            g["region"],
                            g["tracts"],
                            f"{g['residents']:,}",
                            _drivers(g["drivers"]),
                            g["key_numbers"],
                        ]
                        for c, g in shown
                    ],
                )
            )
            if hidden:
                out.append(
                    f"<p>Largest {top_n} communities per change shown, by residents; "
                    f"{hidden} more in the attached CSV.</p>"
                )
        else:
            out.append("<p>No changes in critical status this week.</p>")

    out.append(_h("Where burden stays high"))
    regions = by_region(d.critical)
    out.append(
        _table(
            [
                "Region",
                "Critical tracts",
                "Residents",
                "Drivers (share of tracts)",
                "Most affected communities",
            ],
            [
                [
                    g["region"],
                    g["tracts"],
                    f"{g['residents']:,}",
                    ", ".join(
                        f"{DRIVER_LABELS.get(k, k)} {v:.0%}"
                        for k, v in g["driver_share"].items()
                    ),
                    ", ".join(g["top_communities"]),
                ]
                for g in regions
            ],
        )
    )

    out.append(_h("Recommended actions"))
    if d.actions:
        out.append(
            "<p>Guidance cited most often in this week's briefings for critical tracts:</p><ol>"
        )
        for a in d.actions:
            out.append(
                f'<li>{_e(a.action)} <span style="color:#6b6a66">({_e(source_title(a.citation))}; '
                f"{_n(a.tracts, 'critical tract')})</span></li>"
            )
        out.append("</ol>")
    else:
        out.append(
            "<p>No briefings were generated for this week. Briefings run on request.</p>"
        )

    out.append(_h("About this report"))
    out.append(
        '<p style="color:#6b6a66;font-size:12px">'
        f"Critical means the top 10% of {d.tract_count:,} tracts by compound burden "
        "(heat, PM2.5, and access to emergency care, with better access lowering burden), "
        "so about the same number are critical every week; the changes are the signal. "
        "PM2.5 is interpolated from EPA AQS monitors, which publish months late. Heat is "
        "land surface temperature from a 32-day Landsat composite, not air temperature. "
        "Access is hospital beds within a 30-minute free-flow drive. Communities are LA County "
        "Countywide Statistical Areas and regions are Service Planning Areas; each tract is "
        "assigned to the area it overlaps most. The attached CSV lists every critical tract."
        "</p></div>"
    )
    return "\n".join(out)


def write_csv(d: Digest, path: Path) -> Path:
    """Every critical tract plus every tract that left critical, one row each."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = d.critical + d.left_critical
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "geoid",
                "community",
                "region",
                "residents",
                "rank",
                "previous_rank",
                "burden_class",
                "previous_class",
                "change",
                "drivers",
                "lst_c",
                "pm25_ugm3",
                "beds_per_1000",
                "er_drive_min",
                "briefing_summary",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.geoid,
                    r.community,
                    r.region,
                    r.population,
                    r.rank,
                    _blank(r.prev_rank),
                    r.burden_class,
                    _blank(r.prev_class),
                    _change(r, d),
                    _drivers(r.drivers),
                    *(_blank(r.indicators.get(k)) for k in (LST, PM25, BEDS, ER)),
                    r.summary or "",
                ]
            )
    return path


def source_title(citation: str) -> str:
    """'la_county_heat_action_plan_2026.pdf, p. 10' -> 'LA County Heat Action Plan 2026, p. 10'."""
    name, _, page = citation.partition(", ")
    words = name.removesuffix(".pdf").split("_")
    title = " ".join(ACRONYMS.get(w, w.capitalize()) for w in words)
    return f"{title}, {page}" if page else title


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STYLE_OPEN = '<div style="font-family:Arial,sans-serif;font-size:14px;color:#1f1f1e;max-width:860px">'


def _h(text: str) -> str:
    return f'<h3 style="margin:20px 0 6px;border-bottom:1px solid #ddd">{_e(text)}</h3>'


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    th = "".join(
        f'<th style="text-align:left;padding:4px 8px;border-bottom:1px solid #999">{_e(h)}</th>'
        for h in headers
    )
    trs = "".join(
        "<tr>"
        + "".join(
            f'<td style="padding:4px 8px;border-bottom:1px solid #eee;vertical-align:top">'
            f"{_e(c)}</td>"
            for c in row
        )
        + "</tr>"
        for row in rows
    )
    return f'<table style="border-collapse:collapse"><tr>{th}</tr>{trs}</table>'


def _change(r: TractRow, d: Digest) -> str:
    if not d.has_previous_week:
        return ""
    if r.is_critical and r.was_critical is False:
        return "newly critical"
    if r.was_critical and not r.is_critical:
        return "left critical"
    if r in d.movers:
        return f"rose {r.rank_change} ranks"
    return "still critical"


def _drivers(drivers: list[str]) -> str:
    labels = [DRIVER_LABELS.get(x, x) for x in drivers]
    if len(labels) <= 1:
        return "".join(labels)
    return ", ".join(labels[:-1]) + " and " + labels[-1]


def _people(rows: list[TractRow]) -> str:
    return f"{sum(r.population for r in rows):,} residents"


def _n(count: int, noun: str) -> str:
    return f"{count:,} {noun}" + ("" if count == 1 else "s")


def _blank(value: Any) -> Any:
    return "" if value is None else value


def _e(text: Any) -> str:
    return html.escape(str(text))
