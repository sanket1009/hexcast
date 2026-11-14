"""Number and label formatting: precision matches the data ("1.1M", not "1,103,442.7")."""

from __future__ import annotations

from datetime import date

import pandas as pd

DRIVER_LABELS = {
    "heat": "Heat",
    "air_quality": "Air quality",
    "healthcare_access": "Healthcare access",
    "vulnerability": "Population vulnerability",
}
DRIVER_SHORT = {
    "heat": "Heat",
    "air_quality": "Air",
    "healthcare_access": "Access",
    "vulnerability": "Vulnerability",
}
CLASS_ORDER = ["CRITICAL", "HIGH", "ELEVATED", "LOW"]
# st.badge colors for the burden classes (data, not decoration)
CLASS_BADGE = {"CRITICAL": "red", "HIGH": "orange", "ELEVATED": "yellow", "LOW": "gray"}
CONFIDENCE_ORDER = ["HIGH", "MEDIUM", "LOW"]


def people(n: float | None) -> str:
    if n is None or pd.isna(n):
        return "n/a"
    n = float(n)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 10_000:
        return f"{round(n / 1000):,}K"
    if abs(n) >= 1_000:
        return f"{round(n, -2):,.0f}"
    return f"{n:,.0f}"


def signed_people(n: float) -> str:
    return ("+" if n > 0 else "-" if n < 0 else "") + people(abs(n))


def week_label(day: str | date) -> str:
    d = pd.Timestamp(day)
    return f"{d:%b} {d.day}, {d:%Y}"


def short_week(day: str | date) -> str:
    d = pd.Timestamp(day)
    return f"{d:%b} {d.day}"


def place(row: pd.Series | dict) -> str:
    hood, city = row.get("neighborhood"), row.get("city")
    if hood and city and hood != city:
        return f"{hood}, {city}"
    return city or hood or ""


def tract_label(row: pd.Series | dict) -> str:
    """'Tract 2962.20, San Pedro' for pickers and legends."""
    name = (row.get("name") or row["geoid"]).replace("Census Tract ", "Tract ")
    where = row.get("neighborhood") or row.get("city")
    return f"{name}, {where}" if where else name


def count(n: int, noun: str) -> str:
    """'1 tract', '197 tracts'."""
    return f"{n:,} {noun}{'' if n == 1 else 's'}"
