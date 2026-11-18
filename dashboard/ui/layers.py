"""
Map layers: which column each layer colors by, its bins, colors and legend.

Rules from the Phase 7 spec:
    Darker = worse on every layer. Burden, ML risk and chronic weeks use OrRd
    (faintest step dropped). The four components are signed z-scores where a
    higher value is worse (access_z already carries the minus sign), drawn
    with diverging RdBu centered at 0. Confidence: low agreement is dark.
    No red-green, no teal, cyan or violet.

Bins are fixed cut points, not quantiles of the week, so a color means the
same thing in every week.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

ORRD = ["#FDD49E", "#FDBB84", "#FC8D59", "#E34A33", "#B30000"]
RDBU = ["#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B"]
BLUES = ["#08519C", "#6BAED6", "#C6DBEF"]
Z_LABELS = [
    "1+ SD better than county",
    "0.25 to 1 SD better",
    "Near county average",
    "0.25 to 1 SD worse",
    "1+ SD worse than county",
]


@dataclass(frozen=True)
class Layer:
    key: str
    label: str  # shown in the "Color by" picker
    column: str
    bins: tuple[float, ...]  # right-open cut points: len(colors) - 1
    colors: tuple[str, ...]
    labels: tuple[str, ...]  # legend text, one per color
    missing: str  # legend text for tracts without a value
    fmt: str  # tooltip format for the raw value
    note: str  # one line under the legend


LAYERS: dict[str, Layer] = {
    layer.key: layer
    for layer in [
        Layer(
            "burden", "Burden", "burden_percentile", (25, 50, 75, 90), tuple(ORRD),
            ("Under 25th percentile", "25th to 50th", "50th to 75th: ELEVATED",
             "75th to 90th: HIGH", "90th and over: CRITICAL"),
            "Not scored", "{:.1f} percentile",
            "County percentile of the compound burden score this week.",
        ),
        Layer(
            "heat", "Heat", "lst_z", (-1, -0.25, 0.25, 1), tuple(RDBU), tuple(Z_LABELS),
            "No clear satellite pixels", "{:+.2f} SD",
            "Land surface temperature, 32-day Landsat composite, relative to the county.",
        ),
        Layer(
            "air_quality", "Air quality", "pm25_z", (-1, -0.25, 0.25, 1), tuple(RDBU),
            tuple(Z_LABELS), "Not scored", "{:+.2f} SD",
            "PM2.5, 4-week mean of Kriged monitor readings, relative to the county.",
        ),
        Layer(
            "healthcare_access", "Healthcare access", "access_z", (-1, -0.25, 0.25, 1),
            tuple(RDBU), tuple(Z_LABELS), "No residents or no road link", "{:+.2f} SD",
            "Hospital beds within a 30-minute drive; worse access is red.",
        ),
        Layer(
            "vulnerability", "Population vulnerability", "vulnerability_z",
            (-1, -0.25, 0.25, 1), tuple(RDBU), tuple(Z_LABELS), "Not scored",
            "{:+.2f} SD", "Seven ACS groups: older adults, young children, low income, "
            "no car, limited English, uninsured, disability.",
        ),
        Layer(
            "ml_risk", "Risk of turning CRITICAL", "prob_critical_4w",
            (0.05, 0.2, 0.5, 0.8), tuple(ORRD),
            ("Under 5%", "5% to 20%", "20% to 50%", "50% to 80%", "80% and over"),
            "CRITICAL already", "{:.0%}",
            "Model probability of becoming CRITICAL within 4 weeks.",
        ),
        Layer(
            "chronic", "Chronic burden", "weeks_critical", (1, 13, 26, 39), tuple(ORRD),
            ("Never CRITICAL", "1 to 12 weeks", "13 to 25 weeks",
             "26 to 38 weeks: chronic", "39 to 52 weeks: chronic"),
            "Under 52 weeks of history", "{:.0f} of 52 weeks",
            "Weeks CRITICAL in the last 52. Chronic means 26 or more.",
        ),
        Layer(
            "confidence", "Confidence", "agreement", (0.5, 0.9), tuple(BLUES),
            ("LOW: under 50% agree", "MEDIUM: 50% to 90%", "HIGH: 90% and over"),
            "Not computed", "{:.0%} agree",
            "Share of 200 plausible versions of the index (PM2.5 interpolation error, "
            "component weights) that agree with this week's CRITICAL-or-not call.",
        ),
    ]
}  # fmt: skip


def hex_to_rgb(color: str) -> list[int]:
    color = color.lstrip("#")
    return [int(color[i : i + 2], 16) for i in (0, 2, 4)]


def missing_color(theme: str) -> str:
    return "#3A3A3C" if theme == "dark" else "#D1D1D6"


def bin_index(values: pd.Series, layer: Layer) -> np.ndarray:
    """Index into layer.colors for each value; -1 where the value is missing."""
    v = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    idx = np.digitize(v, layer.bins)  # right-open bins: x < b0 -> 0
    return np.where(np.isnan(v), -1, idx)


def fill_colors(
    values: pd.Series, layer: Layer, theme: str, keep: np.ndarray | None = None
) -> list[list[int]]:
    """RGBA per value. Tracts filtered out (keep False) stay as faint ghosts."""
    palette = np.array(
        [hex_to_rgb(c) for c in layer.colors] + [hex_to_rgb(missing_color(theme))]
    )
    idx = bin_index(values, layer)
    rgb = palette[np.where(idx < 0, len(layer.colors), idx)]
    alpha = np.full(len(rgb), 225)
    if keep is not None:
        alpha = np.where(keep, 225, 28)
    return np.column_stack([rgb, alpha]).astype(int).tolist()


def value_text(value: object, layer: Layer) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return layer.missing
    return layer.fmt.format(value)


def legend_rows(layer: Layer, theme: str) -> list[tuple[str, str]]:
    """(color, text) pairs, worst first, then the missing-value swatch."""
    rows = list(zip(layer.colors, layer.labels, strict=True))
    if layer.key != "confidence":  # confidence bins already run worst (LOW) first
        rows.reverse()
    return rows + [(missing_color(theme), layer.missing)]
