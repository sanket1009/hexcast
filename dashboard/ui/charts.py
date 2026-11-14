"""
Charts for the tract panel and comparison view (Altair / Vega-Lite).

One chart library on the page, as the Streamlit guidance recommends. Each
chart answers one question:
    components_chart     which parts of the index make this tract burdened?
    trend_chart          has it been this bad before, and for how long?
    contributions_chart  why does the model expect it to turn CRITICAL?
Colors follow the map: RdBu for components (red = worse than the county),
class cut-offs drawn as rules on the percentile trend.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from ui.layers import LAYERS, RDBU, bin_index

COMPONENT_ORDER = ["heat", "air_quality", "healthcare_access", "vulnerability"]
# One tract per legend row: names like "Tract 2948.20, Wilmington" need the width
LEGEND = alt.Legend(orient="bottom", direction="vertical", labelLimit=320)
CUTOFFS = pd.DataFrame(
    {"percentile": [50, 75, 90], "label": ["ELEVATED", "HIGH", "CRITICAL"]}
)


def _component_rows(tracts: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    rows = []
    for _, t in tracts.iterrows():
        for key in COMPONENT_ORDER:
            layer = LAYERS[key]
            rows.append(
                {
                    "tract": labels.get(t["geoid"], t["geoid"]),
                    "component": layer.label,
                    "z": t[layer.column],
                }
            )
    df = pd.DataFrame(rows)
    idx = bin_index(df["z"], LAYERS["heat"])
    df["color"] = [RDBU[i] if i >= 0 else "#BDBDBD" for i in idx]
    return df


def components_chart(tracts: pd.DataFrame, labels: dict[str, str]) -> alt.Chart:
    """Signed component z-scores; several tracts are grouped side by side."""
    df = _component_rows(tracts, labels)
    order = [LAYERS[k].label for k in COMPONENT_ORDER]
    base = alt.Chart(df).encode(
        y=alt.Y("component:N", sort=order, title=None, axis=alt.Axis(labelLimit=200)),
        x=alt.X("z:Q", title="SD vs county average (right = worse)"),
        tooltip=[
            alt.Tooltip("tract:N", title="Tract"),
            alt.Tooltip("component:N", title="Component"),
            alt.Tooltip("z:Q", title="SD", format="+.2f"),
        ],
    )
    if df["tract"].nunique() == 1:
        bars = base.mark_bar(height=16).encode(color=alt.Color("color:N", scale=None))
    else:
        bars = base.mark_bar().encode(
            yOffset=alt.YOffset("tract:N"),
            color=alt.Color("tract:N", title=None, legend=LEGEND),
        )
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(opacity=0.5).encode(x="z:Q")
    # Streamlit fits the whole chart (axes and legend included) into height,
    # so the legend's rows are added on top of the plot area
    n = df["tract"].nunique()
    plot = 4 * (30 if n == 1 else 12 * n + 14)
    return (bars + zero).properties(height=plot + 50 + (20 * n + 10 if n > 1 else 0))


def trend_chart(trends: pd.DataFrame, labels: dict[str, str]) -> alt.Chart:
    """Weekly burden percentile with the class cut-offs."""
    df = trends.assign(tract=trends["geoid"].map(labels).fillna(trends["geoid"]))
    many = df["tract"].nunique() > 1
    axis = alt.Axis(values=[0, 25, 50, 75, 90, 100], grid=False)
    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=1.5)
        .encode(
            x=alt.X("week_start_date:T", title=None),
            y=alt.Y(
                "burden_percentile:Q",
                title=None,
                scale=alt.Scale(domain=[0, 100]),
                axis=axis,
            ),
            tooltip=[
                alt.Tooltip("tract:N", title="Tract"),
                alt.Tooltip("week_start_date:T", title="Week of", format="%b %-d, %Y"),
                alt.Tooltip("burden_class:N", title="Class"),
                alt.Tooltip("burden_percentile:Q", title="Percentile", format=".1f"),
            ],
        )
    )
    if many:  # one tract keeps the theme's first color (the text color)
        line = line.encode(color=alt.Color("tract:N", title=None, legend=LEGEND))
    rules = (
        alt.Chart(CUTOFFS)
        .mark_rule(strokeDash=[3, 3], opacity=0.45)
        .encode(y="percentile:Q")
    )
    n = df["tract"].nunique()
    return (rules + line).properties(height=230 + (20 * n + 10 if many else 0))


def contributions_chart(items: list[dict], other: float, n_other: int) -> alt.Chart:
    """Log-odds terms: right pushes the probability up, left pulls it down."""
    df = pd.DataFrame(
        [{"label": i["label"], "contribution": i["contribution"]} for i in items]
        + (
            [{"label": f"{n_other} other features", "contribution": other}]
            if n_other
            else []
        )
    )
    df["direction"] = df["contribution"].map(
        lambda c: "raises risk" if c > 0 else "lowers risk"
    )
    return (
        alt.Chart(df)
        .mark_bar(height=14)
        .encode(
            y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=220)),
            x=alt.X("contribution:Q", title="Effect on log-odds of turning CRITICAL"),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["raises risk", "lowers risk"], range=[RDBU[4], RDBU[0]]
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Feature"),
                alt.Tooltip("contribution:Q", title="Log-odds", format="+.2f"),
            ],
        )
        .properties(height=alt.Step(24))
    )
