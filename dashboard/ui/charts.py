"""
Charts for the tract panel and comparison view (Altair / Vega-Lite).

One chart library on the page, as the Streamlit guidance recommends. Each
chart answers one question:
    components_chart     which parts of the index make this tract burdened?
    trend_chart          has it been this bad before, and for how long?
    contributions_chart  why does the model expect it to turn CRITICAL?
Validation page:
    equity_chart         who lives in CRITICAL tracts, relative to everyone?
    weekly_chart         is a pipeline input (Kriging error, LST coverage) unusual?
    confidence_mix_chart how sure is each week's CRITICAL list?
    pr_auc_chart         does the model rank entrants better than simple rules?
    reliability_chart    when the model says 30%, do 30% turn CRITICAL?
    recall_chart         does the model miss entrants in some kinds of tract?
Colors follow the map: RdBu for components (red = worse than the county),
class cut-offs drawn as rules on the percentile trend.
"""

from __future__ import annotations

import altair as alt
import pandas as pd

from ui.layers import BLUES, LAYERS, RDBU, bin_index

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


# ------------------------------------------------------------ validation page


def _week_rule(week: str) -> alt.Chart:
    """Dashed vertical rule at the week picked in the sidebar."""
    return (
        alt.Chart(pd.DataFrame({"week_start_date": [pd.Timestamp(week)]}))
        .mark_rule(strokeDash=[3, 3], opacity=0.6)
        .encode(x="week_start_date:T")
    )


def equity_chart(groups: pd.DataFrame) -> alt.Chart:
    """Share of each group living in CRITICAL tracts, as a ratio to all residents."""
    df = groups[groups["group"] != "all"].copy()
    df["side"] = df["ratio_to_all"].map(
        lambda r: "More often than all residents" if r > 1 else "Less often"
    )
    top = max(2.0, float(df["ratio_to_all"].max()) * 1.05)
    bars = (
        alt.Chart(df)
        .mark_bar(height=14)
        .encode(
            y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X(
                "ratio_to_all:Q",
                title="Ratio to all residents",
                scale=alt.Scale(domain=[0, top]),
            ),
            color=alt.Color(
                "side:N",
                scale=alt.Scale(
                    domain=["More often than all residents", "Less often"],
                    range=[RDBU[4], RDBU[0]],
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Group"),
                alt.Tooltip("ratio_to_all:Q", title="Ratio", format=".2f"),
                alt.Tooltip(
                    "share_in_critical:Q", title="Share in CRITICAL", format=".1%"
                ),
                alt.Tooltip("in_critical:Q", title="In CRITICAL tracts", format=",.0f"),
                alt.Tooltip("unit:N", title="Counted as"),
            ],
        )
    )
    one = alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(opacity=0.6).encode(x="x:Q")
    return (bars + one).properties(height=alt.Step(24))


def weekly_chart(
    weekly: pd.DataFrame,
    field: str,
    title: str,
    week: str,
    fmt: str,
    domain: tuple[float, float] | None = None,
    extra_tooltip: list[alt.Tooltip] | None = None,
) -> alt.Chart:
    """One pipeline-health series over every week, the selected week marked."""
    y = alt.Y(f"{field}:Q", title=title)
    if domain:
        y = alt.Y(f"{field}:Q", title=title, scale=alt.Scale(domain=list(domain)))
    line = (
        alt.Chart(weekly)
        .mark_line(strokeWidth=1.5, point=alt.OverlayMarkDef(size=12))
        .encode(
            x=alt.X("week_start_date:T", title=None),
            y=y,
            tooltip=[
                alt.Tooltip("week_start_date:T", title="Week of", format="%b %-d, %Y"),
                alt.Tooltip(f"{field}:Q", title=title, format=fmt),
                *(extra_tooltip or []),
            ],
        )
    )
    return (line + _week_rule(week)).properties(height=200)


def confidence_mix_chart(weekly: pd.DataFrame, week: str) -> alt.Chart:
    """CRITICAL tracts each week, split by how often the 200 draws agree."""
    order = ["HIGH", "MEDIUM", "LOW"]
    df = weekly.melt(
        id_vars="week_start_date",
        value_vars=["critical_high", "critical_medium", "critical_low"],
        var_name="confidence",
        value_name="tracts",
    )
    df["confidence"] = df["confidence"].str.removeprefix("critical_").str.upper()
    df["order"] = df["confidence"].map({c: i for i, c in enumerate(order)})
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("week_start_date:T", title=None),
            y=alt.Y("tracts:Q", title="CRITICAL tracts", stack="zero"),
            color=alt.Color(
                "confidence:N",
                # Same ramp as the map's confidence layer: low agreement is dark
                scale=alt.Scale(domain=order, range=[BLUES[2], BLUES[1], BLUES[0]]),
                sort=order,
                legend=alt.Legend(orient="bottom", title="Confidence"),
            ),
            order=alt.Order("order:Q"),
            tooltip=[
                alt.Tooltip("week_start_date:T", title="Week of", format="%b %-d, %Y"),
                alt.Tooltip("confidence:N", title="Confidence"),
                alt.Tooltip("tracts:Q", title="Tracts"),
            ],
        )
    )
    return (bars + _week_rule(week)).properties(height=240)


def pr_auc_chart(scorers: pd.DataFrame, base_rate: float) -> alt.Chart:
    """Pooled PR-AUC per scorer; baselines faint, the rule is a random ranking."""
    bars = (
        alt.Chart(scorers)
        .mark_bar(height=14)
        .encode(
            y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
            x=alt.X(
                "pr_auc:Q",
                title="PR-AUC (higher is better)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            opacity=alt.Opacity(
                "kind:N",
                scale=alt.Scale(domain=["baseline", "model"], range=[0.35, 1.0]),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=[
                alt.Tooltip("label:N", title="Scorer"),
                alt.Tooltip("pr_auc:Q", title="PR-AUC", format=".3f"),
                alt.Tooltip("roc_auc:Q", title="ROC-AUC", format=".3f"),
                alt.Tooltip(
                    "precision_at_k:Q", title="Precision in weekly top 50", format=".0%"
                ),
                alt.Tooltip("fold_pr_auc_sd:Q", title="SD across folds", format=".3f"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"x": [base_rate]}))
        .mark_rule(strokeDash=[3, 3], opacity=0.6)
        .encode(x="x:Q")
    )
    return (bars + rule).properties(height=alt.Step(24))


def reliability_chart(bins: pd.DataFrame) -> alt.Chart:
    """Mean predicted vs observed rate per probability decile; the diagonal is perfect."""
    top = float(max(bins["predicted"].max(), bins["observed"].max())) * 1.05
    diag = (
        alt.Chart(pd.DataFrame({"a": [0, top], "b": [0, top]}))
        .mark_line(strokeDash=[3, 3], opacity=0.5)
        .encode(x="a:Q", y="b:Q")
    )
    pts = (
        alt.Chart(bins)
        .mark_line(point=True, strokeWidth=1.5)
        .encode(
            x=alt.X(
                "predicted:Q",
                title="Mean predicted probability",
                scale=alt.Scale(domain=[0, top]),
                axis=alt.Axis(format="%"),
            ),
            y=alt.Y(
                "observed:Q",
                title="Share that turned CRITICAL",
                scale=alt.Scale(domain=[0, top]),
                axis=alt.Axis(format="%"),
            ),
            tooltip=[
                alt.Tooltip("bin:Q", title="Decile"),
                alt.Tooltip("predicted:Q", title="Predicted", format=".1%"),
                alt.Tooltip("observed:Q", title="Observed", format=".1%"),
                alt.Tooltip("n:Q", title="Tract-weeks", format=","),
            ],
        )
    )
    return (diag + pts).properties(height=260)


def recall_chart(equity: pd.DataFrame) -> alt.Chart:
    """Top-k recall per tercile of each tract attribute."""
    order = ["low", "mid", "high"]
    names = {"low": "Lowest third", "mid": "Middle third", "high": "Highest third"}
    df = equity.assign(third=equity["tercile"].map(names))
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            y=alt.Y(
                "attribute_label:N",
                sort=None,
                title=None,
                axis=alt.Axis(labelLimit=220),
            ),
            yOffset=alt.YOffset("third:N", sort=[names[t] for t in order]),
            x=alt.X(
                "topk_recall:Q",
                title="Share of entrants caught in the weekly top 50",
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color(
                "third:N",
                sort=[names[t] for t in order],
                scale=alt.Scale(
                    domain=[names[t] for t in order],
                    range=[BLUES[2], BLUES[1], BLUES[0]],
                ),
                legend=alt.Legend(orient="bottom", title="Tracts by the attribute"),
            ),
            tooltip=[
                alt.Tooltip("attribute_label:N", title="Attribute"),
                alt.Tooltip("third:N", title="Tracts"),
                alt.Tooltip("topk_recall:Q", title="Recall", format=".0%"),
                alt.Tooltip(
                    "base_rate:Q", title="Share that turned CRITICAL", format=".1%"
                ),
                alt.Tooltip("pr_auc:Q", title="PR-AUC", format=".2f"),
                alt.Tooltip(
                    "calibration_gap:Q", title="Predicted minus observed", format="+.3f"
                ),
                alt.Tooltip("tracts:Q", title="Tracts"),
            ],
        )
        .properties(height=alt.Step(14))
    )
