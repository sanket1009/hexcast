"""
Watch list: tracts that are not CRITICAL now, ranked by the model's
probability of turning CRITICAL within 4 weeks, with the reasons the model
gives (its three largest positive log-odds terms).

Filters are bound to the URL like the Overview's. Selected rows open in the
Overview through st.switch_page, which carries them in ?tracts=.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui import api, fmt

week = st.session_state["week"]
compare = (
    None if st.session_state["compare"] == "previous" else st.session_state["compare"]
)

try:
    summ = api.summary(week, compare)
    items = api.watchlist(week, limit=500)
    tracts = api.tracts(week)
except api.ApiError as exc:
    st.error(str(exc), title="Could not load the watch list")
    st.stop()

watch = summ["kpis"]["watchlist"]
if not watch or not items:
    st.header("No forecast for this week")
    st.caption(
        f"The model has no predictions for {fmt.week_label(week)}. "
        "Predictions start once a model has been trained on earlier weeks."
    )
    st.stop()

threshold = watch["threshold"]
scored = int(tracts["prob_critical_4w"].notna().sum())
wl = pd.DataFrame(items)
wl["label"] = [fmt.tract_label(r) for r in items]
wl["place"] = [fmt.place(r) for r in items]
wl["why"] = [
    ", ".join(
        c["label"].removesuffix(" component").removesuffix(" this week")
        for c in (r["ml_contributions"] or {}).get("top", [])
    )
    for r in items
]

st.header(
    f"{fmt.count(watch['tracts'], 'tract')} {'has' if watch['tracts'] == 1 else 'have'} "
    f"a {threshold:.0%} or higher chance "
    "of turning CRITICAL within 4 weeks"
)
st.caption(
    f"About {watch['expected_entering']:.0f} of the {scored:,} tracts that are not CRITICAL "
    f"on {fmt.week_label(week)} are expected to turn CRITICAL within 4 weeks: the sum of every "
    "tract's probability, which is a fair count because the model is calibrated "
    f"(see Validation). Model {watch['model_version']}."
)

# ------------------------------------------------------------------- filters

with st.container(horizontal=True, vertical_alignment="bottom", gap="medium"):
    min_prob = st.slider(
        "Probability at least",
        5,
        95,
        round(threshold * 100),
        step=5,
        format="%d%%",
        key="wl_min",
        bind="query-params",
        width=260,
    )
    cls = st.pills(
        "Class now",
        ["HIGH", "ELEVATED", "LOW"],
        selection_mode="multi",
        key="wl_cls",
        bind="query-params",
    )
    drv = st.pills(
        "Main driver",
        list(fmt.DRIVER_LABELS),
        format_func=fmt.DRIVER_LABELS.get,
        selection_mode="multi",
        key="wl_drv",
        bind="query-params",
    )
    spa = st.multiselect(
        "Service Planning Area",
        sorted(wl["region_name"].dropna().unique()),
        key="wl_spa",
        bind="query-params",
        width=300,
    )

keep = wl["prob_critical_4w"] >= min_prob / 100
if cls:
    keep &= wl["burden_class"].isin(cls)
if drv:
    keep &= wl["main_driver"].isin(drv)
if spa:
    keep &= wl["region_name"].isin(spa)
shown = wl[keep].reset_index(drop=True)

if shown.empty:
    st.info(
        "No tract on the watch list matches these filters. Lower the probability or clear a filter."
    )
    st.stop()

people = shown["population"].sum()
truncated = len(items) == 500 and items[-1]["prob_critical_4w"] >= min_prob / 100
drivers = shown["main_driver"].map(fmt.DRIVER_LABELS).value_counts()
st.markdown(
    f"**{fmt.count(len(shown), 'tract')}, {fmt.people(people)} people.** "
    f"Most common main driver: {drivers.index[0].lower()} ({drivers.iloc[0]} tracts)."
    + (" Only the top 500 by probability are listed." if truncated else "")
)

# --------------------------------------------------------------------- table

table = pd.DataFrame(
    {
        "Rank": shown["ml_rank"],
        "Tract": shown["name"].str.replace("Census Tract ", "", regex=False),
        "Place": shown["place"],
        "People": shown["population"],
        "Probability": shown["prob_critical_4w"] * 100,
        "Class now": shown["burden_class"],
        "Percentile": shown["burden_percentile"],
        "Change": shown["percentile_change"],
        "Main driver": shown["main_driver"].map(fmt.DRIVER_SHORT),
        "Why the model flags it": shown["why"],
        "Weeks CRITICAL": shown["weeks_critical"],
        "Confidence": shown["confidence"],
    }
)
event = st.dataframe(
    table,
    hide_index=True,
    height=min(38 + 35 * len(table), 640),
    key="wl_table",
    # What and why first; the rest scrolls on narrow screens
    column_order=[
        "Rank",
        "Tract",
        "Place",
        "Probability",
        "Why the model flags it",
        "Class now",
        "Percentile",
        "Change",
        "Main driver",
        "People",
        "Weeks CRITICAL",
        "Confidence",
    ],  # fmt: skip
    on_select="rerun",
    selection_mode="multi-row",
    column_config={
        "Rank": st.column_config.NumberColumn(
            width=60, help="Rank among all tracts scored this week"
        ),
        "Tract": st.column_config.TextColumn(width=70),
        "Place": st.column_config.TextColumn(width="medium"),
        "People": st.column_config.NumberColumn(format="localized", width=80),
        "Probability": st.column_config.ProgressColumn(
            min_value=0,
            max_value=100,
            format="%.0f%%",
            width=110,
            help="Calibrated probability of entering CRITICAL within 4 weeks",
        ),
        "Class now": st.column_config.TextColumn(width=80),
        "Percentile": st.column_config.NumberColumn(format="%.1f", width=80),
        "Change": st.column_config.NumberColumn(
            format="%+.1f", width=70, help="Percentile change since the week before"
        ),
        "Why the model flags it": st.column_config.TextColumn(
            width="large",
            help="The three features that raise this tract's log-odds the most (coefficient × scaled value)",
        ),
        "Weeks CRITICAL": st.column_config.NumberColumn(
            width=80, help="Weeks CRITICAL out of the last 52"
        ),
        "Confidence": st.column_config.TextColumn(
            width=90,
            help="How sure the index is that the tract is not CRITICAL this week. "
            "LOW: in many of 200 plausible versions of the index it is CRITICAL.",
        ),
    },
    alt="Watch-list table: tracts ranked by probability of turning CRITICAL within 4 weeks",
)

rows = event.selection.rows
with st.container(horizontal=True, vertical_alignment="center"):
    picked = shown.loc[rows[:20], "geoid"].tolist()
    if st.button(
        f"Open {fmt.count(len(picked), 'tract')} in Overview"
        if picked
        else "Open in Overview",
        disabled=not picked,
        help="Select rows in the table first (up to 20).",
    ):
        st.switch_page(
            "app_pages/overview.py",
            query_params={"tracts": shown.loc[rows[:20], "label"].tolist()},
        )
    export = shown[
        ["ml_rank", "geoid", "name", "city", "neighborhood", "region_name", "population",
         "prob_critical_4w", "burden_class", "burden_percentile", "percentile_change",
         "main_driver", "why", "weeks_critical", "confidence"]
    ].rename(columns={"why": "top_reasons"})  # fmt: skip
    st.download_button(
        "Download this list (CSV)",
        export.to_csv(index=False),
        file_name=f"hexcast_watchlist_{week}.csv",
        mime="text/csv",
        type="tertiary",
        icon=":material/download:",
    )
st.caption(
    "Reasons are the model's arithmetic, not causes: correlated features can share or offset "
    "each other's weight. Tracts already CRITICAL are not forecast."
)
