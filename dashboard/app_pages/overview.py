"""
Overview: headline finding, KPI boxes, the map, and a panel for the selected
tracts. Every filter and the selection are bound to the URL.

Selection has one source of truth, the "Selected tracts" picker (key
"tracts"). Map clicks, table rows, search results and the address lookup
all add to it through callbacks, which run before the script, so the picker
always shows the current selection.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui import api, fmt
from ui.charts import components_chart, contributions_chart, trend_chart
from ui.layers import LAYERS
from ui.maps import build_deck, legend_html, selected_geoids

MAX_SELECTED = 20
CUTOFF_NOTE = "Dashed lines: ELEVATED at 50, HIGH at 75, CRITICAL at 90."
FILTER_KEYS = ["cls", "drv", "conf", "hot", "ml", "spa", "city"]

week = st.session_state["week"]
compare = (
    None if st.session_state["compare"] == "previous" else st.session_state["compare"]
)
theme = st.context.theme.type or "light"

try:
    summ = api.summary(week, compare)
    tracts = api.tracts(week)
    hexes = api.hexes()
    outlines = api.outlines()
except api.ApiError as exc:
    st.error(str(exc), title="Could not load this week")
    st.stop()

by_geoid = tracts.set_index("geoid", drop=False)
labels = {g: fmt.tract_label(r) for g, r in by_geoid.iterrows()}


# ----------------------------------------------------------------- selection


def _add(geoids: list[str], source: str) -> None:
    current = list(st.session_state.get("tracts", []))
    new = [g for g in geoids if g in labels and g not in current]
    room = MAX_SELECTED - len(current)
    if len(new) > room:
        st.toast(
            f"Selected the first {room} of {len(new)} tracts from {source}; the limit is {MAX_SELECTED}."
        )
    st.session_state["tracts"] = current + new[: max(room, 0)]


def on_map() -> None:
    now = selected_geoids(st.session_state["map"])
    before = st.session_state.get("_map_before", set())
    removed = before - now
    st.session_state["tracts"] = [
        g for g in st.session_state.get("tracts", []) if g not in removed
    ]
    _add(sorted(now - before), "the map")
    st.session_state["_map_before"] = now


def on_table() -> None:
    rows = st.session_state["table"].selection.rows
    shown = st.session_state["_table_geoids"]
    _add([shown[i] for i in rows], "the table")


def on_hit(hit: dict) -> None:
    if hit["kind"] == "city":
        name = hit["label"].removesuffix(" (all areas)")
        st.session_state["city"] = sorted(
            set(st.session_state.get("city", [])) | {name}
        )
    elif hit["kind"] == "region":
        name = hit["label"].removesuffix(" (Service Planning Area)")
        st.session_state["spa"] = sorted(set(st.session_state.get("spa", [])) | {name})
    else:
        _add(hit["geoids"], hit["label"])
    st.session_state["q"] = ""


def on_address() -> None:
    address = st.session_state.get("addr", "").strip()
    if len(address) < 5:
        return
    try:
        hit = api.geocode(address)
    except api.ApiError as exc:
        st.session_state["_addr_msg"] = str(exc).split(": ", 2)[-1]
        return
    st.session_state["_addr_msg"] = (
        f"{hit['matched_address']} is in {labels.get(hit['geoid'], hit['geoid'])}."
    )
    _add([hit["geoid"]], "the address")


def clear_filters() -> None:
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)


# ------------------------------------------------------------------- sidebar

with st.sidebar:
    st.subheader("Find")
    q = st.text_input(
        "Place or tract",
        key="q",
        placeholder="City, neighborhood, SPA or GEOID",
        type="search",
    )
    if len(q.strip()) >= 2:
        try:
            hits = api.search(q.strip())
        except api.ApiError as exc:
            hits = []
            st.caption(str(exc))
        for i, hit in enumerate(hits[:6]):
            n = len(hit["geoids"])
            st.button(
                f"{hit['label']} ({n} tract{'s' if n != 1 else ''})",
                key=f"hit_{i}",
                type="tertiary",
                on_click=on_hit,
                args=(hit,),
            )
        if not hits:
            st.caption("No match.")
    st.text_input(
        "Address", key="addr", placeholder="Street, city", on_change=on_address
    )
    if msg := st.session_state.pop("_addr_msg", None):
        st.caption(msg)

    st.subheader("Filter tracts")
    cls = st.pills(
        "Burden class",
        fmt.CLASS_ORDER,
        selection_mode="multi",
        key="cls",
        bind="query-params",
    )
    drv = st.pills(
        "Main driver",
        list(fmt.DRIVER_LABELS),
        format_func=fmt.DRIVER_LABELS.get,
        selection_mode="multi",
        key="drv",
        bind="query-params",
    )
    conf = st.pills(
        "Confidence",
        fmt.CONFIDENCE_ORDER,
        selection_mode="multi",
        key="conf",
        bind="query-params",
    )
    hot = st.pills(
        "Gi* cluster",
        ["HOTSPOT", "COLDSPOT", "NOT_SIGNIFICANT"],
        format_func={
            "HOTSPOT": "Hotspot",
            "COLDSPOT": "Coldspot",
            "NOT_SIGNIFICANT": "Neither",
        }.get,
        selection_mode="multi",
        key="hot",
        bind="query-params",
    )
    ml = st.slider(
        "Risk of turning CRITICAL, at least",
        0,
        95,
        0,
        step=5,
        format="%d%%",
        key="ml",
        bind="query-params",
        help="Model probability of entering CRITICAL within 4 weeks. Above 0, tracts that are CRITICAL now drop out.",
    )
    spa = st.multiselect(
        "Service Planning Area",
        sorted(tracts["region_name"].dropna().unique()),
        key="spa",
        bind="query-params",
    )
    city = st.multiselect(
        "City",
        sorted(tracts["city"].dropna().unique()),
        key="city",
        bind="query-params",
    )
    st.button("Clear filters", type="tertiary", on_click=clear_filters)

keep = pd.Series(True, index=tracts.index)
if cls:
    keep &= tracts["burden_class"].isin(cls)
if drv:
    keep &= tracts["main_driver"].isin(drv)
if conf:
    keep &= tracts["confidence"].isin(conf)
if hot:
    cluster = tracts["hotspot_class"].fillna("NOT_SIGNIFICANT").str.split("_").str[0]
    cluster = cluster.where(cluster != "NOT", "NOT_SIGNIFICANT")
    keep &= cluster.isin(hot)
if ml:
    keep &= tracts["prob_critical_4w"].fillna(-1) >= ml / 100
if spa:
    keep &= tracts["region_name"].isin(spa)
if city:
    keep &= tracts["city"].isin(city)
filtered = bool(cls or drv or conf or hot or ml or spa or city)

# ------------------------------------------------------------ headline + KPIs

st.header(summ["headline"]["title"])
if summ["headline"]["detail"]:
    st.caption(summ["headline"]["detail"])

k = summ["kpis"]
chronic, watch = k["chronic"], k["watchlist"]
change = k["people_critical_change"]
# Labels fit about 20 characters at 5 boxes per row; full definitions are in help.
with st.grid(5, gap="small", min_column_width=170):
    st.metric(
        "In CRITICAL tracts",
        f"{fmt.people(k['people_critical'])} people",
        delta=fmt.signed_people(change) if change is not None else "no comparison",
        delta_color="off",
        delta_arrow="auto" if change else "off",
        delta_description=f"vs {fmt.short_week(summ['compare_week'])}"
        if summ["compare_week"]
        else None,
        help=f"Residents of CRITICAL tracts (top tenth of burden): {k['people_critical_share']:.1%} of the county.",
        border=True,
        height="stretch",
    )
    st.metric(
        "In chronic tracts",
        f"{fmt.people(chronic['people'])} people" if chronic else "Not yet",
        delta=fmt.count(chronic["tracts"], "tract") if chronic else "needs 52 weeks",
        delta_color="off",
        delta_arrow="off",
        help="Tracts CRITICAL in at least 26 of the last 52 weeks.",
        border=True,
        height="stretch",
    )
    st.metric(
        "No ER in 30 minutes",
        f"{fmt.people(k['desert_people'])} people",
        delta=fmt.count(k["desert_tracts"], "tract"),
        delta_color="off",
        delta_arrow="off",
        help="Residents with no emergency department within a 30-minute drive (free-flow road network, no traffic).",
        border=True,
        height="stretch",
    )
    st.metric(
        "Watch list",
        fmt.count(watch["tracts"], "tract") if watch else "No forecast",
        delta=f"{watch['expected_entering']:.0f} expected" if watch else "this week",
        delta_color="off",
        delta_arrow="off",
        help=(
            f"Tracts not CRITICAL now with a {watch['threshold']:.0%}+ model probability of turning CRITICAL "
            "within 4 weeks. The model is calibrated, so the sum of all probabilities is the expected number."
        )
        if watch
        else "No model predictions for this week.",
        border=True,
        height="stretch",
    )
    st.metric(
        "Data as of",
        fmt.week_label(week).replace(", 20", " ’"),
        delta="PM2.5 ~20 wks late",
        delta_color="off",
        delta_arrow="off",
        help="\n\n".join(
            f"**{s['source']}** ({s['covers']}). {s['note']}"
            for s in summ["data_sources"]
        ),
        border=True,
        height="stretch",
    )

# ------------------------------------------------------------------ map + panel

map_col, panel_col = st.columns([1.55, 1], gap="medium")

with map_col:
    with st.container(horizontal=True, vertical_alignment="bottom"):
        layer_key = st.selectbox(
            "Color by",
            list(LAYERS),
            format_func=lambda k_: LAYERS[k_].label,
            key="layer",
            bind="query-params",
            width=240,
        )
        fill_by = st.segmented_control(
            "Fill",
            ["hexes", "tracts"],
            format_func={"hexes": "Where people live", "tracts": "Whole tracts"}.get,
            default="hexes",
            required=True,
            key="fill",
            bind="query-params",
        )
        show_hot = st.toggle(
            "Outline Gi* hotspots", key="outline_hot", bind="query-params"
        )
    layer = LAYERS[layer_key]
    selected = list(st.session_state.get("tracts", []))
    deck = build_deck(
        tracts,
        hexes,
        outlines,
        layer,
        theme=theme,
        fill_by=fill_by,
        keep=keep,
        selected=selected,
        show_hotspots=show_hot,
    )
    st.pydeck_chart(
        deck,
        on_select=on_map,
        selection_mode="multi-object",
        key="map",
        height=600,
        alt=f"Map of Los Angeles County census tracts colored by {layer.label.lower()}",
    )
    st.html(legend_html(layer, theme))
    note = layer.note
    if filtered:
        people_kept = tracts.loc[keep, "population"].sum()
        note += f" Showing {int(keep.sum()):,} of {len(tracts):,} tracts ({fmt.people(people_kept)} people) that match the filters."
    st.caption(note)


# ------------------------------------------------------------------ the panel


def tract_panel(geoid: str) -> None:
    try:
        d = api.tract(week, geoid)
        trend = api.trends((geoid,))
    except api.ApiError as exc:
        st.error(str(exc))
        return
    st.subheader(fmt.tract_label(d))
    st.caption(
        f"{fmt.place(d)} · {d['region_name'] or 'no SPA'} · GEOID `{geoid}` · {d['population']:,} residents"
    )

    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.badge(d["burden_class"], color=fmt.CLASS_BADGE[d["burden_class"]])
        if d.get("is_chronic"):
            st.badge("Chronic", color="gray")
        if (d.get("hotspot_class") or "").startswith("HOTSPOT"):
            st.badge("Gi* hotspot", color="gray")

    lines = [
        f"Burden percentile **{d['burden_percentile']:.1f}**"
        + (
            f" (90% range {d['percentile_p05']:.1f} to {d['percentile_p95']:.1f})"
            if d.get("percentile_p05") is not None
            else ""
        )
        + (
            f", {'up' if d['percentile_change'] > 0 else 'down'} {abs(d['percentile_change']):.1f} from the week before"
            if d.get("percentile_change")
            else ""
        )
        + "."
    ]
    if d.get("p_critical") is not None:
        n = round(d["p_critical"] * 200)
        source = (
            "PM2.5 interpolation error"
            if d["confidence_source"] == "pm25_interpolation"
            else "the component weights"
        )
        lines.append(
            f"CRITICAL in **{n} of 200** plausible versions of the index: {d['confidence']} confidence"
            f" in this week's call. Most of the doubt comes from {source}."
        )
    if d.get("weeks_critical") is not None:
        lines.append(
            f"CRITICAL in {d['weeks_critical']} of the last {d['weeks_observed']} weeks."
        )
    if d.get("nearest_er_minutes") is not None:
        lines.append(
            f"Nearest ER {d['nearest_er_minutes']:.0f} minutes away; {d['er_count_30min']} ERs within a 30-minute drive."
        )
    st.markdown("\n\n".join(lines))

    st.markdown("**What drives the score**")
    st.altair_chart(
        components_chart(pd.DataFrame([d]), {geoid: fmt.tract_label(d)}),
        alt="Bar chart of the four index components for this tract",
    )
    st.markdown("**Burden percentile, every week**")
    st.altair_chart(
        trend_chart(trend, {geoid: fmt.tract_label(d)}),
        alt="Line chart of this tract's weekly burden percentile",
    )
    st.caption(CUTOFF_NOTE)

    st.markdown("**Forecast**")
    if d.get("prob_critical_4w") is None:
        st.caption("This tract is CRITICAL now, so the model does not forecast it.")
    else:
        st.markdown(
            f"Probability of becoming CRITICAL within 4 weeks: **{d['prob_critical_4w']:.0%}**"
            f" (rank {d['ml_rank']:,} of {tracts['prob_critical_4w'].notna().sum():,})."
        )
        mc = d.get("ml_contributions")
        if mc:
            st.altair_chart(
                contributions_chart(mc["top"], mc["other_features"], mc["n_other"]),
                alt="Bar chart of the features that raise or lower this tract's forecast",
            )
            st.caption(
                "Each bar is coefficient × scaled value in the logistic regression. "
                "Correlated features can offset each other, so read them as the model's arithmetic, not causes."
            )

    b = d.get("briefing")
    if b:
        with st.expander(
            f"Policy briefing, generated {fmt.week_label(b['generated_at'][:10])}"
        ):
            st.markdown(b["summary"] or "")
            if b.get("key_risks"):
                st.markdown(
                    "**Key risks**\n\n" + "\n".join(f"- {r}" for r in b["key_risks"])
                )
            actions = b.get("actions") or []
            if actions:
                st.markdown(
                    "**Actions**\n\n"
                    + "\n".join(
                        f"- {a['action']}"
                        + (
                            f" *({'; '.join(a.get('citations', []))})*"
                            if a.get("citations")
                            else ""
                        )
                        for a in actions
                    )
                )
            if b.get("data_gaps"):
                st.caption("Gaps: " + " ".join(b["data_gaps"]))
            st.caption(
                f"Written by {b['model']} from retrieved policy documents; citations checked."
            )


def comparison_panel(geoids: list[str]) -> None:
    rows = by_geoid.loc[geoids]
    crit = int((rows["burden_class"] == "CRITICAL").sum())
    top = rows["main_driver"].value_counts()
    c1, c2, c3 = st.columns([1, 1, 1.5], gap="small")
    c1.metric(
        "People", fmt.people(rows["population"].sum()), border=True, height="stretch"
    )
    c2.metric("CRITICAL", f"{crit} of {len(rows)}", border=True, height="stretch")
    c3.metric(
        "Top driver",
        fmt.DRIVER_SHORT.get(top.index[0], "n/a") if len(top) else "n/a",
        border=True,
        height="stretch",
        help="The most common largest component among the selected tracts.",
    )
    names = {g: labels[g] for g in geoids}
    st.markdown("**What drives the score**")
    st.altair_chart(
        components_chart(rows, names), alt="Grouped bars of index components by tract"
    )
    try:
        trend = api.trends(tuple(geoids))
    except api.ApiError as exc:
        st.error(str(exc))
    else:
        st.markdown("**Burden percentile, every week**")
        st.altair_chart(
            trend_chart(trend, names),
            alt="Weekly burden percentile, one line per tract",
        )
        st.caption(CUTOFF_NOTE)
    table = rows[
        ["geoid", "name", "city", "neighborhood", "population", "burden_class", "burden_percentile",
         "main_driver", "prob_critical_4w", "weeks_critical", "confidence"]
    ]  # fmt: skip
    st.download_button(
        "Download these tracts (CSV)",
        table.to_csv(index=False),
        file_name=f"chbindex_{week}_{len(geoids)}_tracts.csv",
        mime="text/csv",
        type="tertiary",
        icon=":material/download:",
    )


with panel_col:
    selected = st.multiselect(
        "Selected tracts",
        list(labels),
        format_func=labels.get,
        key="tracts",
        bind="query-params",
        max_selections=MAX_SELECTED,
        placeholder="Click the map, a table row, or search",
    )
    if len(selected) == 1:
        tract_panel(selected[0])
    elif selected:
        comparison_panel(selected)
    else:
        shown = tracts[keep].sort_values("burden_percentile", ascending=False)
        st.session_state["_table_geoids"] = shown["geoid"].tolist()
        st.markdown(
            f"**{'Matching tracts' if filtered else 'All tracts'}, highest burden first**"
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Tract": shown["name"].str.replace(
                        "Census Tract ", "", regex=False
                    ),
                    "Place": [fmt.place(r) for _, r in shown.iterrows()],
                    "Class": shown["burden_class"],
                    "Percentile": shown["burden_percentile"],
                    "Main driver": shown["main_driver"].map(fmt.DRIVER_LABELS),
                }
            ),
            hide_index=True,
            height=560,
            key="table",
            on_select=on_table,
            selection_mode="multi-row",
            column_config={
                "Tract": st.column_config.TextColumn(width=70),
                "Place": st.column_config.TextColumn(width="medium"),
                "Class": st.column_config.TextColumn(width=80),
                "Percentile": st.column_config.NumberColumn(format="%.1f", width=80),
            },
            alt="Table of tracts sorted by burden percentile",
        )
