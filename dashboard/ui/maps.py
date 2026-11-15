"""
The pydeck map: H3 hexes (or tract polygons) colored by one layer, tract
outlines on top, plus optional hotspot outlines and the selected tracts.

Layer order, bottom to top:
    hexes      H3HexagonLayer, colored, where people live (hex mode only;
               not pickable: too many objects for Streamlit selections)
    tracts     GeoJsonLayer of all tracts: the fill in "Whole tracts" mode,
               transparent in hex mode; pickable, so a click selects the tract
    outlines   every tract boundary, thin, stroke only
    hotspots   Gi* hotspot tracts (95% or 99%), thicker stroke
    selected   the selected tracts, thickest stroke

Every layer has an id, which st.pydeck_chart needs for selections.
map_style=None lets Streamlit pick a light or dark basemap from the theme.
"""

from __future__ import annotations

import pandas as pd
import pydeck as pdk

from ui.layers import Layer, fill_colors, legend_rows, value_text

# pydeck reads plain strings as data expressions ("@@=pixels"); constants need String
PIXELS = pdk.types.String("pixels")

VIEW = pdk.ViewState(
    latitude=34.28, longitude=-118.25, zoom=8.25, min_zoom=7.5, max_zoom=15
)


def _ink(theme: str, alpha: int) -> list[int]:
    return [236, 236, 232, alpha] if theme == "dark" else [26, 26, 26, alpha]


def tract_tooltip_fields(df: pd.DataFrame, layer: Layer) -> pd.DataFrame:
    """Per-tract text for the tooltip, computed once and joined to hexes."""
    place = df["neighborhood"].fillna(df["city"]).fillna("")
    city = df["city"].fillna("")
    place = place.where(place == city, place + ", " + city)
    return pd.DataFrame(
        {
            "geoid": df["geoid"],
            "tip_name": df["name"].fillna(df["geoid"]),
            "tip_place": place,
            "tip_class": df["burden_class"],
            "tip_value": [value_text(v, layer) for v in df[layer.column]],
        }
    )


def _features(
    outlines: dict, geoids: set[str], props: pd.DataFrame | None = None
) -> dict:
    """A FeatureCollection of the chosen tracts; geometry objects are shared, not copied."""
    lookup = props.set_index("geoid").to_dict("index") if props is not None else {}
    feats = [
        {
            "type": "Feature",
            "geometry": f["geometry"],
            "properties": {"geoid": f["id"], **lookup.get(f["id"], {})},
        }
        for f in outlines["features"]
        if f["id"] in geoids
    ]
    return {"type": "FeatureCollection", "features": feats}


def build_deck(
    tracts: pd.DataFrame,
    hexes: pd.DataFrame,
    outlines: dict,
    layer: Layer,
    *,
    theme: str,
    fill_by: str,
    keep: pd.Series,
    selected: list[str],
    show_hotspots: bool,
) -> pdk.Deck:
    tips = tract_tooltip_fields(tracts, layer)
    tips["fill"] = fill_colors(tracts[layer.column], layer, theme, keep.to_numpy())
    layers = []

    if fill_by == "hexes":
        # Paint only. Streamlit stops returning clicks once a pickable layer
        # holds ~24,000 objects, so the hexes are not pickable; the tract
        # layer below takes the clicks and tooltips instead.
        data = hexes[["h3_index", "geoid"]].merge(
            tips[["geoid", "fill"]], on="geoid", how="inner"
        )
        layers.append(
            pdk.Layer(
                "H3HexagonLayer",
                id="hexes",
                data=data,
                get_hexagon="h3_index",
                get_fill_color="fill",
                get_line_color=_ink(theme, 18),
                line_width_min_pixels=0.5,
                stroked=True,
                filled=True,
                extruded=False,
                pickable=False,
            )
        )
        tips["fill"] = [[0, 0, 0, 0]] * len(tips)  # invisible, still clickable

    # 2,498 tract shapes: the colored fill in "Whole tracts" mode, an invisible
    # click and hover target over the hexes otherwise
    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            id="tracts",
            data=_features(outlines, set(tracts["geoid"]), tips),
            get_fill_color="properties.fill",
            stroked=False,
            filled=True,
            pickable=True,
            auto_highlight=True,
            highlight_color=_ink(theme, 40),
        )
    )
    tooltip_html = (
        "<b>{tip_name}</b><br/>{tip_place}<br/>{tip_class} &middot; {tip_value}"
    )

    layers.append(
        pdk.Layer(
            "GeoJsonLayer",
            id="outlines",
            data=outlines,
            stroked=True,
            filled=False,
            get_line_color=_ink(theme, 45),
            line_width_units=PIXELS,
            get_line_width=0.6,
            pickable=False,
        )
    )
    if show_hotspots:
        hot = set(
            tracts.loc[
                tracts["hotspot_class"].fillna("").str.startswith("HOTSPOT"), "geoid"
            ]
        )
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                id="hotspots",
                data=_features(outlines, hot),
                stroked=True,
                filled=False,
                get_line_color=_ink(theme, 230),
                line_width_units=PIXELS,
                get_line_width=2,
                pickable=False,
            )
        )
    if selected:
        layers.append(
            pdk.Layer(
                "GeoJsonLayer",
                id="selected",
                data=_features(outlines, set(selected)),
                stroked=True,
                filled=False,
                get_line_color=_ink(theme, 255),
                line_width_units=PIXELS,
                get_line_width=3.5,
                pickable=False,
            )
        )

    bg, fg = ("#202020", "#ECECE8") if theme == "dark" else ("#FAFAF7", "#1A1A1A")
    return pdk.Deck(
        layers=layers,
        initial_view_state=VIEW,
        map_style=None,
        tooltip={
            "html": tooltip_html,
            "style": {
                "backgroundColor": bg,
                "color": fg,
                "fontFamily": "'Public Sans', sans-serif",
                "fontSize": "13px",
                "border": "0.5px solid #888",
                "borderRadius": "4px",
                "padding": "6px 8px",
            },
        },
    )


def selected_geoids(event: object) -> set[str]:
    """GEOIDs of every tract selected on the map."""
    objects = getattr(getattr(event, "selection", None), "objects", None) or {}
    out: set[str] = set()
    for layer_id in ("tracts",):
        for obj in objects.get(layer_id, []):
            geoid = obj.get("geoid") or (obj.get("properties") or {}).get("geoid")
            if geoid:
                out.add(geoid)
    return out


def legend_html(layer: Layer, theme: str) -> str:
    """Swatches for the legend. Streamlit has no native map legend, so this is
    the one place the app writes HTML; colors come from the layer, not CSS."""
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin:0 14px 4px 0;'
        f'font-size:0.8rem;white-space:nowrap">'
        f'<span style="width:12px;height:12px;border-radius:2px;background:{color};'
        f'border:0.5px solid rgba(128,128,128,.6)"></span>{text}</span>'
        for color, text in legend_rows(layer, theme)
    )
    return f'<div style="display:flex;flex-wrap:wrap;line-height:1.4">{items}</div>'
