"""
Tests for the Streamlit dashboard (dashboard/).

Pure helpers are tested directly. The app itself runs headless with
streamlit.testing AppTest; its API client (ui.api._get) is pointed at the
FastAPI app in-process (TestClient), backed by the same 4-tract fixture
warehouse as tests/test_api.py, so the whole chain from warehouse to page
runs without a network.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from src.serving.api import app, get_service
from tests.test_api import A, B, C, config, db_path, svc  # noqa: F401 (fixtures)

DASHBOARD = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(DASHBOARD))

from ui import api as ui_api  # noqa: E402
from ui import fmt  # noqa: E402
from ui.layers import LAYERS, bin_index, fill_colors, legend_rows  # noqa: E402
from ui.maps import build_deck, selected_geoids  # noqa: E402

# ------------------------------------------------------------------ helpers


def test_bins_are_right_open_and_missing_is_minus_one():
    burden = LAYERS["burden"]
    idx = bin_index(pd.Series([10, 25, 89.9, 90, None]), burden)
    assert idx.tolist() == [0, 1, 3, 4, -1]  # 90 is CRITICAL, 89.9 is not
    z = bin_index(
        pd.Series([-1.5, -0.5, -0.25, 0.25, 2]), LAYERS["heat"]
    )  # -0.25 is "near average"
    assert z.tolist() == [0, 1, 2, 3, 4]


def test_fill_colors_fade_filtered_tracts():
    colors = fill_colors(
        pd.Series([95.0, None]), LAYERS["burden"], "light", np.array([True, False])
    )
    assert colors[0] == [179, 0, 0, 225]  # #B30000, CRITICAL
    assert colors[1][3] == 28  # filtered out: faint


def test_legend_puts_worst_first():
    assert legend_rows(LAYERS["burden"], "light")[0][1].endswith("CRITICAL")
    assert legend_rows(LAYERS["confidence"], "light")[0][1].startswith("LOW")
    assert all(
        len(layer.colors) == len(layer.labels) == len(layer.bins) + 1
        for layer in LAYERS.values()
    )


def test_people_precision():
    assert fmt.people(1_064_775) == "1.1M"
    assert fmt.people(810_396) == "810K"
    assert fmt.people(2_691) == "2,700"
    assert fmt.signed_people(-18_576) == "-19K"


def _deck(fill_by="hexes", selected=(), hot=False):
    tracts = pd.DataFrame(
        {
            "geoid": ["g1", "g2"],
            "name": ["Census Tract 1", "Census Tract 2"],
            "city": ["Glendale", "Los Angeles"],
            "neighborhood": [None, "Watts"],
            "burden_class": ["CRITICAL", "LOW"],
            "burden_percentile": [95.0, 10.0],
            "hotspot_class": ["HOTSPOT_95", "NOT_SIGNIFICANT"],
        }
    )
    hexes = pd.DataFrame({"h3_index": ["h1", "h2", "h3"], "geoid": ["g1", "g1", "g2"]})
    ring = [[-118, 34], [-118, 34.01], [-117.99, 34], [-118, 34]]
    outlines = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": g,
                "properties": {"geoid": g},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
            for g in ["g1", "g2"]
        ],
    }
    return build_deck(
        tracts, hexes, outlines, LAYERS["burden"], theme="light", fill_by=fill_by,
        keep=pd.Series([True, True]), selected=list(selected), show_hotspots=hot,
    )  # fmt: skip


def test_deck_layers_and_picking():
    spec = json.loads(_deck(selected=["g2"], hot=True).to_json())
    layers = {layer["id"]: layer for layer in spec["layers"]}
    assert list(layers) == ["hexes", "tracts", "outlines", "hotspots", "selected"]
    # Hexes are paint only; Streamlit drops clicks on a 24,000-object layer
    assert not layers["hexes"].get("pickable", False)
    assert layers["tracts"]["pickable"]
    # Over the hexes the tract layer is an invisible click target
    fills = [f["properties"]["fill"] for f in layers["tracts"]["data"]["features"]]
    assert all(f == [0, 0, 0, 0] for f in fills)
    # A constant must not turn into a data expression ("@@=pixels")
    assert layers["outlines"]["lineWidthUnits"] == "pixels"
    assert [
        f["properties"]["geoid"] for f in layers["hotspots"]["data"]["features"]
    ] == ["g1"]


def test_whole_tract_mode_colors_the_tract_layer():
    spec = json.loads(_deck(fill_by="tracts").to_json())
    ids = [layer["id"] for layer in spec["layers"]]
    assert ids == ["tracts", "outlines"]
    fills = [f["properties"]["fill"] for f in spec["layers"][0]["data"]["features"]]
    assert fills[0][:3] == [179, 0, 0]


def test_selected_geoids_reads_tract_features():
    class Sel(dict):
        __getattr__ = dict.get

    event = Sel(
        selection=Sel(
            objects={"tracts": [{"properties": {"geoid": "g1"}}, {"geoid": "g2"}]}
        )
    )
    assert selected_geoids(event) == {"g1", "g2"}
    assert selected_geoids(Sel(selection=None)) == set()


# ----------------------------------------------------------------- the app


@pytest.fixture
def at(svc, monkeypatch):  # noqa: F811
    app.dependency_overrides[get_service] = lambda: svc
    client = TestClient(app)

    def fake_get(path, **params):
        resp = client.get(path, params=params)
        if resp.status_code >= 400:
            raise ui_api.ApiError(
                f"{path}: {resp.status_code} {resp.json().get('detail')}"
            )
        return resp.json()

    monkeypatch.setattr(ui_api, "_get", fake_get)
    st.cache_data.clear()
    yield AppTest.from_file(str(DASHBOARD / "streamlit_app.py"), default_timeout=30)
    app.dependency_overrides.clear()


def test_overview_renders_headline_and_kpis(at):
    at.run()
    assert not at.exception
    assert at.header[0].value.startswith("3,000 people live in CRITICAL tracts")
    labels = [m.label for m in at.metric]
    assert labels == [
        "In CRITICAL tracts",
        "In chronic tracts",
        "No ER in 30 minutes",
        "Watch list",
        "Data as of",
    ]
    assert at.metric[0].value == "3,000 people"
    assert at.metric[3].value == "1 tract"


def test_selecting_tracts_opens_panel_then_comparison(at):
    at.run()
    at.multiselect(key="tracts").set_value([C]).run()
    assert not at.exception
    assert at.main.subheader[-1].value == "Tract 3, Watts"
    assert any(
        "Probability of becoming CRITICAL within 4 weeks: **70%**" in m.value
        for m in at.markdown
    )

    at.multiselect(key="tracts").set_value([A, B]).run()
    assert not at.exception
    strip = next(m.value for m in at.markdown if "CRITICAL · top driver" in m.value)
    assert strip.startswith("**3,000 people** · **2 of 2** CRITICAL")
    assert (
        "**air quality**" in strip or "**heat**" in strip
    )  # A and B tie at one tract each


def test_search_and_filters(at):
    at.run()
    at.text_input(key="q").set_value("watts").run()
    hits = [b for b in at.button if str(b.key).startswith("hit_")]
    assert hits and hits[0].label.startswith("Los Angeles - Watts")
    hits[0].click().run()
    assert at.multiselect(key="tracts").value == [C]

    at.multiselect(key="tracts").set_value([]).run()
    at.pills(key="cls").set_value(["CRITICAL"]).run()
    assert not at.exception
    assert any("Showing 2 of 4 tracts" in c.value for c in at.caption)


def test_watch_list_page_and_open_in_overview(at):
    at.run()
    at.switch_page("app_pages/watchlist.py").run()
    assert not at.exception
    # C (0.7) is the only tract at or above the 0.5 threshold
    assert at.header[0].value.startswith("1 tract has a 50% or higher chance")
    table = at.dataframe(key="wl_table").value
    assert table["Tract"].tolist() == ["Tract 3"]
    assert table["Why the model flags it"].iloc[0] != ""

    at.slider(key="wl_min").set_value(5).run()
    assert at.dataframe(key="wl_table").value["Tract"].tolist() == [
        "Tract 3",
        "Tract 4",
    ]


def test_validation_page(at):
    at.run()
    at.switch_page("app_pages/validation.py").run()
    assert not at.exception
    assert at.header[0].value == "The data check passes for Jun 22, 2026"
    subs = [s.value for s in at.subheader]
    assert any("PR-AUC 0.53 against 0.40" in s for s in subs)
    assert any(s.startswith("Recall is lowest in the highest third") for s in subs)
    assert any("No accepted baseline published" in c.value for c in at.caption)


def test_methods_page_links_every_story_to_the_readme(at):
    at.run()
    at.switch_page("app_pages/methods.py").run()
    assert not at.exception
    links = [m.value for m in at.markdown if "](https://github.com/" in m.value]
    assert len(links) == len(at.subheader) == 12  # 11 stories + limitations


def test_overview_reads_tracts_from_the_url(at):
    # The watch list's "Open in Overview" sends the selected rows this way
    at.query_params["tracts"] = ["Tract 3, Watts"]
    at.run()
    assert not at.exception
    assert at.multiselect(key="tracts").value == [C]
