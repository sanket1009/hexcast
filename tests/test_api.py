"""
Tests for the serving layer (src/serving): a 4-tract, 3-week DuckDB fixture
whose expected numbers are worked out by hand in the comments.

    tract  pop   income  class 06-15  class 06-22  notes (06-22)
    A      1000  30k     CRITICAL     CRITICAL     chronic, confidence HIGH, briefing
    B      2000  50k     HIGH         CRITICAL     confidence MEDIUM
    C      3000  70k     HIGH         HIGH         ML 0.7, no ER but access unscored
    D      4000  90k     LOW          LOW          ML 0.1, no ER within 30 min (desert)
"""

from datetime import date

import duckdb
import pandas as pd
import pytest
import requests
from fastapi.testclient import TestClient

from src.serving.api import app, get_service
from src.serving.service import Service, add_drivers, format_people, split_community
from src.serving.store import LocalStore

SCHEMA = "chbindex_06037"
FIPS = "06037"
A, B, C, D = (f"06037000{i}00" for i in range(1, 5))
WEEKS = [date(2026, 6, 8), date(2026, 6, 15), date(2026, 6, 22)]
CLASSES = {
    WEEKS[0]: ["HIGH", "HIGH", "ELEVATED", "LOW"],
    WEEKS[1]: ["CRITICAL", "HIGH", "HIGH", "LOW"],
    WEEKS[2]: ["CRITICAL", "CRITICAL", "HIGH", "LOW"],
}
PCT = {"CRITICAL": 95.0, "HIGH": 80.0, "ELEVATED": 60.0, "LOW": 20.0}
# Signed components per tract (pm25, lst, access, vulnerability)
Z = {
    A: (2.0, 0.1, 0.2, 0.3),  # air quality
    B: (0.1, 1.5, 0.2, 0.3),  # heat
    C: (0.1, 0.2, 1.8, 0.3),  # healthcare access
    D: (-1.0, -1.0, -1.0, -0.5),  # vulnerability least low, but last everywhere
}


def _write(con, name: str, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df["city_fips"] = FIPS
    con.register("tmp", df)
    con.execute(f'CREATE TABLE "{SCHEMA}"."{name}" AS SELECT * FROM tmp')
    con.unregister("tmp")


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("wh") / "w.duckdb")
    con = duckdb.connect(path)
    con.execute(f'CREATE SCHEMA "{SCHEMA}"')
    geoids = [A, B, C, D]
    _write(
        con,
        "tract_geometries",
        [
            {"geoid": g, "name": f"Tract {i}", "population": p, "median_income": m,
             "centroid_lat": 34.0, "centroid_lon": -118.0}
            for i, (g, p, m) in enumerate(
                zip(geoids, [1000, 2000, 3000, 4000], [30e3, 50e3, 70e3, 90e3], strict=False), 1
            )
        ],
    )  # fmt: skip
    _write(
        con,
        "tract_reporting_areas",
        [
            {"geoid": A, "region_name": "San Fernando Valley", "region_id": "2",
             "community_name": "City of Glendale"},
            {"geoid": B, "region_name": "San Fernando Valley", "region_id": "2",
             "community_name": "City of Glendale"},
            {"geoid": C, "region_name": "South", "region_id": "6",
             "community_name": "Los Angeles - Watts"},
            {"geoid": D, "region_name": "San Gabriel Valley", "region_id": "3",
             "community_name": "Unincorporated - Altadena"},
        ],
    )  # fmt: skip
    scores, hot, chronic, conf, pm, lst, access = [], [], [], [], [], [], []
    for w in WEEKS:
        for g, cls in zip(geoids, CLASSES[w], strict=False):
            z = Z[g]
            scores.append(
                {"geoid": g, "burden_class": cls, "burden_percentile": PCT[cls],
                 "raw_burden": sum(z), "compound_score": 0.5, "pm25_z": z[0],
                 "lst_z": z[1], "access_z": z[2], "vulnerability_z": z[3],
                 "week_start_date": w}
            )  # fmt: skip
            hot.append({"geoid": g, "hotspot_class": "NOT_SIGNIFICANT",
                        "gi_star_zscore": 0.1, "week_start_date": w})  # fmt: skip
            chronic.append(
                {"geoid": g, "weeks_critical": 30 if g == A else 2,
                 "weeks_observed": 52, "week_start_date": w,
                 "is_chronic": (g == A) if w == WEEKS[2] else None}
            )  # fmt: skip
            p = {A: 0.95, B: 0.6, C: 0.2, D: 0.0}[g]
            conf.append(
                {"geoid": g, "p_critical": p, "class_retention": 0.9,
                 "retention_pm25": 0.7, "retention_weights": 0.95,
                 "percentile_p05": 70.0, "percentile_p95": 99.0,
                 "pm25_error_sd": 0.9,
                 "confidence": {A: "HIGH", B: "MEDIUM", C: "MEDIUM", D: "HIGH"}[g],
                 "n_draws": 200, "week_start_date": w}
            )  # fmt: skip
            pm.append({"geoid": g, "pm25_kriged": 8.123456, "cv_mae": 1.0 + WEEKS.index(w),
                       "n_monitors": 21, "week_start_date": w})  # fmt: skip
            lst.append({"geoid": g, "lst_mean_celsius": 40.04, "week_start_date": w,
                        "lst_zscore": None if g == D else 0.1})  # fmt: skip
            access.append(
                {"geoid": g, "access_score_er": 1.0,
                 "er_count_30min": 0 if g in (C, D) else 5,
                 "nearest_er_minutes": 12.0, "scored": g != C, "week_start_date": w}
            )  # fmt: skip
    for name, rows in [
        ("compound_scores", scores),
        ("hotspot_detection", hot),
        ("tract_chronic", chronic),
        ("tract_confidence", conf),
        ("pm25_kriged", pm),
        ("lst_tract_stats", lst),
        ("healthcare_access", access),
    ]:
        _write(con, name, rows)

    w = WEEKS[2]
    _write(
        con,
        "ml_predictions",
        [
            {"geoid": C, "prob_critical_4w": 0.7, "burden_percentile": 80.0,
             "model_version": "v1", "week_start_date": w, "rank": 1},
            {"geoid": D, "prob_critical_4w": 0.1, "burden_percentile": 20.0,
             "model_version": "v1", "week_start_date": w, "rank": 2},
        ],
    )  # fmt: skip
    # logit = intercept + terms; C: terms 1.5, -0.4, 0.2 ; D: -2.0, 0.1, 0.0
    _write(
        con,
        "ml_contributions",
        [
            {"geoid": g, "feature": f, "value": 1.0, "imputed": False,
             "contribution": c, "logit": lg, "model_version": "v1",
             "week_start_date": w}
            for g, lg, terms in [
                (C, 0.8, [("access_z", 1.5), ("pm25_z", -0.4), ("lst_z", 0.2)]),
                (D, -2.2, [("access_z", -2.0), ("pm25_z", 0.1), ("lst_z", 0.0)]),
            ]
            for f, c in terms
        ],
    )  # fmt: skip
    _write(
        con,
        "acs_vulnerability",
        [
            {"geoid": g, "population": p, "no_vehicle_pct": 10.0,
             "no_vehicle_base": p // 2, "limited_english_pct": nv,
             "limited_english_base": p // 2, "low_income_pct": 20.0,
             "acs_vintage": "2020-2024 ACS 5-year"}
            for g, p, nv in zip(geoids, [1000, 2000, 3000, 4000], [40.0, 40.0, 0.0, 0.0], strict=False)
        ],
    )  # fmt: skip
    _write(
        con,
        "acs_race_ethnicity",
        [
            {"geoid": g, "total": p, "latino": p // 2, "white": p // 2, "black": 0,
             "asian": 0, "other": 0, "latino_pct": 50.0, "white_pct": 50.0,
             "black_pct": 0.0, "asian_pct": 0.0, "other_pct": 0.0,
             "acs_vintage": "2020-2024 ACS 5-year"}
            for g, p in zip(geoids, [1000, 2000, 3000, 4000], strict=False)
        ],
    )  # fmt: skip
    _write(con, "hospitals", [{"hospital_id": "h1", "data_date": "10/16/2026"}])
    _write(
        con,
        "h3_hexes",
        [{"h3_index": f"89283082{i}ffff", "geoid": g, "population": 10 * i}
         for i, g in enumerate(geoids, 1)],
    )  # fmt: skip
    _write(
        con,
        "tract_outlines",
        [{"geoid": g, "geometry_geojson":
          '{"type":"Polygon","coordinates":[[[-118,34],[-118,34.01],[-117.99,34],[-118,34]]]}'}
         for g in geoids],
    )  # fmt: skip
    _write(
        con,
        "briefings",
        [
            {"geoid": A, "week_start_date": w, "urgency": "CRITICAL",
             "summary": "Air quality drives this tract.",
             "key_risks_json": '["PM2.5"]', "actions_json": '[{"action": "Open cooling centers"}]',
             "data_gaps_json": "[]", "sources_json": '["doc1"]',
             "retrieved_chunk_ids_json": "[]", "model": "claude-haiku-5-5",
             "generated_at": pd.Timestamp("2026-11-10 12:00"),
             "vulnerable_groups_json": "[]", "inputs_json": "{}"}
        ],
    )  # fmt: skip
    _write(
        con,
        "drift_checks",
        [{"metric": "lst_coverage", "value": 0.998, "status": "ok",
          "moderate_threshold": 0.95, "major_threshold": 0.8,
          "trailing_weeks": 8, "week_start_date": w}],
    )  # fmt: skip
    _write(
        con,
        "calenviroscreen",
        [{"geoid": g, "ces_percentile": 50.0} for g in geoids],
    )
    con.close()
    return path


@pytest.fixture(scope="module")
def config(db_path):
    return {
        "full_fips": FIPS,
        "city_name": "Los Angeles",
        "gcp": {"bigquery_dataset": SCHEMA},
        "storage": {"backend": "local", "local_db_path": db_path},
        "airflow": {"ingest_lag_weeks": 20},
        "pm25_window_weeks": 4,
        "gee": {"composite_window_days": 32},
        "product": {"watchlist_threshold": 0.5, "api_cache_seconds": 600},
    }


@pytest.fixture(scope="module")
def svc(db_path, config):
    return Service(LocalStore(db_path, SCHEMA), config)


@pytest.fixture(scope="module")
def client(svc):
    app.dependency_overrides[get_service] = lambda: svc
    yield TestClient(app)
    app.dependency_overrides.clear()


# --------------------------------------------------------------------- store


def test_store_binds_parameters_and_is_read_only(db_path):
    store = LocalStore(db_path, SCHEMA)
    df = store.query(
        "SELECT COUNT(*) AS n FROM {dataset}.compound_scores WHERE geoid = @g",
        {"g": A},
    )
    assert df["n"].iloc[0] == 3
    # A quote in a value stays a value, never SQL
    df = store.query(
        "SELECT COUNT(*) AS n FROM {dataset}.compound_scores WHERE geoid = @g",
        {"g": "x' OR '1'='1"},
    )
    assert df["n"].iloc[0] == 0
    with pytest.raises(duckdb.Error):
        store.query("DELETE FROM {dataset}.compound_scores")


# ------------------------------------------------------------------- helpers


def test_helpers():
    assert format_people(1_064_775) == "1.1M"
    assert format_people(812_400) == "812K"
    assert format_people(9_430) == "9,400"
    assert split_community("City of Glendale") == ("Glendale", None)
    assert split_community("Los Angeles - Watts") == ("Los Angeles", "Watts")
    assert split_community(None) == (None, None)


def test_drivers_follow_top_quarter_rule():
    df = pd.DataFrame(
        [
            dict(
                zip(
                    ["pm25_z", "lst_z", "access_z", "vulnerability_z"],
                    Z[g],
                    strict=False,
                )
            )
            for g in Z
        ]
    )
    out = add_drivers(df)
    assert out["main_driver"].tolist() == [
        "air_quality", "heat", "healthcare_access", "vulnerability",
    ]  # fmt: skip
    # Tract A is top for air quality only (rank pct 1.0); vulnerability ties
    # at 0.3 for A, B, C, so their percentile is 0.75 and also counts
    assert out["drivers"][0] == ["air_quality", "vulnerability"]
    # D is in no top quarter: falls back to its single largest component
    assert out["drivers"][3] == ["vulnerability"]


# ------------------------------------------------------------------ endpoints


def test_health_and_meta(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "backend": "local", "latest_week": "2026-06-22"}
    meta = client.get("/meta").json()
    assert meta["weeks"] == [w.isoformat() for w in WEEKS]
    assert meta["components"]["healthcare_access"] == "healthcare access"


def test_summary_kpis(client):
    s = client.get("/weeks/latest/summary").json()
    k = s["kpis"]
    assert s["compare_week"] == "2026-06-15"
    assert k["people_critical"] == 3000  # A + B
    assert k["people_critical_share"] == 0.3
    assert k["people_critical_change"] == 2000  # only A the week before
    assert k["tracts_critical"] == 2
    assert k["chronic"] == {"tracts": 1, "people": 1000}
    # C has no ER but is unscored; only D counts
    assert (k["desert_tracts"], k["desert_people"]) == (1, 4000)
    assert k["watchlist"]["tracts"] == 1  # C at 0.7
    assert k["watchlist"]["expected_entering"] == 0.8
    assert k["critical_confidence"] == {"HIGH": 1, "MEDIUM": 1, "LOW": 0}
    assert s["headline"]["title"] == (
        "3,000 people live in CRITICAL tracts in the week of June 22, 2026, "
        "2,000 more than June 15"
    )
    assert s["data_sources"][2]["covers"] == "hospital list dated 10/16/2026"


def test_summary_without_history(client):
    k = client.get("/weeks/2026-06-08/summary").json()
    assert k["compare_week"] is None
    assert k["kpis"]["chronic"] is None  # window not full
    assert k["kpis"]["watchlist"] is None  # no predictions that week
    assert "than" not in k["headline"]["title"]


def test_week_tracts(client):
    body = client.get("/weeks/2026-06-22/tracts").json()
    assert body["model_version"] == "v1"
    rows = {t["geoid"]: t for t in body["tracts"]}
    assert len(rows) == 4
    a, c = rows[A], rows[C]
    assert (a["city"], a["region_name"]) == ("Glendale", "San Fernando Valley")
    assert a["main_driver"] == "air_quality"
    assert a["agreement"] == 0.95  # CRITICAL: agreement = p_critical
    assert c["agreement"] == 0.8  # not CRITICAL: 1 - p_critical
    assert a["prob_critical_4w"] is None  # already CRITICAL
    assert c["prob_critical_4w"] == 0.7
    assert rows[B]["prev_class"] == "HIGH"
    assert rows[B]["percentile_change"] == 15.0
    assert a["pm25_kriged"] == 8.12  # rounded
    assert a["confidence_source"] == "pm25_interpolation"


def test_tract_detail(client):
    d = client.get(f"/weeks/latest/tracts/{C}?top_features=2").json()
    mc = d["ml_contributions"]
    assert [t["feature"] for t in mc["top"]] == ["access_z", "pm25_z"]  # by size
    assert mc["top"][0]["label"] == "Healthcare access component"
    assert mc["other_features"] == 0.2 and mc["n_other"] == 1
    assert d["briefing"] is None
    assert d["acs"]["no_vehicle_pct"] == 10.0

    a = client.get(f"/weeks/latest/tracts/{A}").json()
    assert a["ml_contributions"] is None
    assert a["briefing"]["actions"] == [{"action": "Open cooling centers"}]
    assert a["is_chronic"] is True


def test_not_found(client):
    assert client.get("/weeks/2020-01-06/tracts").status_code == 404
    assert client.get("/weeks/june/tracts").status_code == 404
    assert client.get("/weeks/latest/tracts/123").status_code == 404
    assert client.get("/weeks/latest/tracts/06037999999").status_code == 404
    assert client.get("/trends?geoids=abc").status_code == 404
    assert client.get("/search?q=a").status_code == 422  # too short


def test_watchlist(client):
    items = client.get("/weeks/latest/watchlist").json()["tracts"]
    assert [i["geoid"] for i in items] == [C, D]
    # Only terms that push the probability up
    assert [t["feature"] for t in items[0]["ml_contributions"]["top"]] == [
        "access_z", "lst_z",
    ]  # fmt: skip
    only = client.get("/weeks/latest/watchlist?min_prob=0.5").json()["tracts"]
    assert [i["geoid"] for i in only] == [C]


def test_equity(client):
    groups = {
        g["group"]: g for g in client.get("/weeks/latest/equity").json()["groups"]
    }
    assert groups["all"]["share_in_critical"] == 0.3
    # income terciles over 4 tracts: low = A, B; middle = C; high = D
    assert groups["income_low"]["share_in_critical"] == 1.0
    assert groups["income_low"]["ratio_to_all"] == 3.33
    assert groups["income_high"]["in_critical"] == 0
    # limited English: 40% of 500 + 40% of 1000 households, all in A and B
    assert groups["limited_english"]["total"] == 600
    assert groups["limited_english"]["share_in_critical"] == 1.0
    assert groups["latino"]["unit"] == "people"


def test_validation(client):
    v = client.get("/weeks/latest/validation").json()
    assert v["drift_checks"][0]["metric"] == "lst_coverage"
    assert v["ces_agreement"] is None  # too few tracts to correlate
    last = v["weekly"][-1]
    assert last["kriging_loo_mae"] == 3.0
    assert last["lst_coverage"] == 0.75  # D has no LST
    assert (last["critical_high"], last["critical_medium"]) == (1, 1)
    assert v["weekly"][0]["critical_high"] == 0  # no CRITICAL on 06-08


def test_trends(client):
    pts = client.get(f"/trends?geoids={A},{B}&first=2026-06-15").json()
    assert len(pts) == 4
    assert {p["geoid"] for p in pts} == {A, B}
    pts = client.get(f"/trends?geoids={A}&geoids={C}").json()
    assert len(pts) == 6


def test_hexes_and_outlines(client):
    h = client.get("/hexes").json()
    assert len(h["h3_index"]) == len(h["geoid"]) == len(h["population"]) == 4
    r = client.get("/outlines")
    assert r.headers["content-type"].startswith("application/geo+json")
    fc = r.json()
    assert fc["type"] == "FeatureCollection"
    assert fc["features"][0]["properties"]["geoid"] == A


def test_search(client):
    hits = client.get("/search?q=glen").json()
    # "City of Glendale" is the same tracts as city Glendale: listed once
    assert hits == [{"kind": "city", "label": "Glendale", "geoids": [A, B]}]
    assert client.get("/search?q=watts").json()[0]["label"] == "Los Angeles - Watts"
    assert client.get("/search?q=valley").json()[0]["kind"] == "region"
    assert client.get("/search?q=000300").json()[0]["geoids"] == [C]


class _Resp:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def json(self):
        return self.payload


def _match(geoid):
    return {
        "result": {
            "addressMatches": [
                {
                    "matchedAddress": "1 MAIN ST, GLENDALE, CA, 91201",
                    "coordinates": {"x": -118.25, "y": 34.15},
                    "geographies": {"Census Tracts": [{"GEOID": geoid}]},
                }
            ]
        }
    }


def test_geocode(client, monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(_match(A)))
    r = client.get("/geocode?address=1 Main St, Glendale, CA")
    assert r.status_code == 200 and r.json()["geoid"] == A

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(_match("06059001000")))
    assert client.get("/geocode?address=1 Main St, Irvine, CA").status_code == 404

    monkeypatch.setattr(
        requests, "get", lambda *a, **k: _Resp({"result": {"addressMatches": []}})
    )
    assert client.get("/geocode?address=nowhere at all").status_code == 404

    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp({}, status=503))
    assert client.get("/geocode?address=1 Main St, Glendale").status_code == 502
