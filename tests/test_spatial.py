"""
Tests for the spatial layer on small synthetic inputs: Kriging's variogram
pieces, the compound score's direction and classes, Gi* hot spots, and the
two 2SFCA steps. No warehouse, network, or Earth Engine.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from src.spatial import compound_score, healthcare_access, hotspots, kriging

PERCENTILES = {"burden_percentiles": {"elevated": 50, "high_risk": 75, "critical": 90}}


# ---------------------------------------------------------------------------
# Kriging
# ---------------------------------------------------------------------------

def test_spherical_model_starts_at_nugget_and_levels_at_sill():
    h = np.array([0.0, 50_000.0, 100_000.0, 200_000.0])
    gamma = kriging._spherical(h, psill=2.0, range_m=100_000.0, nugget=0.5)
    assert gamma[0] == pytest.approx(0.5)
    assert 0.5 < gamma[1] < 2.5
    assert gamma[2] == pytest.approx(2.5)
    assert gamma[3] == pytest.approx(2.5)


def test_nugget_floor_is_zero_for_full_weeks_and_positive_for_sparse_samplers():
    full = pd.DataFrame({"reading_count": [7, 7], "pm25_std": [2.0, 3.0]})
    assert kriging._nugget_floor(full) == 0.0

    # A 1-in-3-day sampler: 2 readings, std 3 -> 9/2 * (1 - 2/7) = 3.214...
    sparse = pd.DataFrame({"reading_count": [2], "pm25_std": [3.0]})
    assert kriging._nugget_floor(sparse) == pytest.approx(9 / 2 * (1 - 2 / 7))


def test_empirical_variogram_uses_equal_count_bins_within_half_max_distance():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 100_000, 30), rng.uniform(0, 100_000, 30)
    z = rng.normal(9, 1, 30)
    bins = kriging._empirical_variogram(x, y, z, min_pairs_per_bin=30)

    coords = np.column_stack([x, y])
    max_pair = max(np.hypot(*(coords[i] - coords[j])) for i in range(30) for j in range(i + 1, 30))
    assert (bins["n_pairs"] >= 30).all()
    assert bins["lag_m"].max() <= max_pair / 2
    assert bins["lag_m"].is_monotonic_increasing


def test_fit_variogram_recovers_known_parameters_and_respects_nugget_floor():
    lags = np.linspace(5_000, 90_000, 8)
    truth = {"psill": 2.5, "range": 80_000.0, "nugget": 0.3}
    bins = pd.DataFrame({
        "lag_m": lags,
        "semivariance": kriging._spherical(lags, truth["psill"], truth["range"], truth["nugget"]),
        "n_pairs": 40,
    })
    fit = kriging._fit_variogram(bins)
    assert fit["psill"] == pytest.approx(truth["psill"], rel=0.05)
    assert fit["range"] == pytest.approx(truth["range"], rel=0.05)

    floored = kriging._fit_variogram(bins, nugget_floor=0.6)
    assert floored["nugget"] >= 0.6 - 1e-9


# ---------------------------------------------------------------------------
# Compound score
# ---------------------------------------------------------------------------

def _signals(n=100, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "geoid": [f"t{i:03d}" for i in range(n)],
        "pm25_kriged": rng.normal(9, 0.5, n),
        "lst_zscore": rng.normal(0, 1, n),
        "access_score_er": rng.uniform(0.5, 3.0, n),
    })


def test_better_access_lowers_burden():
    df = _signals()
    df.loc[0, ["pm25_kriged", "lst_zscore"]] = df.loc[1, ["pm25_kriged", "lst_zscore"]].values
    df.loc[0, "access_score_er"], df.loc[1, "access_score_er"] = 3.0, 0.5   # same exposure, 0 has better access
    out = compound_score._build_composite(df, PERCENTILES).set_index("geoid")
    assert out.loc["t000", "raw_burden"] < out.loc["t001", "raw_burden"]
    assert out.loc["t000", "access_z"] < 0 < out.loc["t001", "access_z"]


def test_classes_follow_city_percentiles():
    out = compound_score._build_composite(_signals(), PERCENTILES)
    counts = out["burden_class"].value_counts()
    assert counts["CRITICAL"] == 11       # percentiles 90.0 to 100.0 inclusive
    assert counts["HIGH"] == 15
    assert counts["ELEVATED"] == 25
    assert counts["LOW"] == 49
    assert out["compound_score"].between(0, 1).all()


def test_missing_signals_are_filled_with_the_city_median():
    df = _signals(10)
    df.loc[3, "access_score_er"] = np.nan
    filled, n = compound_score._impute_missing(df)
    assert n == 1
    assert filled.loc[3, "access_score_er"] == pytest.approx(df["access_score_er"].median())


def test_validation_fails_if_burden_rises_with_access():
    signals = _signals()
    scores = compound_score._build_composite(signals, PERCENTILES)
    compound_score._validate_output(scores, signals)  # correct direction passes

    flipped = scores.copy()
    flipped["compound_score"] = signals["access_score_er"].rank(pct=True).values  # burden up with access
    with pytest.raises(ValueError, match="rises with access"):
        compound_score._validate_output(flipped, signals)


# ---------------------------------------------------------------------------
# Hot spots
# ---------------------------------------------------------------------------

def test_gi_star_finds_a_high_cluster_in_the_corner():
    side = 10
    cells = [box(c, r, c + 1, r + 1) for r in range(side) for c in range(side)]
    grid = gpd.GeoDataFrame({"geoid": [f"g{i:03d}" for i in range(side * side)]}, geometry=cells)
    rng = np.random.default_rng(3)
    scores = rng.normal(0.3, 0.05, side * side)
    corner = [r * side + c for r in range(3) for c in range(3)]
    scores[corner] = 0.9

    W = hotspots._build_spatial_weights(grid)
    out = hotspots._compute_gi_star(
        scores, W, grid["geoid"].values,
        {"hotspot_significance": {"standard": 0.05, "high": 0.01}},
    ).set_index("geoid")

    assert out.loc["g011", "hotspot_class"] == "HOTSPOT_99"   # center of the corner block
    far_corner = out.loc["g099", "hotspot_class"]
    assert not far_corner.startswith("HOTSPOT")


# ---------------------------------------------------------------------------
# 2SFCA
# ---------------------------------------------------------------------------

def test_two_step_floating_catchment_by_hand():
    # Hospital H1 (100 beds) reaches tracts A and B; H2 (50 beds) reaches only B.
    # C is reachable but has no hospital in reach (a desert); D is unreachable.
    travel = pd.DataFrame({
        "geoid":       ["A", "B", "B"],
        "hospital_id": ["H1", "H1", "H2"],
        "decay_weight": [1.0, 0.5, 1.0],
    })
    hospitals_df = pd.DataFrame({"hospital_id": ["H1", "H2"], "beds": [100, 50], "in_city": [True, True]})
    tracts = pd.DataFrame({
        "geoid": ["A", "B", "C", "D"],
        "population": [1000, 2000, 500, 800],
        "in_city": True,
        "reachable": [True, True, True, False],
        "snap_distance_m": 10.0,
    })

    ratios = healthcare_access._compute_hospital_ratios(travel, hospitals_df, tracts).set_index("hospital_id")
    # H1 demand = 1000*1 + 2000*0.5 = 2000 -> 0.05 beds per resident; H2 = 50/2000 = 0.025
    assert ratios.loc["H1", "ratio"] == pytest.approx(0.05)
    assert ratios.loc["H2", "ratio"] == pytest.approx(0.025)

    access = healthcare_access._compute_tract_access(travel, ratios.reset_index(), tracts).set_index("geoid")
    assert access.loc["A", "access_score_er"] == pytest.approx(0.05 * 1000)            # 50 per 1,000
    assert access.loc["B", "access_score_er"] == pytest.approx((0.05 * 0.5 + 0.025) * 1000)
    assert access.loc["C", "access_score_er"] == 0                                      # desert
    assert not access.loc["C", "er_in_catchment"]
    assert np.isnan(access.loc["D", "access_score_er"])                                 # unreachable


# ---------------------------------------------------------------------------
# Healthcare access: reuse across weeks
# ---------------------------------------------------------------------------

from datetime import datetime  # noqa: E402

HC_CONFIG = {
    "full_fips": "06037",
    "city_name": "Los Angeles",
    "healthcare": {"er_catchment_minutes": 30, "buffer_km": 30},
}


def _demand():
    return pd.DataFrame({
        "geoid": ["A", "B"], "centroid_lat": [34.0, 34.1], "centroid_lon": [-118.2, -118.3],
        "population": [1000, 2000], "in_city": [True, True],
    })


def _supply(beds=100):
    return pd.DataFrame({
        "hospital_id": ["H1"], "name": ["General"], "beds": [beds],
        "latitude": [34.05], "longitude": [-118.25], "in_city": [True],
    })


def test_fingerprint_changes_only_when_inputs_change(tmp_path, monkeypatch):
    monkeypatch.setattr(healthcare_access, "NETWORK_CACHE_DIR", tmp_path)
    base = healthcare_access._inputs_fingerprint(_demand(), _supply(), HC_CONFIG)
    assert base == healthcare_access._inputs_fingerprint(_demand().iloc[::-1], _supply(), HC_CONFIG)  # row order
    assert base != healthcare_access._inputs_fingerprint(_demand(), _supply(beds=120), HC_CONFIG)
    changed = {**HC_CONFIG, "healthcare": {**HC_CONFIG["healthcare"], "er_catchment_minutes": 45}}
    assert base != healthcare_access._inputs_fingerprint(_demand(), _supply(), changed)


def test_unchanged_inputs_reuse_last_week_without_the_road_network(tmp_path, monkeypatch):
    monkeypatch.setattr(healthcare_access, "NETWORK_CACHE_DIR", tmp_path)
    fingerprint = healthcare_access._inputs_fingerprint(_demand(), _supply(), HC_CONFIG)
    previous = pd.DataFrame({
        "geoid": ["A", "B"], "access_score_er": [2.0, 1.5], "scored": [True, True],
        "er_in_catchment": [True, True], "week_start_date": [pd.Timestamp("2026-06-15").date()] * 2,
        "city_fips": ["06037"] * 2,
    })

    def read_query(query, config):
        if "healthcare_access_runs" in query:
            assert fingerprint in query
            return pd.DataFrame({"week_start_date": [pd.Timestamp("2026-06-15")]})
        return previous.copy()

    written = {}
    monkeypatch.setattr(healthcare_access, "_load_demand_tracts", lambda c: _demand())
    monkeypatch.setattr(healthcare_access, "_load_hospital_data", lambda c: _supply())
    monkeypatch.setattr(healthcare_access, "table_exists", lambda name, c: True)
    monkeypatch.setattr(healthcare_access, "read_query", read_query)
    monkeypatch.setattr(healthcare_access, "_validate_output", lambda df, minutes: None)
    monkeypatch.setattr(healthcare_access, "write_dataframe", lambda df, table, c, **kw: written.setdefault(table, df))

    def no_network(*a, **k):
        raise AssertionError("road network should not be loaded when inputs are unchanged")

    monkeypatch.setattr(healthcare_access, "_get_road_network", no_network)

    out = healthcare_access.compute_2sfca(HC_CONFIG, week_start=datetime(2026, 6, 22))
    assert (out["week_start_date"] == pd.Timestamp("2026-06-22").date()).all()
    assert list(out["access_score_er"]) == [2.0, 1.5]
    runs = written["healthcare_access_runs"]
    assert runs["computed_from"].iloc[0] == "2026-06-15"
    assert runs["inputs_hash"].iloc[0] == fingerprint
