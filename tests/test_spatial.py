"""
Tests for the spatial layer on small synthetic inputs: Kriging's variogram
pieces, the compound score's direction and classes, Gi* hot spots, and the
two 2SFCA steps. No warehouse, network, or Earth Engine.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
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
    max_pair = max(
        np.hypot(*(coords[i] - coords[j])) for i in range(30) for j in range(i + 1, 30)
    )
    assert (bins["n_pairs"] >= 30).all()
    assert bins["lag_m"].max() <= max_pair / 2
    assert bins["lag_m"].is_monotonic_increasing


def test_fit_variogram_recovers_known_parameters_and_respects_nugget_floor():
    lags = np.linspace(5_000, 90_000, 8)
    truth = {"psill": 2.5, "range": 80_000.0, "nugget": 0.3}
    bins = pd.DataFrame(
        {
            "lag_m": lags,
            "semivariance": kriging._spherical(
                lags, truth["psill"], truth["range"], truth["nugget"]
            ),
            "n_pairs": 40,
        }
    )
    fit = kriging._fit_variogram(bins)
    assert fit["psill"] == pytest.approx(truth["psill"], rel=0.05)
    assert fit["range"] == pytest.approx(truth["range"], rel=0.05)

    floored = kriging._fit_variogram(bins, nugget_floor=0.6)
    assert floored["nugget"] >= 0.6 - 1e-9


# ---------------------------------------------------------------------------
# Compound score
# ---------------------------------------------------------------------------


_ORIGINAL_LOAD_VULNERABILITY = compound_score._load_vulnerability


@pytest.fixture(autouse=True)
def _no_acs_query(monkeypatch):
    # _load_signals also reads acs_vulnerability. Tests that fake read_query
    # for the weekly signals get a neutral vulnerability instead.
    monkeypatch.setattr(
        compound_score,
        "_load_vulnerability",
        lambda config: pd.DataFrame(
            {"geoid": [f"t{i:03d}" for i in range(200)], "vulnerability": 0.0}
        ),
    )


def _signals(n=100, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "geoid": [f"t{i:03d}" for i in range(n)],
            "pm25_kriged": rng.normal(9, 0.5, n),
            "lst_zscore": rng.normal(0, 1, n),
            "access_score_er": rng.uniform(0.5, 3.0, n),
            "vulnerability": rng.normal(0, 1, n),
        }
    )


def _weekly(n):
    """What the weekly signals query returns: no vulnerability column."""
    return _signals(n).drop(columns="vulnerability")


def test_better_access_lowers_burden():
    df = _signals()
    df.loc[0, ["pm25_kriged", "lst_zscore"]] = df.loc[
        1, ["pm25_kriged", "lst_zscore"]
    ].values
    df.loc[0, "access_score_er"], df.loc[1, "access_score_er"] = (
        3.0,
        0.5,
    )  # same exposure, 0 has better access
    out = compound_score._build_composite(df, PERCENTILES).set_index("geoid")
    assert out.loc["t000", "raw_burden"] < out.loc["t001", "raw_burden"]
    assert out.loc["t000", "access_z"] < 0 < out.loc["t001", "access_z"]


def test_classes_follow_city_percentiles():
    out = compound_score._build_composite(_signals(), PERCENTILES)
    counts = out["burden_class"].value_counts()
    assert counts["CRITICAL"] == 11  # percentiles 90.0 to 100.0 inclusive
    assert counts["HIGH"] == 15
    assert counts["ELEVATED"] == 25
    assert counts["LOW"] == 49
    assert out["compound_score"].between(0, 1).all()


def test_missing_signals_are_filled_with_the_city_median():
    df = _signals(10)
    df.loc[3, "access_score_er"] = np.nan
    filled, n = compound_score._impute_missing(df)
    assert n == 1
    assert filled.loc[3, "access_score_er"] == pytest.approx(
        df["access_score_er"].median()
    )


def test_validation_fails_if_the_access_sign_is_flipped():
    signals = _signals()
    scores = compound_score._build_composite(signals, PERCENTILES)
    compound_score._validate_output(scores, signals)  # correct direction passes

    flipped = scores.copy()
    flipped["access_z"] = -flipped["access_z"]  # access entering positively
    with pytest.raises(ValueError, match="does not fall as access rises"):
        compound_score._validate_output(flipped, signals)


def test_validation_passes_when_exposure_overlaps_good_access():
    # LA fall and winter weeks: the best-served tracts are also the hottest and
    # most polluted, so total burden barely tracks access (week of 2025-10-27:
    # rho +0.007). The access sign is still right, so validation must pass.
    signals = _signals()
    rank = signals["access_score_er"].rank(pct=True)
    signals["pm25_kriged"] = 9 + 3 * rank
    signals["lst_zscore"] = 3 * rank
    scores = compound_score._build_composite(signals, PERCENTILES)
    total = scores["compound_score"].corr(signals["access_score_er"], method="spearman")
    assert total > -0.5  # total burden does not fall clearly with access
    compound_score._validate_output(scores, signals)


def test_signals_join_healthcare_access_for_the_same_week(monkeypatch):
    # Without the week filter every stored week of access matched each tract,
    # tripling the rows once a second week existed.
    queries = []

    def read_query(query, config):
        queries.append(query)
        return _weekly(5).assign(pm25_weeks=1)

    monkeypatch.setattr(compound_score, "read_query", read_query)
    compound_score._load_signals({"full_fips": "06037"}, "2025-08-04")
    joined = " ".join(queries[0].split())
    assert "a.week_start_date = '2025-08-04'" in joined
    assert "l.week_start_date = '2025-08-04'" in joined


def test_pm25_window_starts_three_weeks_back_for_four_weeks():
    assert compound_score._window_start("2026-06-22", 4) == "2026-06-01"
    assert compound_score._window_start("2026-06-22", 1) == "2026-06-22"
    # Crosses a year boundary
    assert compound_score._window_start("2026-01-12", 4) == "2025-12-22"


def test_pm25_is_averaged_over_the_window(monkeypatch):
    queries = []

    def read_query(query, config):
        queries.append(query)
        return _weekly(5).assign(pm25_weeks=4)

    monkeypatch.setattr(compound_score, "read_query", read_query)
    out = compound_score._load_signals(
        {"full_fips": "06037"}, "2026-06-22", pm25_window_weeks=4
    )
    joined = " ".join(queries[0].split())
    assert "AVG(pm25_kriged)" in joined
    assert "BETWEEN '2026-06-01' AND '2026-06-22'" in joined
    # The helper column does not leak into the signals
    assert list(out.columns) == [
        "geoid",
        "pm25_kriged",
        "lst_zscore",
        "access_score_er",
        "vulnerability",
    ]


def test_short_pm25_window_logs_a_warning(monkeypatch):
    # The project logger does not propagate, so capture its calls directly
    warnings = []
    monkeypatch.setattr(
        compound_score,
        "read_query",
        lambda q, c: _weekly(5).assign(pm25_weeks=2),
    )
    monkeypatch.setattr(compound_score.logger, "warning", warnings.append)
    # start of the record: no window week has monitor data without Kriging
    monkeypatch.setattr(compound_score, "_check_window_complete", lambda *a: None)
    compound_score._load_signals(
        {"full_fips": "06037"}, "2025-06-30", pm25_window_weeks=4
    )
    assert any("2 of 4 weeks" in w for w in warnings)


def test_latest_good_week_skips_cloudy_weeks():
    counts = pd.DataFrame(
        {"week": ["2025-11-10", "2025-11-17", "2025-11-24"], "n": [95, 96, 9]}
    )
    assert compound_score._latest_good_week(counts, 100, 0.8) == "2025-11-17"
    assert compound_score._latest_good_week(counts.iloc[2:], 100, 0.8) is None


def test_cloudy_week_reuses_last_good_week_lst(monkeypatch):
    # 2025-12-01 had LST for 9% of tracts: the median fill gave the rest an
    # average heat score. The week should reuse 2025-11-17 instead.
    week = _weekly(10).assign(pm25_weeks=4)
    week.loc[1:, "lst_zscore"] = np.nan
    earlier = pd.DataFrame(
        {"geoid": week["geoid"], "lst_zscore": np.arange(10, dtype=float)}
    )
    counts = pd.DataFrame({"week": ["2025-11-17", "2025-11-24"], "n": [10, 1]})
    queries = []

    def read_query(query, config):
        queries.append(" ".join(query.split()))
        if "COUNT(lst_zscore)" in query:
            return counts
        if "SELECT geoid, lst_zscore" in query:
            return earlier
        return week

    monkeypatch.setattr(compound_score, "read_query", read_query)
    monkeypatch.setattr(compound_score.logger, "warning", lambda m: None)
    out = compound_score._load_signals(
        {"full_fips": "06037", "gee": {"min_coverage": 0.8}},
        "2025-12-01",
        pm25_window_weeks=4,
    )
    assert out["lst_zscore"].tolist() == list(range(10))
    assert "week_start_date >= '2025-10-06'" in queries[1]
    assert "week_start_date = '2025-11-17'" in queries[2]


def test_well_covered_week_keeps_its_own_lst(monkeypatch):
    queries = []

    def read_query(query, config):
        queries.append(query)
        return _weekly(10).assign(pm25_weeks=4)

    monkeypatch.setattr(compound_score, "read_query", read_query)
    out = compound_score._load_signals(
        {"full_fips": "06037"}, "2026-06-22", pm25_window_weeks=4
    )
    assert len(queries) == 1
    assert out["lst_zscore"].notna().all()


def test_no_good_week_falls_back_to_median_with_a_warning(monkeypatch):
    week = _weekly(10).assign(pm25_weeks=4)
    week.loc[1:, "lst_zscore"] = np.nan
    warnings = []

    def read_query(query, config):
        if "COUNT(lst_zscore)" in query:
            return pd.DataFrame({"week": ["2025-11-24"], "n": [1]})
        return week

    monkeypatch.setattr(compound_score, "read_query", read_query)
    monkeypatch.setattr(compound_score.logger, "warning", warnings.append)
    out = compound_score._load_signals(
        {"full_fips": "06037"}, "2025-12-01", pm25_window_weeks=4
    )
    assert out["lst_zscore"].isna().sum() == 9
    assert any("city median" in w for w in warnings)


def test_shrinkage_moves_noisy_shares_more():
    # Real spread: shares 10 and 30 (variance 100). Tract a is precise
    # (SE 1), tract d is noisy (SE 20): d moves most of the way to the mean.
    pct = pd.Series([10.0, 30.0, 10.0, 30.0])
    moe = pd.Series([1.0, 1.0, 1.0, 20.0]) * compound_score.Z90
    out = compound_score.shrink_share(pct, moe)
    real = 100 - (1 + 1 + 1 + 400) / 4  # observed variance - mean SE^2 = -0.75 -> 0
    assert real < 0
    # No real spread left after removing noise: every share goes to the mean
    assert np.allclose(out, 20.0)

    pct = pd.Series([0.0, 40.0, 0.0, 40.0, 20.0])
    moe = pd.Series([2.0, 2.0, 2.0, 2.0, 30.0]) * compound_score.Z90
    out = compound_score.shrink_share(pct, moe)
    mean, real = 20.0, pct.var(ddof=0) - (4 * 4 + 900) / 5
    b_precise = real / (real + 4)
    assert out[0] == pytest.approx(mean + b_precise * (0 - mean))
    assert abs(out[0] - pct[0]) < 1  # precise share barely moves


def test_shrinkage_keeps_small_precise_shares():
    # 1% uninsured measured to +/- 0.5 points: a CV rule (CV 0.30) would be
    # borderline, shrinkage leaves it near 1%
    pct = pd.Series([1.0, 5.0, 9.0, 13.0, 17.0])
    moe = pd.Series([0.5, 1.0, 1.0, 1.0, 1.0])
    out = compound_score.shrink_share(pct, moe)
    assert out[0] == pytest.approx(1.0, abs=0.05)


def test_vulnerability_score_averages_indicator_z_scores():
    n = 6
    acs = pd.DataFrame({"geoid": [f"g{i}" for i in range(n)]})
    for name in compound_score.VULNERABILITY_INDICATORS:
        acs[f"{name}_pct"] = np.arange(n, dtype=float)  # same order everywhere
        acs[f"{name}_moe"] = 0.0  # exact: no shrinkage
    acs.loc[5, [f"{x}_pct" for x in compound_score.VULNERABILITY_INDICATORS]] = np.nan
    out = compound_score.vulnerability_score(acs).set_index("geoid")["vulnerability"]
    assert out.iloc[:5].is_monotonic_increasing
    assert np.isnan(out["g5"])  # no residents: left for the median fill
    # One missing indicator counts as average (z = 0), not as a gap
    acs.loc[0, "uninsured_pct"] = np.nan
    one_missing = compound_score.vulnerability_score(acs).set_index("geoid")
    assert one_missing.loc["g0", "vulnerability"] > out["g0"]


def test_vulnerability_enters_with_a_positive_sign():
    df = _signals()
    df.loc[0, "vulnerability"] = 5.0
    out = compound_score._build_composite(df, PERCENTILES).set_index("geoid")
    assert out.loc["t000", "vulnerability_z"] > 0
    assert out.loc["t000", "burden_class"] in {"HIGH", "CRITICAL"}


def test_missing_acs_table_is_reported(monkeypatch):
    monkeypatch.setattr(compound_score, "read_query", lambda q, c: pd.DataFrame())
    with pytest.raises(ValueError, match="acs_vulnerability"):
        _ORIGINAL_LOAD_VULNERABILITY({"full_fips": "06037"})


# ---------------------------------------------------------------------------
# Hot spots
# ---------------------------------------------------------------------------


def test_gi_star_finds_a_high_cluster_in_the_corner():
    side = 10
    cells = [box(c, r, c + 1, r + 1) for r in range(side) for c in range(side)]
    grid = gpd.GeoDataFrame(
        {"geoid": [f"g{i:03d}" for i in range(side * side)]}, geometry=cells
    )
    rng = np.random.default_rng(3)
    scores = rng.normal(0.3, 0.05, side * side)
    corner = [r * side + c for r in range(3) for c in range(3)]
    scores[corner] = 0.9

    W = hotspots._build_spatial_weights(grid)
    out = hotspots._compute_gi_star(
        scores,
        W,
        grid["geoid"].values,
        {"hotspot_significance": {"standard": 0.05, "high": 0.01}},
    ).set_index("geoid")

    assert (
        out.loc["g011", "hotspot_class"] == "HOTSPOT_99"
    )  # center of the corner block
    far_corner = out.loc["g099", "hotspot_class"]
    assert not far_corner.startswith("HOTSPOT")


def test_gi_star_classes_use_two_tailed_cutoffs():
    # A tract at z = 1.8 was HOTSPOT_95 under the one-tailed p (cutoff 1.645).
    # Two-tailed, 95% needs |z| > 1.96.
    side = 10
    cells = [box(c, r, c + 1, r + 1) for r in range(side) for c in range(side)]
    grid = gpd.GeoDataFrame(
        {"geoid": [f"g{i:03d}" for i in range(side * side)]}, geometry=cells
    )
    scores = np.random.default_rng(5).normal(0, 1, side * side)
    W = hotspots._build_spatial_weights(grid)
    out = hotspots._compute_gi_star(
        scores,
        W,
        grid["geoid"].values,
        {"hotspot_significance": {"standard": 0.05, "high": 0.01}},
    )
    z = out["gi_star_zscore"].abs()
    sig = out["hotspot_class"] != "NOT_SIGNIFICANT"
    assert (z[sig] > 1.959).all()
    assert (z[~sig] <= 1.96).all()
    # Stored p-value is two-tailed: p = 0.05 at |z| = 1.96
    assert np.allclose(out["gi_star_pvalue"], 2 * norm.sf(z), atol=1e-4)


def test_gi_star_weights_include_self():
    grid = gpd.GeoDataFrame(
        {"geoid": ["a", "b", "c"]},
        geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(2, 0, 3, 1)],
    )
    W = hotspots._gi_star_weights(hotspots._build_spatial_weights(grid))
    # Middle tract: itself and its two neighbors, a third each
    assert W.neighbors[1] and 1 in W.neighbors[1]
    assert np.allclose(sorted(W.weights[1]), [1 / 3] * 3)


def test_fdr_count_by_hand():
    # m = 4, alpha 0.05: cutoffs 0.0125, 0.025, 0.0375, 0.05
    assert hotspots.fdr_count(np.array([0.01, 0.02, 0.04, 0.9]), 0.05) == 2
    assert hotspots.fdr_count(np.array([0.2, 0.3]), 0.05) == 0
    assert hotspots.fdr_count(np.array([]), 0.05) == 0


# ---------------------------------------------------------------------------
# 2SFCA
# ---------------------------------------------------------------------------


def test_two_step_floating_catchment_by_hand():
    # Hospital H1 (100 beds) reaches tracts A and B; H2 (50 beds) reaches only B.
    # C is reachable but has no hospital in reach (a desert); D is unreachable.
    travel = pd.DataFrame(
        {
            "geoid": ["A", "B", "B"],
            "hospital_id": ["H1", "H1", "H2"],
            "decay_weight": [1.0, 0.5, 1.0],
        }
    )
    hospitals_df = pd.DataFrame(
        {"hospital_id": ["H1", "H2"], "beds": [100, 50], "in_city": [True, True]}
    )
    tracts = pd.DataFrame(
        {
            "geoid": ["A", "B", "C", "D"],
            "population": [1000, 2000, 500, 800],
            "in_city": True,
            "reachable": [True, True, True, False],
            "snap_distance_m": 10.0,
        }
    )

    ratios = healthcare_access._compute_hospital_ratios(
        travel, hospitals_df, tracts
    ).set_index("hospital_id")
    # H1 demand = 1000*1 + 2000*0.5 = 2000 -> 0.05 beds per resident; H2 = 50/2000 = 0.025
    assert ratios.loc["H1", "ratio"] == pytest.approx(0.05)
    assert ratios.loc["H2", "ratio"] == pytest.approx(0.025)

    access = healthcare_access._compute_tract_access(
        travel, ratios.reset_index(), tracts
    ).set_index("geoid")
    assert access.loc["A", "access_score_er"] == pytest.approx(
        0.05 * 1000
    )  # 50 per 1,000
    assert access.loc["B", "access_score_er"] == pytest.approx(
        (0.05 * 0.5 + 0.025) * 1000
    )
    assert access.loc["C", "access_score_er"] == 0  # desert
    assert not access.loc["C", "er_in_catchment"]
    assert np.isnan(access.loc["D", "access_score_er"])  # unreachable


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
    return pd.DataFrame(
        {
            "geoid": ["A", "B"],
            "centroid_lat": [34.0, 34.1],
            "centroid_lon": [-118.2, -118.3],
            "population": [1000, 2000],
            "in_city": [True, True],
        }
    )


def _supply(beds=100):
    return pd.DataFrame(
        {
            "hospital_id": ["H1"],
            "name": ["General"],
            "beds": [beds],
            "latitude": [34.05],
            "longitude": [-118.25],
            "in_city": [True],
        }
    )


def test_fingerprint_changes_only_when_inputs_change(tmp_path, monkeypatch):
    monkeypatch.setattr(healthcare_access, "NETWORK_CACHE_DIR", tmp_path)
    base = healthcare_access._inputs_fingerprint(_demand(), _supply(), HC_CONFIG)
    assert base == healthcare_access._inputs_fingerprint(
        _demand().iloc[::-1], _supply(), HC_CONFIG
    )  # row order
    assert base != healthcare_access._inputs_fingerprint(
        _demand(), _supply(beds=120), HC_CONFIG
    )
    changed = {
        **HC_CONFIG,
        "healthcare": {**HC_CONFIG["healthcare"], "er_catchment_minutes": 45},
    }
    assert base != healthcare_access._inputs_fingerprint(_demand(), _supply(), changed)


def test_unchanged_inputs_reuse_last_week_without_the_road_network(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(healthcare_access, "NETWORK_CACHE_DIR", tmp_path)
    fingerprint = healthcare_access._inputs_fingerprint(_demand(), _supply(), HC_CONFIG)
    previous = pd.DataFrame(
        {
            "geoid": ["A", "B"],
            "access_score_er": [2.0, 1.5],
            "scored": [True, True],
            "er_in_catchment": [True, True],
            "week_start_date": [pd.Timestamp("2026-06-15").date()] * 2,
            "city_fips": ["06037"] * 2,
        }
    )

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
    monkeypatch.setattr(
        healthcare_access,
        "write_dataframe",
        lambda df, table, c, **kw: written.setdefault(table, df),
    )

    def no_network(*a, **k):
        raise AssertionError(
            "road network should not be loaded when inputs are unchanged"
        )

    monkeypatch.setattr(healthcare_access, "_get_road_network", no_network)

    out = healthcare_access.compute_2sfca(HC_CONFIG, week_start=datetime(2026, 6, 22))
    assert (out["week_start_date"] == pd.Timestamp("2026-06-22").date()).all()
    assert list(out["access_score_er"]) == [2.0, 1.5]
    runs = written["healthcare_access_runs"]
    assert runs["computed_from"].iloc[0] == "2026-06-15"
    assert runs["inputs_hash"].iloc[0] == fingerprint


def test_runs_record_stores_computed_from_as_text_even_when_empty(monkeypatch):
    # The first recorded week is computed, not copied (computed_from None).
    # DuckDB typed that all-None column INTEGER, so copied weeks could not
    # store their source date. The column must be text from the first row.
    import duckdb

    written = {}
    monkeypatch.setattr(
        healthcare_access,
        "write_dataframe",
        lambda df, table, c, **kw: written.setdefault(table, df),
    )
    healthcare_access._record_run(HC_CONFIG, datetime(2026, 6, 22), "abc123", None)
    # duckdb reads the local variable `runs` by name
    runs = written["healthcare_access_runs"]  # noqa: F841
    types = {
        row[0]: row[1] for row in duckdb.sql("DESCRIBE SELECT * FROM runs").fetchall()
    }
    assert types["computed_from"] == "VARCHAR"
