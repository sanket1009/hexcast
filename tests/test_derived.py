"""Tests for the Phase 7 dashboard tables: confidence, chronic, hex grid."""

import numpy as np
import pandas as pd
import pytest

from src.spatial import chronic, confidence, hex_grid

CUTOFFS = {"elevated": 50, "high_risk": 75, "critical": 90}


def _toy_points(seed=0):
    rng = np.random.default_rng(seed)
    monitors = rng.uniform(0, 50_000, size=(8, 2))
    tracts = rng.uniform(0, 50_000, size=(30, 2))
    values = rng.normal(10, 1, size=8)
    return monitors, tracts, values


def test_error_variance_matches_pykrige():
    from pykrige.ok import OrdinaryKriging

    monitors, tracts, values = _toy_points()
    nugget, psill, range_m = 0.3, 2.5, 40_000.0
    ok = OrdinaryKriging(
        monitors[:, 0],
        monitors[:, 1],
        values,
        variogram_model="spherical",
        variogram_parameters={"psill": psill, "range": range_m, "nugget": nugget},
    )
    _, pykrige_var = ok.execute("points", tracts[:, 0], tracts[:, 1])
    sigma = confidence.error_covariance(
        monitors, tracts, nugget, psill, range_m, include_nugget=True
    )
    np.testing.assert_allclose(np.diag(sigma), pykrige_var, rtol=1e-6, atol=1e-8)


def test_leaving_out_nugget_lowers_each_variance_by_the_nugget():
    monitors, tracts, _ = _toy_points()
    with_n = confidence.error_covariance(monitors, tracts, 0.3, 2.5, 40_000.0, True)
    without = confidence.error_covariance(monitors, tracts, 0.3, 2.5, 40_000.0)
    np.testing.assert_allclose(np.diag(with_n) - np.diag(without), 0.3)


def test_symmetric_factor_reproduces_covariance_and_is_symmetric():
    monitors, tracts, _ = _toy_points()
    sigma = confidence.error_covariance(monitors, tracts, 0.3, 2.5, 40_000.0)
    f = confidence.covariance_factor(sigma)
    np.testing.assert_allclose(f @ f.T, sigma, atol=1e-8)
    np.testing.assert_allclose(f, f.T, atol=1e-10)


def _scores(n=40, seed=1):
    rng = np.random.default_rng(seed)
    comps = rng.normal(size=(n, 4))
    pm = 8 + comps[:, 0]  # PM2.5 whose z is the first component
    comps[:, 0] = (pm - pm.mean()) / pm.std()
    classes, _ = confidence.classes(comps.sum(1)[:, None], CUTOFFS)
    labels = np.array(["LOW", "ELEVATED", "HIGH", "CRITICAL"])[classes[:, 0]]
    return pm, comps, labels


def test_no_error_and_no_jitter_keeps_every_class():
    pm, comps, labels = _scores()
    out = confidence.simulate_retention(
        pm,
        comps,
        labels,
        np.zeros((len(pm), len(pm))),
        CUTOFFS,
        n_draws=20,
        jitter=0.0,
        rng=np.random.default_rng(0),
    )
    assert (out["class_retention"] == 1).all()
    assert (out["retention_weights"] == 1).all()
    np.testing.assert_array_equal(out["p_critical"] == 1, labels == "CRITICAL")


def test_huge_error_breaks_retention():
    pm, comps, labels = _scores()
    out = confidence.simulate_retention(
        pm,
        comps,
        labels,
        50 * np.eye(len(pm)),
        CUTOFFS,
        n_draws=200,
        jitter=0.0,
        rng=np.random.default_rng(0),
    )
    assert out["retention_pm25"].mean() < 0.6


def test_label_thresholds():
    prod = {"confidence_high": 0.9, "confidence_low": 0.5}
    got = confidence.label(pd.Series([0.95, 0.9, 0.7, 0.5, 0.49]), prod)
    assert list(got) == ["HIGH", "HIGH", "MEDIUM", "MEDIUM", "LOW"]


def test_pm25_check_rejects_a_mismatch():
    pm = np.array([8.0, 9.0, 10.0])
    z = (pm - pm.mean()) / pm.std()
    confidence._check_pm25_matches(pm, z, "w")
    with pytest.raises(ValueError, match="differs"):
        confidence._check_pm25_matches(pm, z + 0.01, "w")


def test_chronic_flag_needs_a_full_window():
    df = pd.DataFrame(
        {
            "geoid": ["a", "b", "c"],
            "weeks_critical": [30, 10, 30],
            "weeks_observed": [52, 52, 40],
        }
    )
    out = chronic.add_flags(df, window=52, min_weeks=26)
    assert out["is_chronic"].tolist()[:2] == [True, False]
    assert pd.isna(out["is_chronic"].iloc[2])


def test_hex_takes_the_tract_with_most_people():
    # three blocks at one point (one res-9 cell), two tracts
    blocks = pd.DataFrame(
        {
            "geoid": ["06037000100", "06037000100", "06037000200"],
            "population": [30, 20, 40],
            "lat": [34.05] * 3,
            "lon": [-118.25] * 3,
        }
    )
    out = hex_grid.hexes_from_blocks(blocks, 9)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["geoid"] == "06037000100" and row["population"] == 90
    assert row["n_tracts"] == 2 and row["main_share"] == pytest.approx(0.556, abs=1e-3)


def test_scoring_refuses_a_window_with_late_kriging(monkeypatch):
    from src.spatial import compound_score as cs

    monkeypatch.setattr(
        cs, "read_query", lambda q, c: pd.DataFrame({"week": ["2024-07-15"]})
    )
    with pytest.raises(ValueError, match="2024-07-15"):
        cs._check_window_complete({"full_fips": "06037"}, "2024-07-01", "2024-07-22")
    monkeypatch.setattr(cs, "read_query", lambda q, c: pd.DataFrame({"week": []}))
    cs._check_window_complete({"full_fips": "06037"}, "2024-06-03", "2024-06-24")
