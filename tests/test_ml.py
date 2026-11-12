"""Phase 6 ML panel and CV tests: target, look-ahead, grid, splits."""

import numpy as np
import pandas as pd
import pytest

from src.ml.features import (
    SPATIAL_LAG_FEATURES,
    TARGET,
    TEMPORAL_FEATURES,
    build_panel,
    check_no_leakage,
    complete_grid,
)
from src.ml.spatial_cv import SpaceTimeBlockCV, assign_blocks_to_folds

N_TRACTS, N_WEEKS = 12, 14
WEEK0 = pd.Timestamp("2025-06-23")
PANEL_FEATURES = TEMPORAL_FEATURES + SPATIAL_LAG_FEATURES


def toy_inputs(seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N_TRACTS):
        for w in range(N_WEEKS):
            pct = float(rng.uniform(0, 100))
            rows.append(
                {
                    "geoid": f"06037{i:06d}",
                    "week_start_date": WEEK0 + pd.Timedelta(weeks=w),
                    "burden_percentile": pct,
                    "burden_class": "CRITICAL"
                    if pct >= 90
                    else ("HIGH" if pct >= 75 else "LOW"),
                    "raw_burden": pct / 20 - 2.5,
                    "pm25_z": rng.normal(),
                    "lst_z": rng.normal(),
                    "access_z": 0.1 * i,
                    "vulnerability_z": -0.1 * i,
                    "pm25_1w": rng.uniform(5, 15),
                    "lst_mean_c": rng.uniform(30, 50),
                }
            )
    static = pd.DataFrame(
        {
            "geoid": [f"06037{i:06d}" for i in range(N_TRACTS)],
            "centroid_lat": 34.0 + 0.01 * np.arange(N_TRACTS),
            "centroid_lon": -118.2 + 0.005 * (np.arange(N_TRACTS) % 4),
        }
    )
    return pd.DataFrame(rows), static


def set_class(weekly: pd.DataFrame, geoid: str, week: int, critical: bool) -> None:
    m = (weekly["geoid"] == geoid) & (
        weekly["week_start_date"] == WEEK0 + pd.Timedelta(weeks=week)
    )
    weekly.loc[m, "burden_percentile"] = 95.0 if critical else 60.0
    weekly.loc[m, "burden_class"] = "CRITICAL" if critical else "ELEVATED"


def test_target_and_eligibility():
    weekly, static = toy_inputs()
    g = "06037000000"
    for w in range(N_WEEKS):
        set_class(weekly, g, w, critical=False)
    set_class(weekly, g, 9, critical=True)
    p = build_panel(weekly, static).set_index(["geoid", "week_idx"])
    # Week 5's window is 6..9 and includes week 9; week 4's window 5..8 does not.
    assert p.loc[(g, 5), TARGET] == 1
    assert p.loc[(g, 4), TARGET] == 0
    assert p.loc[(g, 4), "eligible"]
    assert not p.loc[(g, 3), "eligible"]  # no 4-week lag yet
    assert not p.loc[(g, 9), "eligible"]  # CRITICAL now
    assert not p.loc[(g, N_WEEKS - 4), "eligible"]  # only 3 future weeks
    assert p.loc[(g, N_WEEKS - 6), "eligible"]  # week 8: future 9..12 observed


def test_features_do_not_look_ahead():
    weekly, static = toy_inputs()
    base = build_panel(weekly, static)
    t = 6
    changed = weekly.copy()
    future = changed["week_start_date"] > WEEK0 + pd.Timedelta(weeks=t)
    rng = np.random.default_rng(99)
    for col in [
        "burden_percentile",
        "raw_burden",
        "pm25_z",
        "lst_z",
        "pm25_1w",
        "lst_mean_c",
    ]:
        changed.loc[future, col] = rng.uniform(-50, 150, future.sum())
    changed.loc[future, "burden_class"] = "CRITICAL"
    after = build_panel(changed, static)
    cols = ["geoid", "week_idx", *PANEL_FEATURES]
    a = base.loc[base["week_idx"] <= t, cols].reset_index(drop=True)
    b = after.loc[after["week_idx"] <= t, cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    # ...while the label for week t does see the future.
    assert (after.loc[after["week_idx"] == t, TARGET] == 1).all()


def test_complete_grid_missing_week_is_nan_not_shifted():
    weekly, _ = toy_inputs()
    g = "06037000001"
    drop = (weekly["geoid"] == g) & (
        weekly["week_start_date"] == WEEK0 + pd.Timedelta(weeks=3)
    )
    grid = complete_grid(weekly[~drop])
    assert len(grid) == N_TRACTS * N_WEEKS
    row = grid[(grid["geoid"] == g) & (grid["week_idx"] == 3)].iloc[0]
    assert np.isnan(row["burden_percentile"]) and np.isnan(row["is_critical"])


def test_skipped_week_becomes_nan_and_blocks_its_label_windows():
    weekly, static = toy_inputs()
    skipped = WEEK0 + pd.Timedelta(weeks=7)
    p = build_panel(weekly[weekly["week_start_date"] != skipped], static)
    assert p["week_idx"].nunique() == N_WEEKS
    wk = p[p["week_idx"] == 7]
    assert wk["burden_percentile"].isna().all() and not wk["eligible"].any()
    # Rows whose window t+1..t+4 includes week 7 (t = 3..6) get no label.
    assert p.loc[p["week_idx"].between(3, 6), TARGET].isna().all()
    assert not p.loc[p["week_idx"].between(3, 6), "eligible"].any()
    # Week 8's lag-1 is the skipped week: NaN, not week 6's value.
    assert p.loc[p["week_idx"] == 8, "pct_lag_1w"].isna().all()


def test_complete_grid_rejects_off_cycle_dates():
    weekly, _ = toy_inputs()
    m = weekly["week_start_date"] == WEEK0 + pd.Timedelta(weeks=5)
    weekly.loc[m, "week_start_date"] += pd.Timedelta(days=2)
    with pytest.raises(ValueError, match="not 7 days apart"):
        complete_grid(weekly)


def test_check_no_leakage():
    check_no_leakage(PANEL_FEATURES)
    with pytest.raises(ValueError):
        check_no_leakage(["burden_percentile", "next_week_score"])


def test_space_time_cv_embargo_and_coverage():
    rng = np.random.default_rng(1)
    n_tracts, weeks = 60, np.arange(4, 49)
    groups_t = rng.integers(0, 20, n_tracts)
    week_idx = np.tile(weeks, n_tracts)
    groups = np.repeat(groups_t, len(weeks))
    cv = SpaceTimeBlockCV(
        n_spatial_splits=5, n_time_windows=3, test_window_weeks=8, embargo_weeks=4
    )
    assert cv.time_windows(week_idx) == [(25, 32), (33, 40), (41, 48)]
    seen = np.zeros(len(week_idx), dtype=int)
    for info, tr, te in cv.split(week_idx, groups):
        start, end = info["test_weeks"]
        assert (
            week_idx[tr].max() <= start - 5
        )  # last train label window ends at start-1
        assert set(groups[tr]).isdisjoint(groups[te])
        assert week_idx[te].min() >= start and week_idx[te].max() <= end
        seen[te] += 1
    assert (seen[week_idx >= 25] == 1).all() and (seen[week_idx < 25] == 0).all()


def test_assign_blocks_to_folds_keeps_blocks_together():
    groups = np.array(["a", "b", "a", "c", "d", "e", "b"])
    fold = assign_blocks_to_folds(groups, n_splits=3, random_seed=0)
    assert fold[0] == fold[2] and fold[1] == fold[6]
    with pytest.raises(ValueError):
        assign_blocks_to_folds(np.array(["a", "b"]), n_splits=3, random_seed=0)


def test_inner_split_stays_inside_outer_train_with_gaps():
    from src.ml.tuning import inner_split

    rng = np.random.default_rng(2)
    n_tracts, weeks = 80, np.arange(4, 101)
    week_idx = np.tile(weeks, n_tracts)
    groups = np.repeat(rng.integers(0, 30, n_tracts), len(weeks))
    fold = assign_blocks_to_folds(groups, n_splits=5, random_seed=42)
    cv = SpaceTimeBlockCV(n_spatial_splits=5, random_seed=42)
    for info, tr, _ in cv.split(week_idx, groups):
        k, (start, _) = info["spatial_fold"], info["test_weeks"]
        in_tr, in_val, inner = inner_split(week_idx, fold, k, start, n_splits=5)
        assert set(in_tr) <= set(tr) and set(in_val) <= set(tr)
        assert set(groups[in_tr]).isdisjoint(groups[in_val])
        val_start, val_end = inner["val_weeks"]
        assert val_end == start - 5 and val_end - val_start == 7
        assert (
            week_idx[in_tr].max() <= val_start - 5
        )  # inner labels end before validation
        assert (fold[in_val] == (k + 1) % 5).all()


def test_weighted_ap_matches_sklearn_with_ties():
    from sklearn.metrics import average_precision_score

    from src.ml.compare_stats import _presort, _weighted_ap_presorted

    rng = np.random.default_rng(3)
    y = (rng.random(500) < 0.1).astype(float)
    score = np.round(rng.random(500) + y * 0.3, 1)  # many ties
    order, groups = _presort(score)
    ours = _weighted_ap_presorted(y[order], np.ones(500), groups)
    assert ours == pytest.approx(average_precision_score(y, score))


def test_block_bootstrap_and_delong_detect_a_clear_difference():
    from src.ml.compare_stats import block_bootstrap_diff, delong_paired

    rng = np.random.default_rng(4)
    n = 4000
    y = (rng.random(n) < 0.1).astype(int)
    good = y + rng.normal(0, 0.5, n)
    noise = rng.normal(0, 1, n)
    blocks = rng.integers(0, 60, n)
    boot = block_bootstrap_diff(y, good, noise, blocks, n_boot=300)
    assert boot["ci_low"] > 0 and boot["share_le_0"] == 0
    same = block_bootstrap_diff(y, good, good, blocks, n_boot=50)
    assert same["diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    dl = delong_paired(y, good, noise)
    assert dl["diff"] > 0.3 and dl["p_value"] < 1e-6


def _toy_oof(n: int = 3000, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = rng.uniform(0, 0.4, n)
    return pd.DataFrame(
        {
            "geoid": [f"g{i % 300:03d}" for i in range(n)],
            "week_idx": np.repeat(np.arange(n // 100), 100),
            "time_window": np.arange(n) * 3 // n,
            TARGET: (rng.random(n) < p).astype(int),
            "score": p,
        }
    )


def test_calibration_report_on_calibrated_scores():
    from src.ml.report import calibration_report

    rep = calibration_report(_toy_oof(20000), "score", TARGET)
    assert rep["ece"] < 0.02
    assert rep["brier"] < rep["brier_base_rate_only"]
    assert set(rep["per_window"]) == {0, 1, 2}


def test_equity_audit_terciles_and_flag():
    from src.ml.report import equity_audit

    oof = _toy_oof()
    tracts = pd.DataFrame(
        {
            "geoid": [f"g{i:03d}" for i in range(300)],
            "median_income": np.arange(300) * 1000.0,
        }
    )
    eq = equity_audit(oof, tracts, "score", TARGET, k=10, attributes=["median_income"])
    a = eq["median_income"]
    assert list(a["terciles"]) == ["low", "mid", "high"]
    assert sum(r["tracts"] for r in a["terciles"].values()) == 300
    assert a["lowest_recall_tercile"] in {"low", "mid", "high"}
    assert isinstance(a["flag"], bool)
