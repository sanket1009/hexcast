"""Tests for the B1 index validation checks (src/evals/index_validation.py)."""

import numpy as np
import pandas as pd
import pytest

from src.evals import index_validation as iv
from src.spatial import compound_score

PERCENTILES = {"burden_percentiles": {"elevated": 50, "high_risk": 75, "critical": 90}}


def _signals(n=200, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "geoid": [f"t{i:03d}" for i in range(n)],
            "pm25_kriged": rng.normal(9, 1, n),
            "lst_zscore": rng.normal(0, 1, n),
            "access_score_er": rng.uniform(0.5, 3.0, n),
        }
    )


def test_jaccard():
    assert iv.jaccard({1, 2}, {1, 2}) == 1.0
    assert iv.jaccard({1, 2}, {3}) == 0.0
    assert iv.jaccard({1, 2, 3}, {2, 3, 4}) == 0.5


def test_unit_weights_match_production():
    s = _signals()
    base = compound_score._build_composite(s, PERCENTILES)
    same = compound_score._build_composite(
        s, PERCENTILES, weights={c: 1.0 for c in compound_score.SIGNED_SIGNALS}
    )
    pd.testing.assert_frame_equal(base, same)


def test_sensitivity_is_identity_at_weight_one():
    out = iv.weight_sensitivity(_signals(), PERCENTILES, factors=(1.0,))
    assert (out["spearman"] > 0.9999).all()
    assert (out["jaccard"] == 1.0).all()
    assert (out["left_critical"] == 0).all()


def test_dropping_a_component_changes_more_than_a_small_change():
    out = iv.weight_sensitivity(_signals(), PERCENTILES).set_index(["component", "weight"])
    for col in compound_score.SIGNED_SIGNALS:
        assert out.loc[(col, 0.0), "jaccard"] < out.loc[(col, 1.25), "jaccard"]


def _weekly(classes_by_week):
    rows = []
    for i, (week, critical) in enumerate(classes_by_week.items()):
        for g in ["a", "b", "c", "d"]:
            rows.append(
                {
                    "week_start_date": week,
                    "geoid": g,
                    "raw_burden": float(ord(g)) + i,
                    "burden_class": "CRITICAL" if g in critical else "LOW",
                }
            )
    return pd.DataFrame(rows)


def test_temporal_stability_counts_retained_and_persistent():
    scores = _weekly(
        {"2026-01-05": {"a", "b"}, "2026-01-12": {"a", "c"}, "2026-01-19": {"a", "c"}}
    )
    per_week, summary = iv.temporal_stability(scores)
    assert per_week["retained"].tolist() == [0.5, 1.0]
    assert summary["ever_critical"] == 3       # a, b, c
    assert summary["persistent_critical"] == 2  # a (3 of 3), c (2 of 3)
    assert summary["n_weeks"] == 3


def test_temporal_stability_needs_two_weeks():
    with pytest.raises(ValueError):
        iv.temporal_stability(_weekly({"2026-01-05": {"a"}}))
