"""Tests for the ACS B03002 pull. Fake Census responses, no network."""

import numpy as np
import pandas as pd
import pytest

from src.ingestion import acs_race_ethnicity as race
from src.ingestion.acs_vulnerability import CONTROLLED_MOE


def _raw(**counts):
    """One tract; rows default to 0 with MOE 0; total controlled."""
    cols = {"geoid": ["06037101110"]}
    for row in race.variables():
        cols[f"{row}E"] = [0.0]
        cols[f"{row}M"] = [0.0]
    for row, (e, m) in counts.items():
        cols[f"{row}E"], cols[f"{row}M"] = [e], [m]
    df = pd.DataFrame(cols)
    df[f"{race.TOTAL}M"] = CONTROLLED_MOE
    return df


def test_variables_are_total_plus_eight_groups():
    assert len(race.variables()) == 9
    assert race.TOTAL in race.variables()


def test_other_combines_four_rows_with_root_sum_square_moe():
    raw = _raw(
        B03002_001=(1000.0, 0),
        B03002_005=(10.0, 3.0),
        B03002_007=(10.0, 4.0),
        B03002_008=(0.0, 12.0),
        B03002_009=(30.0, 0.0),
        B03002_012=(950.0, 20.0),
    )
    out = race.compute_groups(raw).iloc[0]
    assert out["other"] == 50
    assert out["other_moe"] == pytest.approx(13.0)  # sqrt(9 + 16 + 144)
    assert out["other_pct"] == pytest.approx(5.0)
    assert out["latino_pct"] == pytest.approx(95.0)
    # controlled total: share MOE = count MOE / total
    assert out["latino_pct_moe"] == pytest.approx(2.0)


def test_zero_population_tract_has_no_shares():
    out = race.compute_groups(_raw(B03002_001=(0.0, 0))).iloc[0]
    assert np.isnan(out["white_pct"])


def test_validate_catches_groups_not_adding_up():
    df = race.compute_groups(_raw(B03002_001=(100.0, 0), B03002_003=(60.0, 5.0)))
    df["city_fips"] = "06037"
    with pytest.raises(ValueError, match="add up"):
        race._validate(df, {})
