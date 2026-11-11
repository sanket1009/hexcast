"""Tests for the CalEnviroScreen pull. Fake ArcGIS pages, no network."""

import pandas as pd
import pytest

from src.ingestion import calenviroscreen as ces


def _row(tract, score=40.0):
    return {
        "tract": tract,
        "Population": 3000,
        "CIscore": score,
        "CIscoreP": 70.0,
        "PollutionScore": 6.0,
        "Pollution_Pctl": 80.0,
        "PopCharScore": 5.0,
        "PM2_5": 11.2,
        "asthma": 55.0,
        "Cardiovascular": -999.0,
    }


def test_geoid_gets_its_leading_zero_back():
    df = ces._process([_row(6037101110)], "06037")
    assert df.loc[0, "geoid"] == "06037101110"
    assert df.loc[0, "pm25_ces"] == 11.2
    assert df.loc[0, "asthma_ed_rate"] == 55.0
    assert pd.isna(df.loc[0, "cardio_ed_rate"])  # negative = missing


def test_negative_scores_are_missing():
    df = ces._process([_row(6037101110, score=-999)], "06037")
    assert df["ces_score"].isna().all()
    assert df.loc[0, "population"] == 3000


def test_fetch_follows_pages_and_filters_the_county(monkeypatch):
    calls = []
    pages = [
        {
            "features": [{"attributes": _row(6037000100)}] * 2,
            "exceededTransferLimit": True,
        },
        {"features": [{"attributes": _row(6037000200)}]},
    ]

    def fake(params):
        calls.append(params)
        return pages[len(calls) - 1]

    monkeypatch.setattr(ces, "_request_with_retry", fake)
    rows = ces._fetch("06037")
    assert len(rows) == 3
    assert [c["resultOffset"] for c in calls] == [0, 2]
    assert calls[0]["where"] == "tract >= 6037000000 AND tract < 6038000000"


def test_service_error_is_raised(monkeypatch):
    monkeypatch.setattr(
        ces, "_request_with_retry", lambda p: {"error": {"code": 400, "message": "bad"}}
    )
    with pytest.raises(RuntimeError):
        ces._fetch("06037")


def test_only_california():
    with pytest.raises(ValueError):
        ces.pull_calenviroscreen({"state_fips": "17", "full_fips": "17031"})
