"""Tests for the Census 2020 block pull. Fake TIGERweb pages, no network."""

import pandas as pd
import pytest

from src.ingestion import census_blocks as cb


def _feature(geoid, pop, lat="+34.0500000", lon="-118.2500000"):
    return {
        "attributes": {
            "GEOID": geoid,
            "POP100": pop,
            "HU100": pop // 3,
            "INTPTLAT": lat,
            "INTPTLON": lon,
            "AREALAND": 10_000,
        }
    }


def test_parse_signed_strings_and_tract_prefix():
    df = cb.parse_features(
        [_feature("060371011101000", 43, "+34.4590326", "-118.4106242")]
    )
    row = df.iloc[0]
    assert row["geoid"] == "06037101110"
    assert row["lat"] == pytest.approx(34.4590326)
    assert row["lon"] == pytest.approx(-118.4106242)
    assert row["population"] == 43


def test_fetch_all_pages_until_transfer_limit_clears(monkeypatch):
    pages = [
        {
            "features": [_feature("060371011101000", 1)] * 2,
            "exceededTransferLimit": True,
        },
        {"features": [_feature("060371011101001", 1)], "exceededTransferLimit": False},
    ]
    offsets = []

    def fake_get(params):
        offsets.append(params["resultOffset"])
        return pages[len(offsets) - 1]

    monkeypatch.setattr(cb, "_get", fake_get)
    assert len(cb.fetch_all("x")) == 3
    assert offsets == [0, 2]


def _config():
    return {"full_fips": "06037"}


def _patch_tracts(monkeypatch, geoids):
    monkeypatch.setattr(cb, "read_query", lambda q, c: pd.DataFrame({"geoid": geoids}))


def test_validate_rejects_population_mismatch(monkeypatch):
    _patch_tracts(monkeypatch, ["06037101110"])
    df = cb.parse_features([_feature("060371011101000", 10)])
    with pytest.raises(ValueError, match="population"):
        cb._validate(df, {"n": 1, "pop": 11}, _config())


def test_validate_rejects_populated_block_in_unknown_tract(monkeypatch):
    _patch_tracts(monkeypatch, ["06037999999"])
    df = cb.parse_features([_feature("060371011101000", 10)])
    with pytest.raises(ValueError, match="missing from tract_geometries"):
        cb._validate(df, {"n": 1, "pop": 10}, _config())


def test_validate_allows_empty_block_in_unknown_tract(monkeypatch):
    _patch_tracts(monkeypatch, ["06037101110"])
    df = cb.parse_features(
        [_feature("060371011101000", 10), _feature("060379902001000", 0)]
    )
    cb._validate(df, {"n": 2, "pop": 10}, _config())
