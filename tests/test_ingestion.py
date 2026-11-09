"""Tests for assigning tracts to reporting areas. Synthetic squares, no network."""

import geopandas as gpd
from shapely.geometry import box

from src.ingestion.reporting_areas import assign_largest_overlap

CRS = "EPSG:26911"


def test_tract_goes_to_the_area_it_overlaps_most():
    tracts = gpd.GeoDataFrame(
        {"geoid": ["A", "B"]}, geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10)], crs=CRS
    )
    areas = gpd.GeoDataFrame(
        {"name": ["West", "East"], "code": ["1", "2"]},
        geometry=[
            box(-5, -5, 7, 15),
            box(7, -5, 40, 15),
        ],  # A is 70% West, 30% East; B is all East
        crs=CRS,
    )
    out = assign_largest_overlap(tracts, areas, "name", "code").set_index("geoid")

    assert out.loc["A", "area_name"] == "West"
    assert out.loc["A", "share"] == 0.7
    assert out.loc["A", "area_id"] == "1"
    assert out.loc["B", "area_name"] == "East"
    assert out.loc["B", "share"] == 1.0


def test_tract_outside_every_area_is_left_out():
    tracts = gpd.GeoDataFrame(
        {"geoid": ["A", "Z"]},
        geometry=[box(0, 0, 1, 1), box(100, 100, 101, 101)],
        crs=CRS,
    )
    areas = gpd.GeoDataFrame({"name": ["Only"]}, geometry=[box(0, 0, 5, 5)], crs=CRS)
    out = assign_largest_overlap(tracts, areas, "name")
    assert list(out["geoid"]) == ["A"]
    assert "area_id" not in out.columns


def test_share_ignores_parts_no_area_covers():
    # Half the tract is ocean that no area covers; the covered half is all in one area
    tracts = gpd.GeoDataFrame({"geoid": ["C"]}, geometry=[box(0, 0, 10, 10)], crs=CRS)
    areas = gpd.GeoDataFrame({"name": ["Coast"]}, geometry=[box(0, 0, 5, 10)], crs=CRS)
    assert assign_largest_overlap(tracts, areas, "name")["share"].iloc[0] == 1.0


# ---------------------------------------------------------------------------
# EPA AQS: yearly requests for backfills
# ---------------------------------------------------------------------------

from datetime import datetime, timedelta  # noqa: E402

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.ingestion import epa_aqs  # noqa: E402

AQS_CONFIG = {
    "city_name": "Los Angeles",
    "full_fips": "06037",
    "epa_aqs": {"state_code": "06", "county_code": "037", "monitor_count": 2},
    "neighbor_counties": ["06059", "06111"],
}


def _fake_daily(state_code, county_code, start_date, end_date, config):
    """Two monitors per county, one reading per day; county 06111 has nothing in 2026."""
    if county_code == "111" and start_date.year == 2026:
        return pd.DataFrame()
    days = pd.date_range(start_date, end_date, freq="D")
    rows = [
        {
            "monitor_id": f"{state_code}-{county_code}-{n:04d}",
            "date": d,
            "pm25_value": 8.0 + n,
            "latitude": 34.0,
            "longitude": -118.0,
            "monitor_name": f"m{n}",
            "county_fips": state_code + county_code,
        }
        for d in days
        for n in (1, 2)
    ]
    return pd.DataFrame(rows)


def test_year_ranges_split_at_new_year():
    ranges = epa_aqs._year_ranges(datetime(2025, 12, 29), datetime(2026, 1, 4))
    assert ranges == [
        (datetime(2025, 12, 29), datetime(2025, 12, 31)),
        (datetime(2026, 1, 1), datetime(2026, 1, 4)),
    ]
    assert len(epa_aqs._year_ranges(datetime(2026, 1, 5), datetime(2026, 12, 27))) == 1


def test_backfill_makes_one_request_per_county_per_year(monkeypatch):
    calls, written = [], []

    def fetch(**kw):
        calls.append((kw["county_code"], kw["start_date"].year))
        return _fake_daily(**kw)

    monkeypatch.setattr(epa_aqs, "_fetch_daily_readings", fetch)
    monkeypatch.setattr(epa_aqs, "_filter_to_buffer", lambda df, config: df)
    monkeypatch.setattr(
        epa_aqs, "write_dataframe", lambda df, table, config, **kw: written.append(df)
    )

    # 52 weeks from Jan 6, 2025 to Dec 29, 2025; the last week ends Jan 4, 2026
    first = datetime(2025, 1, 6)
    last = first + timedelta(weeks=51)
    result = epa_aqs.pull_pm25_range(AQS_CONFIG, first, last)

    assert len(calls) == 6  # 3 counties x 2 calendar years, not 3 x 52
    assert len(result["weeks_written"]) == 52
    assert result["weeks_missing"] == []
    assert len(written) == 1  # one write for all weeks
    df = written[0]
    assert df["week_start_date"].nunique() == 52
    assert (df.groupby("week_start_date")["reading_count"].max() == 7).all()


def test_backfill_reports_weeks_without_city_readings(monkeypatch):
    def fetch(**kw):
        if kw["county_code"] == "037" and kw["start_date"].month == 1:
            df = _fake_daily(**kw)
            return df[
                df["date"] >= datetime(2026, 1, 12)
            ]  # city has no data the first week
        return _fake_daily(**kw)

    monkeypatch.setattr(epa_aqs, "_fetch_daily_readings", fetch)
    monkeypatch.setattr(epa_aqs, "_filter_to_buffer", lambda df, config: df)
    monkeypatch.setattr(epa_aqs, "write_dataframe", lambda *a, **k: None)

    result = epa_aqs.pull_pm25_range(
        AQS_CONFIG, datetime(2026, 1, 5), datetime(2026, 1, 12)
    )
    assert result["weeks_missing"] == ["2026-01-05"]
    assert result["weeks_written"] == ["2026-01-12"]


def _weekly(means, in_city=None):
    n = len(means)
    return pd.DataFrame(
        {
            "monitor_id": [f"m{i}" for i in range(n)],
            "pm25_mean": means,
            "in_city": in_city if in_city is not None else [True] * n,
        }
    )


def test_one_faulty_monitor_is_dropped_and_the_week_kept():
    # Week of 2026-01-12: 2 of 21 monitors reported a negative weekly mean
    df = _weekly([9.0, -0.4, 11.2, 8.7, 10.1, 7.9, 12.3, 9.5])
    out = epa_aqs._drop_invalid_monitors(df, datetime(2026, 1, 12))
    assert len(out) == 7
    assert "m1" not in set(out["monitor_id"])
    assert (out["pm25_mean"] >= 0).all()


def test_too_many_invalid_monitors_still_rejects_the_week():
    df = _weekly([9.0, -0.4, -1.2, 640.0, 10.1, 7.9, 12.3, 9.5])  # 3 of 8
    with pytest.raises(ValueError, match="3 of 8 monitors"):
        epa_aqs._drop_invalid_monitors(df, datetime(2026, 1, 12))


def test_clean_week_is_unchanged():
    df = _weekly([9.0, 11.2, 8.7])
    out = epa_aqs._drop_invalid_monitors(df, datetime(2026, 1, 12))
    pd.testing.assert_frame_equal(out, df)
