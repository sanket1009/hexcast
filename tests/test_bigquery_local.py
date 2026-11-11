"""Local (DuckDB) write paths of src/utils/bigquery.py. No GCP."""

import datetime

import duckdb
import pandas as pd

from src.utils import bigquery as bq


def _config(tmp_path):
    return {
        "storage": {"backend": "local", "local_db_path": str(tmp_path / "w.duckdb")},
        "gcp": {"bigquery_dataset": "chbindex_test"},
        "bigquery": {"write_disposition": "WRITE_APPEND"},
    }


def _week(day, **cols):
    return pd.DataFrame(
        {
            "geoid": ["a", "b"],
            "week_start_date": [datetime.date(2026, 6, day)] * 2,
            "city_fips": ["06037"] * 2,
            **cols,
        }
    )


def test_replace_partition_adds_a_new_column(tmp_path):
    # compound_scores gained vulnerability_z in Phase 3: older weeks have no
    # value for it, and the write must not fail on the missing column
    config = _config(tmp_path)
    bq.write_dataframe(
        _week(15, pm25_z=[1.0, 2.0]), "scores", config, bq.REPLACE_PARTITION
    )
    bq.write_dataframe(
        _week(22, pm25_z=[3.0, 4.0], vulnerability_z=[0.5, -0.5]),
        "scores",
        config,
        bq.REPLACE_PARTITION,
    )
    con = duckdb.connect(config["storage"]["local_db_path"], read_only=True)
    df = con.execute(
        'SELECT * FROM "chbindex_test".scores ORDER BY week_start_date, geoid'
    ).df()
    con.close()
    assert len(df) == 4
    assert df["vulnerability_z"].isna().tolist() == [True, True, False, False]
    assert df["vulnerability_z"].dropna().tolist() == [0.5, -0.5]


def test_replace_partition_still_replaces_a_week(tmp_path):
    config = _config(tmp_path)
    bq.write_dataframe(_week(22, x=[1.0, 2.0]), "t", config, bq.REPLACE_PARTITION)
    bq.write_dataframe(_week(22, x=[5.0, 6.0]), "t", config, bq.REPLACE_PARTITION)
    con = duckdb.connect(config["storage"]["local_db_path"], read_only=True)
    assert con.execute('SELECT SUM(x) FROM "chbindex_test".t').fetchone()[0] == 11.0
    con.close()


def test_bigquery_types_for_new_columns():
    assert bq._bq_type(pd.Series([1.5])) == "FLOAT"
    assert bq._bq_type(pd.Series([1])) == "INTEGER"
    assert bq._bq_type(pd.Series([True])) == "BOOLEAN"
    assert bq._bq_type(pd.Series(["x"])) == "STRING"
