"""Partition and cluster choices for the BigQuery copy of the warehouse."""

import importlib.util
from pathlib import Path

import duckdb
import pandas as pd

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "sync_warehouse_to_bigquery.py"
)
spec = importlib.util.spec_from_file_location("sync_bq", SCRIPT)
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)


def test_weekly_tables_are_partitioned_and_clustered():
    df = pd.DataFrame(
        {
            "geoid": ["a"],
            "city_fips": ["06037"],
            "week_start_date": [pd.Timestamp("2026-06-22")],
        }
    )
    assert sync.layout(df) == {
        "partition": "week_start_date",
        "cluster": ["city_fips", "geoid"],
    }


def test_static_tables_are_not_partitioned():
    assert sync.layout(pd.DataFrame({"h3_index": ["x"], "geoid": ["a"]})) == {
        "partition": None,
        "cluster": ["geoid"],
    }
    assert sync.layout(pd.DataFrame({"metric": ["m"]})) == {
        "partition": None,
        "cluster": None,
    }


def test_local_tables_lists_base_tables_smallest_first(tmp_path):
    con = duckdb.connect(str(tmp_path / "w.duckdb"))
    con.execute('CREATE SCHEMA "s"')
    con.execute('CREATE TABLE "s"."big" AS SELECT range AS x FROM range(5)')
    con.execute('CREATE TABLE "s"."small" AS SELECT 1 AS x')
    con.execute('CREATE VIEW "s"."v" AS SELECT * FROM "s"."big"')
    assert sync.local_tables(con, "s") == {"small": 1, "big": 5}
