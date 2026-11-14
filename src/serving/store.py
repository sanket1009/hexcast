"""
store.py
========
Read-only warehouse access for the API.

One method, query(sql, params), with two backends:
    LocalStore     DuckDB file, opened read_only for each query
    BigQueryStore  BigQuery with typed query parameters

SQL is written once for both: {dataset} for the dataset prefix and @name for
a bound value (BigQuery's syntax; LocalStore rewrites it to DuckDB's $name).
Keep SQL to what both dialects share (SELECT, JOIN, WHERE, GROUP BY, COUNT,
SUM, AVG, CASE) and do date arithmetic and casts in Python.

Why not src.utils.bigquery.read_query: it pastes values into the SQL text and
opens DuckDB read-write. The API takes values from outside, so they go in as
bound parameters, never as SQL text; and a read_only connection means no API
bug can change the warehouse.
"""

from __future__ import annotations

import re
import time
from datetime import date
from typing import Any, Protocol

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

_PARAM = re.compile(r"@(\w+)")


class Store(Protocol):
    backend: str

    def query(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Run a SELECT and return its rows."""


class LocalStore:
    """DuckDB file, one read_only connection per query."""

    backend = "local"

    def __init__(self, db_path: str, schema: str, retries: int = 3) -> None:
        self.db_path = db_path
        self.schema = schema
        self.retries = retries

    def query(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        import duckdb

        sql = _PARAM.sub(r"$\1", sql.replace("{dataset}", f'"{self.schema}"'))
        for attempt in range(1, self.retries + 1):
            try:
                con = duckdb.connect(self.db_path, read_only=True)
                try:
                    return con.execute(sql, params or {}).df()
                finally:
                    con.close()
            except duckdb.IOException:
                # An Airflow task holds the write lock; DuckDB allows one
                # writer process and no readers while it writes
                if attempt == self.retries:
                    raise
                time.sleep(0.5 * attempt)
        raise RuntimeError("unreachable")


class BigQueryStore:
    """BigQuery with query parameters; the client is created on first use."""

    backend = "gcp"

    def __init__(self, project: str, dataset: str) -> None:
        self.project = project
        self.dataset = dataset
        self._client = None

    def _bq_param(self, name: str, value: Any):
        from google.cloud import bigquery

        if isinstance(value, bool):
            kind = "BOOL"
        elif isinstance(value, int):
            kind = "INT64"
        elif isinstance(value, float):
            kind = "FLOAT64"
        elif isinstance(value, date):
            kind = "DATE"
        else:
            kind, value = "STRING", str(value)
        return bigquery.ScalarQueryParameter(name, kind, value)

    def query(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        from google.cloud import bigquery

        if self._client is None:
            self._client = bigquery.Client(project=self.project)
        # Backticks around the project only: hyphenated project IDs need them
        sql = sql.replace("{dataset}", f"`{self.project}`.{self.dataset}")
        job_config = bigquery.QueryJobConfig(
            query_parameters=[self._bq_param(k, v) for k, v in (params or {}).items()],
            use_query_cache=True,
        )
        return self._client.query(sql, job_config=job_config).to_dataframe()


def make_store(config: dict[str, Any]) -> Store:
    storage = config.get("storage", {})
    if storage.get("backend", "gcp") == "local":
        return LocalStore(storage["local_db_path"], config["gcp"]["bigquery_dataset"])
    return BigQueryStore(
        config["secrets"]["GCP_PROJECT_ID"], config["gcp"]["bigquery_dataset"]
    )
