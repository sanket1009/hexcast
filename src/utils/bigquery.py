"""
bigquery.py
===========
Shared BigQuery read/write utilities for the CHBIndex pipeline.

Every module that needs to interact with BigQuery imports from here.
This prevents duplicated boilerplate across ingestion, spatial,
ML, and RAG modules.

Usage:
    from src.utils.bigquery import write_dataframe, read_query, table_exists

    # Write a DataFrame to BigQuery
    write_dataframe(df, "epa_pm25_raw", config)

    # Read a SQL query result into a DataFrame
    df = read_query("SELECT * FROM epa_pm25_raw WHERE week = '2026-08-18'", config)

    # Check if a table exists before writing
    if table_exists("epa_pm25_raw", config):
        write_dataframe(df, "epa_pm25_raw", config)

Storage backends:
    config["storage"]["backend"] chooses where tables live.
      "gcp"   -> BigQuery (production)
      "local" -> one DuckDB file on disk (development, no cloud account)
    Every public function checks this first, so pipeline modules never
    need to know which backend is active. In local mode, each city's
    BigQuery dataset name (e.g. chbindex_06037) becomes a DuckDB schema.
"""

from pathlib import Path
from typing import Any

import pandas as pd
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from google.oauth2 import service_account

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Local backend (DuckDB)
# ---------------------------------------------------------------------------


def _is_local(config: dict[str, Any]) -> bool:
    """True when config selects the local DuckDB backend instead of BigQuery."""
    return config.get("storage", {}).get("backend", "gcp") == "local"


def _local_connect(config: dict[str, Any]):
    """Open the DuckDB file, creating its folder on first use."""
    import duckdb  # imported here so GCP-only installs don't need it

    db_path = Path(config["storage"]["local_db_path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def _local_schema(config: dict[str, Any]) -> str:
    """The city's BigQuery dataset name doubles as its DuckDB schema."""
    return config["gcp"]["bigquery_dataset"]


def _local_table_exists(con, schema: str, table_name: str) -> bool:
    return (
        con.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_schema = ? AND table_name = ?",
            [schema, table_name],
        ).fetchone()[0]
        > 0
    )


def _local_write(
    df: pd.DataFrame,
    table_name: str,
    config: dict[str, Any],
    write_disposition: str,
) -> None:
    """Write a DataFrame to DuckDB with the same dispositions as BigQuery."""
    schema = _local_schema(config)
    full_table = f'"{schema}"."{table_name}"'

    con = _local_connect(config)
    try:
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        con.register("incoming_df", df)
        exists = _local_table_exists(con, schema, table_name)

        if write_disposition == "WRITE_EMPTY" and exists:
            if con.execute(f"SELECT COUNT(*) FROM {full_table}").fetchone()[0] > 0:
                raise ValueError(f"WRITE_EMPTY: {full_table} already has data")

        if write_disposition == "WRITE_TRUNCATE" or not exists:
            con.execute(
                f"CREATE OR REPLACE TABLE {full_table} AS SELECT * FROM incoming_df"
            )
        else:
            # WRITE_APPEND: BY NAME matches columns by name, not position,
            # like a BigQuery load job does
            con.execute(f"INSERT INTO {full_table} BY NAME SELECT * FROM incoming_df")
    finally:
        con.close()

    logger.info(f"Successfully wrote {len(df):,} rows to local {full_table}")


def _local_query(query: str, config: dict[str, Any]) -> pd.DataFrame:
    """Run a query against DuckDB, resolving the {dataset} placeholder."""
    schema = _local_schema(config)
    query = query.format(dataset=f'"{schema}"', project="local").replace("`", "")

    logger.info("Executing local DuckDB query")
    logger.debug(f"Query: {query[:200]}...")

    con = _local_connect(config)
    try:
        df = con.execute(query).df()
    finally:
        con.close()

    logger.info(f"Query returned {len(df):,} rows")
    return df


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def get_client(config: dict[str, Any]) -> bigquery.Client:
    """
    Create and return an authenticated BigQuery client.

    Authenticates using the GCP service account JSON key file
    whose path is stored in config["secrets"]["GCP_SERVICE_ACCOUNT_PATH"].

    Parameters
    ----------
    config : dict
        Merged pipeline config from config_loader.load_config().
        Must contain config["secrets"]["GCP_PROJECT_ID"] and
        config["secrets"]["GCP_SERVICE_ACCOUNT_PATH"].

    Returns
    -------
    bigquery.Client
        Authenticated BigQuery client ready to use.

    Raises
    ------
    FileNotFoundError
        If the service account key file does not exist.
    google.auth.exceptions.DefaultCredentialsError
        If the service account key is invalid.

    Examples
    --------
    >>> client = get_client(config)
    >>> client.project
    'your-gcp-project-id'
    """
    project_id = config["secrets"]["GCP_PROJECT_ID"]
    sa_path = config["secrets"]["GCP_SERVICE_ACCOUNT_PATH"]

    logger.debug(f"Creating BigQuery client for project {project_id}")

    # If service account path is provided and exists, use it
    # Otherwise fall back to Application Default Credentials (ADC)
    # ADC works when running on GCP (Cloud Run, GCE) automatically
    if sa_path and sa_path != "/tmp/fake-key.json":
        credentials = service_account.Credentials.from_service_account_file(
            sa_path,
            scopes=["https://www.googleapis.com/auth/bigquery"],
        )
        client = bigquery.Client(
            project=project_id,
            credentials=credentials,
        )
    else:
        # Fall back to Application Default Credentials
        # Used in CI/CD and when running on GCP infrastructure
        client = bigquery.Client(project=project_id)

    logger.debug("BigQuery client created successfully")
    return client


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Partition overwrite (REPLACE_PARTITION)
# ---------------------------------------------------------------------------
# Re-running a week must replace that week's rows, not append to them.
# REPLACE_PARTITION deletes rows for each (week_start_date, city_fips) pair
# in the incoming DataFrame, then inserts the new rows, in one transaction.
# Other weeks and cities are left alone.
#
# Additive schema changes: a column in the DataFrame that the table lacks is
# added first (NULL for existing rows), e.g. vulnerability_z in
# compound_scores (Phase 3). Columns are never dropped or retyped.

REPLACE_PARTITION = "REPLACE_PARTITION"
PARTITION_KEYS = ["week_start_date", "city_fips"]


def _partition_values(df: pd.DataFrame) -> list[tuple]:
    """Distinct (week_start_date, city_fips) pairs in df: the partitions to overwrite."""
    missing = [k for k in PARTITION_KEYS if k not in df.columns]
    if missing:
        raise ValueError(f"REPLACE_PARTITION needs columns {missing} in the DataFrame")
    pairs = df[PARTITION_KEYS].drop_duplicates()
    return [tuple(row) for row in pairs.itertuples(index=False, name=None)]


def _local_replace_partition(
    df: pd.DataFrame, table_name: str, config: dict[str, Any]
) -> None:
    """DuckDB: delete the incoming partitions, insert the new rows, one transaction."""
    schema = _local_schema(config)
    full_table = f'"{schema}"."{table_name}"'
    keys = _partition_values(df)

    con = _local_connect(config)
    try:
        con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        con.register("incoming_df", df)

        if not _local_table_exists(con, schema, table_name):
            con.execute(f"CREATE TABLE {full_table} AS SELECT * FROM incoming_df")
        else:
            _local_add_new_columns(con, full_table)
            con.execute("BEGIN TRANSACTION")
            try:
                for week, city in keys:
                    con.execute(
                        f"DELETE FROM {full_table} WHERE week_start_date = ? AND city_fips = ?",
                        [week, city],
                    )
                con.execute(
                    f"INSERT INTO {full_table} BY NAME SELECT * FROM incoming_df"
                )
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
    finally:
        con.close()

    logger.info(
        f"Replaced {len(keys)} partition(s) in local {full_table} with {len(df):,} rows"
    )


def _local_add_new_columns(con, full_table: str) -> None:
    """Add incoming_df columns that full_table lacks, with DuckDB's inferred types."""
    existing = {row[0] for row in con.execute(f"DESCRIBE {full_table}").fetchall()}
    for name, col_type, *_ in con.execute(
        "DESCRIBE SELECT * FROM incoming_df"
    ).fetchall():
        if name not in existing:
            con.execute(f'ALTER TABLE {full_table} ADD COLUMN "{name}" {col_type}')
            logger.info(f"Added column {name} ({col_type}) to {full_table}")


def _bq_type(series: pd.Series) -> str:
    """BigQuery type for a new nullable column."""
    if pd.api.types.is_bool_dtype(series):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(series):
        return "INTEGER"
    if pd.api.types.is_float_dtype(series):
        return "FLOAT"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    return "STRING"


def _gcp_add_new_columns(
    df: pd.DataFrame, target: bigquery.Table, client: bigquery.Client
) -> bigquery.Table:
    """Append nullable fields for df columns the BigQuery table lacks."""
    existing = {field.name for field in target.schema}
    new = [
        bigquery.SchemaField(c, _bq_type(df[c]), mode="NULLABLE")
        for c in df.columns
        if c not in existing
    ]
    if not new:
        return target
    target.schema = [*target.schema, *new]
    logger.info(f"Adding columns {[f.name for f in new]} to {target.table_id}")
    return client.update_table(target, ["schema"])


def _gcp_replace_partition(
    df: pd.DataFrame, full_table_id: str, client: bigquery.Client
) -> None:
    """
    BigQuery: load to a staging table, then delete matching partitions and
    insert from staging in one multi-statement transaction.

    The staging table uses the target's schema, so types line up with the
    target. The first write to a new table is a plain append (no delete).
    """
    _partition_values(df)  # validates the key columns exist
    staging_id = f"{full_table_id}_staging"
    target = _gcp_add_new_columns(df, client.get_table(full_table_id), client)

    load_config = bigquery.LoadJobConfig(
        write_disposition="WRITE_TRUNCATE",
        schema=target.schema,
    )
    client.load_table_from_dataframe(df, staging_id, job_config=load_config).result()

    cols = ", ".join(f"`{c}`" for c in df.columns)
    sql = f"""
    BEGIN TRANSACTION;
    DELETE FROM `{full_table_id}` AS t
    WHERE EXISTS (
        SELECT 1 FROM `{staging_id}` AS s
        WHERE s.week_start_date = t.week_start_date
          AND s.city_fips = t.city_fips
    );
    INSERT INTO `{full_table_id}` ({cols})
    SELECT {cols} FROM `{staging_id}`;
    COMMIT TRANSACTION;
    """
    client.query(sql).result()
    client.delete_table(staging_id, not_found_ok=True)


def _gcp_table_exists(client: bigquery.Client, full_table_id: str) -> bool:
    try:
        client.get_table(full_table_id)
        return True
    except NotFound:
        return False


def write_dataframe(
    df: pd.DataFrame,
    table_name: str,
    config: dict[str, Any],
    write_disposition: str | None = None,
) -> None:
    """
    Write a pandas DataFrame to a BigQuery table.

    The full table reference is constructed automatically from config:
    {GCP_PROJECT_ID}.{bigquery_dataset}.{table_name}

    For example, if config has:
        gcp.bigquery_dataset = "chbindex_06037"
        secrets.GCP_PROJECT_ID = "my-project"
    And table_name = "epa_pm25_raw"
    Then the full table is: my-project.chbindex_06037.epa_pm25_raw

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to write. Column names become BigQuery column names.
        Must not be empty.
    table_name : str
        Name of the destination table (without project or dataset prefix).
    config : dict
        Merged pipeline config from config_loader.load_config().
    write_disposition : str, optional
        How to handle existing data in the table.
        "WRITE_APPEND"   — add new rows (default, for weekly updates)
        "WRITE_TRUNCATE" — delete all existing rows first (full reload)
        "WRITE_EMPTY"    — fail if table already has data
        "REPLACE_PARTITION" — delete rows for each (week_start_date, city_fips)
            in df, then insert df. Re-runs replace a week instead of duplicating it.
        If None, reads from config["bigquery"]["write_disposition"].

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the DataFrame is empty.
    google.api_core.exceptions.GoogleAPIError
        If the BigQuery write job fails.

    Examples
    --------
    >>> write_dataframe(epa_df, "epa_pm25_raw", config)
    >>> write_dataframe(scores_df, "compound_scores", config, write_disposition="WRITE_TRUNCATE")
    """
    if df.empty:
        logger.warning(f"DataFrame is empty — skipping write to {table_name}")
        return

    # Use write disposition from config if not explicitly provided
    if write_disposition is None:
        write_disposition = config["bigquery"]["write_disposition"]

    # Local backend needs no GCP project; check it before reading secrets
    if _is_local(config):
        if write_disposition == REPLACE_PARTITION:
            _local_replace_partition(df, table_name, config)
        else:
            _local_write(df, table_name, config, write_disposition)
        return

    # Construct full table reference (GCP only)
    project_id = config["secrets"]["GCP_PROJECT_ID"]
    dataset = config["gcp"]["bigquery_dataset"]
    full_table_id = f"{project_id}.{dataset}.{table_name}"

    logger.info(f"Writing {len(df):,} rows to {full_table_id}")
    logger.debug(f"Write disposition: {write_disposition}")
    logger.debug(f"Columns: {list(df.columns)}")

    client = get_client(config)

    if write_disposition == REPLACE_PARTITION:
        if _gcp_table_exists(client, full_table_id):
            _gcp_replace_partition(df, full_table_id, client)
            logger.info(f"Replaced partitions in {full_table_id} with {len(df):,} rows")
            return
        write_disposition = "WRITE_APPEND"  # first write creates the table

    # Configure the load job
    job_config = bigquery.LoadJobConfig(
        write_disposition=write_disposition,
        # autodetect=True lets BigQuery infer schema from the DataFrame
        # For production, you'd define explicit schemas to prevent
        # type inference errors on edge cases
        autodetect=True,
        # Partition by the field defined in pipeline.yaml
        # This keeps query costs low as the dataset grows over weeks
        time_partitioning=bigquery.TimePartitioning(
            field=config["bigquery"]["partition_field"],
        )
        if _has_partition_field(df, config)
        else None,
    )

    # Submit the load job and wait for completion
    # job.result() blocks until the job finishes or raises an exception
    job = client.load_table_from_dataframe(
        df,
        full_table_id,
        job_config=job_config,
    )
    job.result()

    logger.info(f"Successfully wrote {len(df):,} rows to {full_table_id}")


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


def read_query(
    query: str,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Execute a BigQuery SQL query and return results as a DataFrame.

    Parameters
    ----------
    query : str
        Standard SQL query string.
        Use fully qualified table names:
        SELECT * FROM `project.dataset.table`
        Or use the {dataset} placeholder which gets substituted automatically.
    config : dict
        Merged pipeline config from config_loader.load_config().

    Returns
    -------
    pd.DataFrame
        Query results. Empty DataFrame if query returns no rows.

    Examples
    --------
    >>> df = read_query(
    ...     "SELECT * FROM `my-project.chbindex_06037.compound_scores`"
    ...     " WHERE week_start_date = '2026-08-18'",
    ...     config
    ... )
    >>> len(df)
    2310
    """
    if _is_local(config):
        return _local_query(query, config)

    # Substitute {dataset} placeholder for convenience
    # Allows queries like: "SELECT * FROM {dataset}.compound_scores"
    project_id = config["secrets"]["GCP_PROJECT_ID"]
    dataset = config["gcp"]["bigquery_dataset"]
    query = query.format(
        dataset=f"{project_id}.{dataset}",
        project=project_id,
    )

    logger.info("Executing BigQuery query")
    logger.debug(f"Query: {query[:200]}...")  # log first 200 chars only

    client = get_client(config)

    # Configure query job
    job_config = bigquery.QueryJobConfig(
        # Use cached results if the same query was run recently
        # Saves query costs during development when running same queries
        use_query_cache=True,
    )

    # Execute query and convert to DataFrame
    df = client.query(query, job_config=job_config).to_dataframe()

    logger.info(f"Query returned {len(df):,} rows")
    return df


def read_table(
    table_name: str,
    config: dict[str, Any],
    columns: list[str] | None = None,
    where: str | None = None,
) -> pd.DataFrame:
    """
    Read an entire BigQuery table into a DataFrame.

    Convenience wrapper around read_query for simple table reads
    without writing full SQL.

    Parameters
    ----------
    table_name : str
        Table name without project or dataset prefix.
    config : dict
        Merged pipeline config.
    columns : list of str, optional
        Specific columns to read. If None, reads all columns.
    where : str, optional
        SQL WHERE clause (without the WHERE keyword).
        Example: "week_start_date = '2026-08-18'"

    Returns
    -------
    pd.DataFrame
        Table contents matching the filter.

    Examples
    --------
    >>> # Read all compound scores for this week
    >>> df = read_table(
    ...     "compound_scores",
    ...     config,
    ...     columns=["geoid", "compound_score", "urgency_level"],
    ...     where="week_start_date = '2026-08-18'"
    ... )
    """
    if _is_local(config):
        full_table_id = f"{{dataset}}.{table_name}"
    else:
        project_id = config["secrets"]["GCP_PROJECT_ID"]
        dataset = config["gcp"]["bigquery_dataset"]
        full_table_id = f"`{project_id}.{dataset}.{table_name}`"

    # Build SELECT clause
    select_clause = ", ".join(columns) if columns else "*"

    # Build query
    query = f"SELECT {select_clause} FROM {full_table_id}"
    if where:
        query += f" WHERE {where}"

    return read_query(query, config)


# ---------------------------------------------------------------------------
# Utility operations
# ---------------------------------------------------------------------------


def table_exists(
    table_name: str,
    config: dict[str, Any],
) -> bool:
    """
    Check whether a BigQuery table exists.

    Used before writing to decide whether to create or append.
    Prevents errors from writing to non-existent tables.

    Parameters
    ----------
    table_name : str
        Table name without project or dataset prefix.
    config : dict
        Merged pipeline config.

    Returns
    -------
    bool
        True if the table exists, False otherwise.

    Examples
    --------
    >>> if not table_exists("compound_scores", config):
    ...     logger.info("Table doesn't exist yet — will be created on first write")
    """
    if _is_local(config):
        con = _local_connect(config)
        try:
            return _local_table_exists(con, _local_schema(config), table_name)
        finally:
            con.close()

    project_id = config["secrets"]["GCP_PROJECT_ID"]
    dataset = config["gcp"]["bigquery_dataset"]
    full_table_id = f"{project_id}.{dataset}.{table_name}"

    client = get_client(config)

    try:
        client.get_table(full_table_id)
        logger.debug(f"Table {full_table_id} exists")
        return True
    except Exception:
        logger.debug(f"Table {full_table_id} does not exist")
        return False


def get_row_count(
    table_name: str,
    config: dict[str, Any],
) -> int:
    """
    Return the number of rows in a BigQuery table.

    Used for data quality checks after writes —
    confirms the expected number of rows landed correctly.

    Parameters
    ----------
    table_name : str
        Table name without project or dataset prefix.
    config : dict
        Merged pipeline config.

    Returns
    -------
    int
        Number of rows in the table. 0 if table doesn't exist.

    Examples
    --------
    >>> count = get_row_count("epa_pm25_raw", config)
    >>> logger.info(f"EPA table has {count:,} rows")
    """
    if not table_exists(table_name, config):
        return 0

    if _is_local(config):
        df = read_query(
            f"SELECT COUNT(*) AS row_count FROM {{dataset}}.{table_name}", config
        )
        return int(df["row_count"].iloc[0])

    project_id = config["secrets"]["GCP_PROJECT_ID"]
    dataset = config["gcp"]["bigquery_dataset"]

    df = read_query(
        f"SELECT COUNT(*) as row_count FROM `{project_id}.{dataset}.{table_name}`",
        config,
    )
    return int(df["row_count"].iloc[0])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _has_partition_field(
    df: pd.DataFrame,
    config: dict[str, Any],
) -> bool:
    """
    Check if the DataFrame contains the partition field.

    Only apply time partitioning if the DataFrame has the
    partition column — not all tables are time-partitioned.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame being written to BigQuery.
    config : dict
        Merged pipeline config.

    Returns
    -------
    bool
        True if partition field exists in DataFrame columns.
    """
    partition_field = config["bigquery"]["partition_field"]
    return partition_field in df.columns
