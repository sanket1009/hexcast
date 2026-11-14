"""
Serving modules for CHBIndex.

Read-only REST API the dashboard reads from:
    store.py    warehouse access (DuckDB read_only locally, BigQuery in production)
    service.py  queries and the logic on top (drivers, KPIs, equity)
    schemas.py  Pydantic response models (also the /docs contract)
    api.py      FastAPI endpoints
"""
