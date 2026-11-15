"""
api.py
======
CHBIndex REST API (FastAPI). The Streamlit dashboard reads only from here.

Run locally:
    PYTHONPATH=. .venv/bin/uvicorn src.serving.api:app --reload
    open http://localhost:8000/docs

Environment:
    CHBINDEX_CITY      city config to load (default los_angeles)
    CHBINDEX_DB_PATH   local DuckDB file, overriding storage.local_db_path
    CHBINDEX_BACKEND   "gcp" to read BigQuery (Cloud Run), overriding storage.backend
    GCP_PROJECT_ID     BigQuery project when the backend is gcp

Endpoints (week is YYYY-MM-DD or "latest"):
    GET /health                         status, backend, newest week
    GET /meta                           weeks, classes, components
    GET /weeks/{week}/summary           KPI boxes, headline, data sources
    GET /weeks/{week}/tracts            every tract with every map layer
    GET /weeks/{week}/tracts/{geoid}    tract panel: + ML contributions, ACS, briefing
    GET /weeks/{week}/watchlist         top tracts by probability of entering CRITICAL
    GET /weeks/{week}/equity            share of each group living in CRITICAL tracts
    GET /weeks/{week}/validation        drift checks, CES agreement, weekly pipeline health
    GET /evaluation                     watch-list model and RAG evaluation results
    GET /trends?geoids=...              weekly burden series for up to 20 tracts
    GET /hexes                          populated H3 cells and their main tract
    GET /outlines                       tract outlines, GeoJSON
    GET /search?q=...                   GEOID, city, SPA or community name
    GET /geocode?address=...            address -> tract (Census Geocoder)

All reads; nothing here writes to the warehouse or calls a language model.
"""

from __future__ import annotations

import os
from datetime import date
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Query, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response

from src.serving import schemas
from src.serving.service import (
    CLASSES,
    COMPONENT_LABELS,
    NotFoundError,
    Service,
    UpstreamError,
    records,
)
from src.serving.store import make_store

app = FastAPI(
    title="CHBIndex API",
    summary="Weekly compound climate-health burden for census tracts",
    version="1.0.0",
)
# Tract tables and outlines are large and repetitive JSON; gzip cuts them ~4x
app.add_middleware(GZipMiddleware, minimum_size=1000)


@lru_cache
def get_service() -> Service:
    """One Service per process, built on first request (tests override this)."""
    from src.utils.config_loader import load_config

    config = load_config(os.environ.get("CHBINDEX_CITY", "los_angeles"))
    if db_path := os.environ.get("CHBINDEX_DB_PATH"):
        config["storage"]["local_db_path"] = db_path
    # Cloud Run sets CHBINDEX_BACKEND=gcp; the laptop keeps pipeline.yaml's "local"
    if backend := os.environ.get("CHBINDEX_BACKEND"):
        config["storage"]["backend"] = backend
    return Service(make_store(config), config)


Svc = Annotated[Service, Depends(get_service)]
WeekParam = Annotated[str, "YYYY-MM-DD or latest"]


@app.exception_handler(NotFoundError)
async def not_found(_: Request, exc: NotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.exception_handler(UpstreamError)
async def upstream(_: Request, exc: UpstreamError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": str(exc)})


@app.get("/health", response_model=schemas.Health)
def health(svc: Svc) -> dict:
    return {
        "status": "ok",
        "backend": svc.store.backend,
        "latest_week": svc.resolve_week("latest"),
    }


@app.get("/meta", response_model=schemas.Meta)
def meta(svc: Svc) -> dict:
    weeks = svc.weeks()
    cfg = svc.config
    return {
        "city_fips": svc.city,
        "city_name": cfg.get("display_name") or cfg.get("city_name") or svc.city,
        "weeks": weeks,
        "latest_week": weeks[-1],
        "classes": CLASSES,
        "components": COMPONENT_LABELS,
        "watchlist_threshold": svc.watchlist_threshold,
    }


@app.get("/weeks/{week}/summary", response_model=schemas.Summary)
def summary(
    svc: Svc,
    week: WeekParam,
    compare: Annotated[
        date | None, Query(description="Week to compare with; default the week before")
    ] = None,
) -> dict:
    w = svc.resolve_week(week)
    c = svc.resolve_week(compare) if compare else None
    return svc.summary(w, c)


@app.get("/weeks/{week}/tracts", response_model=schemas.WeekTracts)
def week_tracts(svc: Svc, week: WeekParam) -> dict:
    w = svc.resolve_week(week)
    df = svc.tracts(w)
    versions = df["model_version"].dropna()
    return {
        "week": w,
        "model_version": versions.iloc[0] if len(versions) else None,
        "tracts": records(df.drop(columns=["model_version", "week_start_date"])),
    }


@app.get("/weeks/{week}/tracts/{geoid}", response_model=schemas.TractDetail)
def tract(
    svc: Svc,
    week: WeekParam,
    geoid: str,
    top_features: Annotated[int, Query(ge=1, le=33)] = 8,
) -> dict:
    return svc.tract(geoid, svc.resolve_week(week), top_features)


@app.get("/weeks/{week}/watchlist", response_model=schemas.Watchlist)
def watchlist(
    svc: Svc,
    week: WeekParam,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
    min_prob: Annotated[float | None, Query(ge=0, le=1)] = None,
) -> dict:
    w = svc.resolve_week(week)
    return {"week": w, "tracts": svc.watchlist(w, limit, min_prob)}


@app.get("/weeks/{week}/equity", response_model=schemas.Equity)
def equity(svc: Svc, week: WeekParam) -> dict:
    w = svc.resolve_week(week)
    return {"week": w, "groups": svc.equity(w)}


@app.get("/weeks/{week}/validation", response_model=schemas.Validation)
def validation(svc: Svc, week: WeekParam) -> dict:
    return svc.validation(svc.resolve_week(week))


@app.get("/evaluation", response_model=schemas.Evaluation)
def evaluation(svc: Svc) -> dict:
    return svc.evaluation()


@app.get("/trends", response_model=list[schemas.TrendPoint])
def trends(
    svc: Svc,
    geoids: Annotated[list[str], Query(min_length=1, max_length=20)],
    first: date | None = None,
    last: date | None = None,
) -> list[dict]:
    # Accept both ?geoids=a&geoids=b and ?geoids=a,b
    ids = [g for item in geoids for g in item.split(",") if g]
    return records(svc.trends(ids, first, last))


@app.get("/hexes", response_model=schemas.Hexes)
def hexes(svc: Svc, response: Response) -> dict:
    response.headers["Cache-Control"] = "public, max-age=86400"
    df = svc.hexes()
    return {
        "h3_index": df["h3_index"].tolist(),
        "geoid": df["geoid"].tolist(),
        "population": df["population"].astype(int).tolist(),
    }


@app.get(
    "/outlines",
    response_class=Response,
    responses={200: {"content": {"application/geo+json": {}}}},
)
def outlines(svc: Svc) -> Response:
    return Response(
        svc.outlines_geojson(),
        media_type="application/geo+json",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/search", response_model=list[schemas.SearchHit])
def search(
    svc: Svc,
    q: Annotated[str, Query(min_length=2, max_length=80)],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
) -> list[dict]:
    return svc.search(q, limit)


@app.get("/geocode", response_model=schemas.GeocodeHit)
def geocode(
    svc: Svc, address: Annotated[str, Query(min_length=5, max_length=200)]
) -> dict:
    return svc.geocode(address)
