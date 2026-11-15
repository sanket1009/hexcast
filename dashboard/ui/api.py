"""
API client for the dashboard.

The dashboard reads only from the CHBIndex API (src/serving/api.py), never
from the warehouse. Each call is cached with st.cache_data: weekly results
for 10 minutes (the API's own cache), static layers (hexes, outlines) for a
day. Filters and selections are applied in the app on the cached results,
so changing a filter never calls the API again.

CHBINDEX_API_URL sets the API address (default http://localhost:8000).

On Cloud Run the API is private: only the dashboard's service account may
call it. CHBINDEX_API_AUDIENCE (the API's URL) turns on Google ID tokens: each
request carries "Authorization: Bearer <token>", fetched from the Cloud Run
metadata server for the dashboard's service account and reused until shortly
before it expires (tokens last an hour). Unset locally, so nothing changes on
a laptop.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("CHBINDEX_API_URL", "http://localhost:8000").rstrip("/")
AUDIENCE = os.environ.get("CHBINDEX_API_AUDIENCE", "").rstrip("/")
TOKEN_REFRESH = 45 * 60  # seconds; Google ID tokens are valid for 60 minutes
TIMEOUT = 30
WEEK_TTL = 600
STATIC_TTL = 86_400


class ApiError(Exception):
    """The API is down or answered with an error."""


_token: dict[str, Any] = {"value": None, "at": 0.0}
_token_lock = threading.Lock()


def _fetch_id_token(audience: str) -> str:
    # Imported here so the dashboard runs locally without google-auth
    import google.auth.transport.requests
    import google.oauth2.id_token

    return google.oauth2.id_token.fetch_id_token(
        google.auth.transport.requests.Request(), audience
    )


def _auth_headers() -> dict[str, str]:
    """Bearer token for the private API, or nothing when no audience is set."""
    if not AUDIENCE:
        return {}
    with _token_lock:  # Streamlit runs each session in its own thread
        if _token["value"] is None or time.monotonic() - _token["at"] > TOKEN_REFRESH:
            try:
                _token["value"] = _fetch_id_token(AUDIENCE)
            except Exception as exc:
                raise ApiError(
                    f"Could not get an ID token for {AUDIENCE}: {exc}"
                ) from exc
            _token["at"] = time.monotonic()
        return {"Authorization": f"Bearer {_token['value']}"}


def _get(path: str, **params: Any) -> Any:
    headers = _auth_headers()
    try:
        resp = requests.get(
            f"{API_URL}{path}", params=params, headers=headers, timeout=TIMEOUT
        )
    except requests.RequestException as exc:
        raise ApiError(f"Could not reach the API at {API_URL}: {exc}") from exc
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = resp.text[:200]
        raise ApiError(f"{path}: {resp.status_code} {detail}")
    return resp.json()


@st.cache_data(ttl=WEEK_TTL, show_spinner=False)
def meta() -> dict:
    return _get("/meta")


@st.cache_data(ttl=WEEK_TTL, show_spinner=False)
def summary(week: str, compare: str | None = None) -> dict:
    return _get(f"/weeks/{week}/summary", **({"compare": compare} if compare else {}))


@st.cache_data(ttl=WEEK_TTL, show_spinner=False)
def tracts(week: str) -> pd.DataFrame:
    body = _get(f"/weeks/{week}/tracts")
    df = pd.DataFrame(body["tracts"])
    df.attrs["model_version"] = body["model_version"]
    return df


@st.cache_data(ttl=WEEK_TTL, show_spinner=False)
def tract(week: str, geoid: str) -> dict:
    return _get(f"/weeks/{week}/tracts/{geoid}")


@st.cache_data(ttl=WEEK_TTL, show_spinner=False, max_entries=64)
def trends(geoids: tuple[str, ...]) -> pd.DataFrame:
    df = pd.DataFrame(_get("/trends", geoids=",".join(geoids)))
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    return df


@st.cache_data(ttl=WEEK_TTL, show_spinner=False)
def watchlist(week: str, limit: int = 100) -> list[dict]:
    return _get(f"/weeks/{week}/watchlist", limit=limit)["tracts"]


@st.cache_data(ttl=WEEK_TTL, show_spinner=False)
def equity(week: str) -> pd.DataFrame:
    return pd.DataFrame(_get(f"/weeks/{week}/equity")["groups"])


@st.cache_data(ttl=WEEK_TTL, show_spinner=False)
def validation(week: str) -> dict:
    return _get(f"/weeks/{week}/validation")


@st.cache_data(ttl=WEEK_TTL, show_spinner=False)
def evaluation() -> dict:
    return _get("/evaluation")


@st.cache_data(ttl=STATIC_TTL, show_spinner=False)
def hexes() -> pd.DataFrame:
    return pd.DataFrame(_get("/hexes"))


@st.cache_data(ttl=STATIC_TTL, show_spinner=False)
def outlines() -> dict:
    return _get("/outlines")


@st.cache_data(ttl=STATIC_TTL, show_spinner=False, max_entries=256)
def search(q: str) -> list[dict]:
    return _get("/search", q=q, limit=8)


@st.cache_data(ttl=STATIC_TTL, show_spinner=False, max_entries=256)
def geocode(address: str) -> dict:
    return _get("/geocode", address=address)
