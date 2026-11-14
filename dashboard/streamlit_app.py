"""
CHBIndex dashboard (Streamlit). Reads only from the CHBIndex API.

Run from the repo root, with the API running:
    PYTHONPATH=. .venv/bin/uvicorn src.serving.api:app
    .venv/bin/streamlit run dashboard/streamlit_app.py

The week picker lives here so every page shares it; it is bound to the URL
(?week=...), as are the filters and selected tracts, so any view can be
shared as a link.
"""

import streamlit as st

from ui import api, fmt

st.set_page_config(
    page_title="CHBIndex: Los Angeles County",
    page_icon=":material/map:",
    layout="wide",
)

try:
    meta = api.meta()
except api.ApiError as exc:
    st.error(str(exc), title="The CHBIndex API is not reachable")
    st.caption(
        f"Start it with `uvicorn src.serving.api:app`, or set CHBINDEX_API_URL (now {api.API_URL})."
    )
    st.stop()

weeks = meta["weeks"][::-1]  # newest first
with st.sidebar:
    st.selectbox(
        "Week", weeks, key="week", bind="query-params", format_func=fmt.week_label
    )
    st.selectbox(
        "Compare with",
        ["previous"] + weeks,
        key="compare",
        bind="query-params",
        format_func=lambda w: (
            "The week before" if w == "previous" else fmt.week_label(w)
        ),
        help="Changes in the headline, KPI boxes and tract panel are measured against this week.",
    )

page = st.navigation(
    [
        st.Page(
            "app_pages/overview.py",
            title="Overview",
            icon=":material/map:",
            default=True,
            help="This week's burden map, KPI boxes, and a panel for any tract you select.",
        ),
    ],
    position="top",
)
page.run()
