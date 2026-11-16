"""
Methods: short decision stories, each linked to its README section.

Static text on purpose: these are the decisions behind the pipeline and the
results that settled them, recorded once. Live numbers (model scores, weekly
checks) are on the Validation page instead of being repeated here.

CHBINDEX_REPO_URL overrides the repository link.
"""

from __future__ import annotations

import os

import streamlit as st

REPO = os.environ.get("CHBINDEX_REPO_URL", "https://github.com/sanket1009/hexcast")

# (title, story, README anchor or None for the README itself)
STORIES = [
    (
        "PM2.5 between 21 monitors",
        "Tract PM2.5 is interpolated with ordinary Kriging from about 21 regulatory monitors. "
        "A zero nugget made the surface pass exactly through every monitor and overfit, so the "
        "nugget has a floor set by the sampling noise of a weekly mean. Leave-one-out validation "
        "refits the variogram in every fold, and the error is reported as MAE: R² over 21 points "
        "mostly reflects how far apart the monitors are.",
        "pm25-kriging-with-an-honest-error-estimate",
    ),
    (
        "Ocean pixels in the heat layer",
        "Landsat surface temperature averaged over tract polygons pulled 39 coastal tracts about "
        "14 °C too cool, because cold ocean pixels fell inside their boundaries. Water is now "
        "masked with the QA water bit and a majority-vote land mask, and tracts that are only "
        "water get no heat value.",
        "heat-land-surface-temperature",
    ),
    (
        "Drive times that reach every road",
        "One tract looked like an emergency-care desert because its origin snapped to a road "
        "segment cut off from the rest of the network; origins now snap to the largest strongly "
        "connected component. Mountain tracts dropped out until origins moved from polygon "
        "centroids to population-weighted centers. A 30 km buffer of hospitals and population "
        "credits the 56 border tracts whose nearest emergency department is in another county.",
        "access-to-emergency-care-e2sfca",
    ),
    (
        "A composite that knows which way is bad",
        "Spatial PCA put healthcare access on the burden side (correlation +0.86 with the first "
        "component), so the three tracts with no emergency department within 30 minutes scored "
        "LOW. PCA cannot be told which direction is bad. The index is a signed z-score composite "
        "instead: heat and PM2.5 raise burden, access lowers it, and a run fails validation if "
        "burden rises with access. Population vulnerability from the ACS joined later as a "
        "fourth equal-weight component, added rather than multiplied with the hazards.",
        "compound-burden-score",
    ),
    (
        "A four-week PM2.5 window",
        "Weekly PM2.5 made the CRITICAL list churn from one week to the next. The index uses a "
        "4-week rolling mean of PM2.5 instead, and the drift checks track how much the CRITICAL "
        "list overlaps with the week before.",
        None,
    ),
    (
        "Weights left alone after the CalEnviroScreen benchmark",
        "The index was compared with CalEnviroScreen, component by component. The weights were "
        "not tuned to match it: the comparison is reported as it is, on the Validation page, "
        "because matching another index is not the goal.",
        "evaluation",
    ),
    (
        "Confidence from correlated error",
        "An early version drew Kriging error independently for each tract and said 98% of "
        "tracts could change class. Kriging errors are correlated in space, so each of the 200 "
        "draws now samples the full tract-by-tract error covariance from the week's variogram, "
        "plus ±10% on the component weights. The weights barely matter; PM2.5 interpolation "
        "drives almost all of the doubt.",
        None,
    ),
    (
        "Drift checks that do not cry wolf",
        "A PSI on raw PM2.5 or heat flagged every one of the 97 checkable weeks as a major "
        "shift, because the county-wide level moves with the weather. The checks now separate "
        '"did the whole county move" from "did the spatial pattern change", with thresholds '
        "set from the 2024-2026 history. On that history the major flags land on real events: "
        "smoke weeks, cloud-masked heat weeks, and holiday Kriging errors.",
        None,
    ),
    (
        "Forecasting entry, not next week's class",
        "The watch-list model predicts whether a tract that is not CRITICAL now enters CRITICAL "
        "within 4 weeks. A second year of data was backfilled first, so the training years "
        "include a spring. Cross-validation holds out later weeks and whole spatial blocks at "
        "once, so neighboring tracts cannot leak labels. Logistic regression beat tuned XGBoost "
        "and monotone LightGBM, and its terms add up exactly to its log-odds, which is how the "
        "dashboard shows why a tract is flagged without a second model.",
        None,
    ),
    (
        "Retrieval chosen by an eval",
        "A hand-labeled gold set of policy questions decided the embedding model: production "
        "retrieval moved from all-MiniLM-L6-v2 to Qwen3-Embedding-0.6B after it. The briefings "
        "are checked without model calls for numbers that are not in the tract data, talk about "
        "the pipeline, and actions for hazards the tract does not have. An early prompt had 24 "
        "of 25 sample briefings narrating how their urgency was set.",
        "evaluation",
    ),
    (
        "Hexes where people live",
        "Area maps let large, empty rural tracts dominate. The map paints H3 cells only where "
        "Census 2020 blocks have residents, each colored by its main tract. Resolution 8 left "
        "448 small, dense tracts (1.38M people) without a cell of their own; resolution 9 "
        "leaves 6 (10,250 people). Whole-tract fill is still one click away to show the "
        "difference.",
        "limitations",
    ),
]

LIMITS = [
    "Kriging uses distance only, so it misses terrain barriers (for example near Lancaster, Thousand Oaks and Simi Valley).",
    "Drive times are free-flow: they understate real trips and overstate access near downtown freeway hubs.",
    "Heat is a 32-day Landsat composite, not a single week.",
    "Regulatory PM2.5 data arrives months late, so the newest weeks are not real time.",
    "Briefings know only what five policy documents say, so many tracts get similar actions.",
    "Median income of 250,001 is the ACS top code.",
]


def readme(anchor: str | None) -> str:
    return f"{REPO}#{anchor}" if anchor else f"{REPO}#readme"


st.header("How the index is built, and what changed along the way")
st.caption(
    "Each note is a decision and the result that settled it. Model scores and weekly checks "
    "are on the Validation page; the full write-up is in the README."
)

stories, limits = st.columns([2, 1], gap="large")
with stories:
    for title, story, anchor in STORIES:
        st.subheader(title)
        st.markdown(story)
        st.markdown(f"[Read more in the README]({readme(anchor)})")
with limits:
    st.subheader("What the index cannot tell you")
    st.markdown("\n".join(f"- {line}" for line in LIMITS))
    st.markdown(f"[Limitations in the README]({readme('limitations')})")
