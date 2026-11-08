"""
Tests for PDF text cleaning and reference-page filtering in ingest_docs.
No PDFs or embedding model needed.
"""

from src.rag.ingest_docs import _clean_text, _is_reference_page


def test_zero_width_space_between_words_becomes_a_space():
    raw = "This﻿section﻿provides​basic information"
    assert _clean_text(raw) == "This section provides basic information"


def test_soft_hyphen_is_removed_inside_a_word():
    assert _clean_text("respi­rator") == "respirator"


def test_reference_list_page_is_flagged():
    refs = " ".join(
        f"{i}. Author, A., et al., A study of particles, Journal {i}: 1-9 (20{i:02d})."
        for i in range(1, 15)
    )
    assert _is_reference_page(refs)


def test_body_text_with_a_few_years_is_kept():
    body = (
        "Action 3.4.c Increase street medicine outreach during extreme heat. "
        "Street medicine teams reach unhoused residents and provide water, ice, "
        "and hydration kits. In 2024, the County expanded mobile clinics, and "
        "by 2026, field teams covered more of the Antelope Valley. "
    ) * 5
    assert not _is_reference_page(body)


# ---------------------------------------------------------------------------
# pick_excerpts: one chunk per page per driver
# ---------------------------------------------------------------------------

from src.rag.briefing import pick_excerpts  # noqa: E402
from src.rag.retriever import RetrievedChunk  # noqa: E402


def _chunk(cid, page, source="plan.pdf", distance=0.3):
    return RetrievedChunk(id=cid, text="t", source=source, page=page, distance=distance)


def test_pick_excerpts_takes_one_chunk_per_page():
    ranked = [_chunk("a", 79), _chunk("b", 79), _chunk("c", 79), _chunk("d", 80), _chunk("e", 44)]
    picked = pick_excerpts([ranked], quota=3)
    assert [c.id for c in picked] == ["a", "d", "e"]


def test_pick_excerpts_same_page_number_in_different_documents_is_allowed():
    ranked = [_chunk("a", 10, "x.pdf"), _chunk("b", 10, "y.pdf")]
    assert [c.id for c in pick_excerpts([ranked], quota=3)] == ["a", "b"]


def test_pick_excerpts_skips_excluded_and_alternates_groups():
    group1 = [_chunk("a", 1), _chunk("b", 2)]
    group2 = [_chunk("c", 3), _chunk("d", 4)]
    picked = pick_excerpts([group1, group2], quota=3, exclude={"a"})
    assert [c.id for c in picked] == ["c", "b", "d"]


# ---------------------------------------------------------------------------
# risk_inputs: ranks and drivers
# ---------------------------------------------------------------------------

import pandas as pd  # noqa: E402

from src.rag import risk_inputs  # noqa: E402


def _week(n=8):
    return pd.DataFrame({
        "geoid": [f"t{i}" for i in range(n)],
        # Two tracts share a rounded percentile but not a raw burden
        "burden_percentile": [100.0, 100.0, 80.0, 70.0, 50.0, 40.0, 20.0, 10.0],
        "raw_burden": [3.21, 3.20, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0],
        "burden_class": ["CRITICAL", "CRITICAL", "HIGH", "ELEVATED", "ELEVATED", "LOW", "LOW", "LOW"],
        "pm25_z": [2.0, 0.1, 0.2, 0, 0, 0, 0, 0],
        "lst_z": [0.1, 2.0, 0.3, 0, 0, 0, 0, 0],
        "access_z": [0.0, 0.0, 1.5, 0, 0, 0, 0, 0],
        "hotspot_class": "NOT_SIGNIFICANT",
        "pm25_kriged": 9.0, "lst_mean_celsius": 45.0, "access_score_er": 1.5,
        "nearest_er_minutes": 5.0, "asthma_pct": 9.0,
    })


def test_ranks_use_raw_burden_so_rounded_percentiles_do_not_tie(monkeypatch):
    monkeypatch.setattr(risk_inputs, "_load_week", lambda config, week: _week())
    risks = {r.geoid: r for r in risk_inputs.build_tract_risks({}, "2026-06-22")}
    assert set(risks) == {"t0", "t1", "t2"}              # HIGH and CRITICAL only
    assert (risks["t0"].risk_rank, risks["t1"].risk_rank, risks["t2"].risk_rank) == (1, 2, 3)
    assert risks["t0"].urgency == "CRITICAL" and risks["t2"].urgency == "ELEVATED"
    assert risks["t0"].tract_count == 8
