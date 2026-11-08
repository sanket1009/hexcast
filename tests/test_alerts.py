"""Tests for the weekly burden report: change detection, grouping, rendering. No warehouse."""

import csv

from src.alerts.digest import (
    LST,
    MOVER_RANKS,
    ActionGroup,
    Digest,
    TractRow,
    group_actions,
    render_html,
    render_subject,
    source_title,
    summary_lines,
    write_csv,
)


def _row(geoid, rank, cls, prev_rank=None, prev_cls=None, community="City of Pomona",
         region="San Gabriel Valley", pop=4000, drivers=("heat",)):
    return TractRow(
        geoid=geoid, rank=rank, prev_rank=prev_rank, burden_class=cls, prev_class=prev_cls,
        population=pop, region=region, community=community, drivers=list(drivers),
        indicators={LST: 50.0 + rank / 100},
    )


def _digest(rows, has_previous=True, actions=()):
    return Digest(
        city_name="Los Angeles", week="2026-06-29", tract_count=len(rows),
        class_counts={}, rows=rows, has_previous_week=has_previous,
        has_briefings=bool(actions), actions=list(actions),
    )


def test_change_detection():
    d = _digest([
        _row("new", 5, "CRITICAL", prev_rank=300, prev_cls="HIGH"),
        _row("stay", 1, "CRITICAL", prev_rank=2, prev_cls="CRITICAL"),
        _row("mover", 10, "CRITICAL", prev_rank=10 + MOVER_RANKS, prev_cls="CRITICAL"),
        _row("left", 400, "HIGH", prev_rank=20, prev_cls="CRITICAL"),
        _row("low", 2000, "LOW", prev_rank=1900, prev_cls="LOW"),
    ])
    assert [r.geoid for r in d.critical] == ["stay", "new", "mover"]
    assert [r.geoid for r in d.newly_critical] == ["new"]
    assert [r.geoid for r in d.left_critical] == ["left"]
    assert [r.geoid for r in d.movers] == ["mover"]


def test_subject_says_what_changed():
    first = _digest([_row("a", 1, "CRITICAL")], has_previous=False)
    assert render_subject(first).endswith("first week on record")

    quiet = _digest([_row("a", 1, "CRITICAL", prev_rank=1, prev_cls="CRITICAL")])
    assert render_subject(quiet).endswith("no change in critical areas")

    busy = _digest([
        _row("a", 1, "CRITICAL", prev_rank=50, prev_cls="HIGH"),
        _row("b", 2, "CRITICAL", prev_rank=60, prev_cls="HIGH"),
        _row("c", 500, "HIGH", prev_rank=3, prev_cls="CRITICAL"),
    ])
    assert render_subject(busy).endswith("2 tracts newly critical, 1 left")


def test_summary_counts_residents_and_names_the_community():
    d = _digest([
        _row("a", 1, "CRITICAL", prev_rank=50, prev_cls="HIGH", community="City of Covina", pop=3000),
        _row("b", 2, "CRITICAL", prev_rank=60, prev_cls="HIGH", community="City of Covina", pop=2500),
    ])
    first = summary_lines(d)[0]
    assert "2 tracts became critical this week (5,500 residents)" in first
    assert "City of Covina (San Gabriel Valley), mainly heat" in first


def test_html_groups_changes_by_community_and_escapes_text():
    d = _digest(
        [_row("a", 1, "CRITICAL", prev_rank=50, prev_cls="HIGH", community="A <b>&"),
         _row("b", 2, "CRITICAL", prev_rank=60, prev_cls="HIGH", community="A <b>&")],
        actions=[ActionGroup("la_county_heat_action_plan_2026.pdf, p. 10", "Expand outreach.", 2)],
    )
    body = render_html(d)
    assert "Newly critical" in body
    assert "A &lt;b&gt;&amp;" in body and "<b>&" not in body
    assert "LA County Heat Action Plan 2026, p. 10" in body
    assert body.count("A &lt;b&gt;&amp;") >= 1


def test_group_actions_counts_tracts_per_source_and_keeps_shortest_wording():
    groups = group_actions({
        "a": [{"action": "Expand cooling center outreach in the tract.", "citations": ["plan.pdf, p. 10"]}],
        "b": [{"action": "Expand cooling outreach.", "citations": ["plan.pdf, p. 10"]},
              {"action": "Plant trees.", "citations": ["state.pdf, p. 60"]}],
    })
    assert groups[0].citation == "plan.pdf, p. 10"
    assert groups[0].tracts == 2
    assert groups[0].action == "Expand cooling outreach."


def test_csv_lists_critical_and_left_tracts_with_change(tmp_path):
    d = _digest([
        _row("new", 5, "CRITICAL", prev_rank=300, prev_cls="HIGH"),
        _row("left", 400, "HIGH", prev_rank=20, prev_cls="CRITICAL"),
        _row("low", 2000, "LOW"),
    ])
    rows = list(csv.DictReader(open(write_csv(d, tmp_path / "r.csv"), encoding="utf-8")))
    assert [(r["geoid"], r["change"]) for r in rows] == [("new", "newly critical"), ("left", "left critical")]


def test_source_title():
    assert source_title("carb_roadway_exposure_technical_advisory_2017.pdf, p. 21") == \
        "CARB Roadway Exposure Technical Advisory 2017, p. 21"
