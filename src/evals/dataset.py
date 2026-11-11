"""
dataset.py
==========
Load what the evals score: each stored briefing, paired with the tract data
the model was given when it wrote that briefing.

The briefings table stores the model's output but not its input, so the
input is rebuilt from the warehouse with build_tract_risks, the same function
the briefing run used.

Rank caveat:
    Briefings written before commit 32d3f38 (2026-11-08) were ranked on
    burden_percentile (rounded to 0.1, so ties share a rank). Later runs rank
    on raw_burden. rank_by="burden_percentile" rebuilds the older rank, which
    is what the stored 2026-06-22 briefings saw.

Briefings can also be read from JSONL files that carry their own input
(Briefing.inputs): a run's full_checkpoint.jsonl, which is what the warehouse
rows were written from, or a frozen eval sample (judge sample, CI fixtures).
No warehouse is needed for those.

Usage:
    cases = load_cases(config, "2026-06-22", rank_by="burden_percentile")
    cases[0].briefing.summary, cases[0].tract.indicators

    cases = load_cases_jsonl("data/briefing_runs/2026-06-22_groups/full_checkpoint.jsonl")
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.rag.briefing import Briefing, TractRisk
from src.rag.briefings_table import BRIEFINGS_TABLE
from src.rag.risk_inputs import build_tract_risks
from src.utils.bigquery import read_query
from src.utils.logger import get_logger

logger = get_logger(__name__)

RankBy = Literal["raw_burden", "burden_percentile"]


@dataclass(frozen=True)
class EvalCase:
    """One briefing and the tract data it was written from."""

    briefing: Briefing
    tract: TractRisk


def load_cases(
    config: dict[str, Any], week_str: str, rank_by: RankBy = "raw_burden"
) -> list[EvalCase]:
    """Every stored briefing for the week, paired with its rebuilt tract input."""
    briefings = load_briefings(config, week_str)
    tracts = {t.geoid: t for t in build_tract_risks(config, week_str)}

    if rank_by == "burden_percentile":
        ranks = _percentile_ranks(config, week_str)
        tracts = {g: replace(t, risk_rank=ranks[g]) for g, t in tracts.items()}

    # Every briefing must have an input and every input a briefing. A mismatch
    # means the warehouse changed since the run, and the evals would be wrong.
    missing_input = sorted(set(b.geoid for b in briefings) - set(tracts))
    missing_briefing = sorted(set(tracts) - set(b.geoid for b in briefings))
    if missing_input or missing_briefing:
        raise ValueError(
            f"Week {week_str}: {len(missing_input)} briefings without input, "
            f"{len(missing_briefing)} inputs without briefing "
            f"(e.g. {(missing_input + missing_briefing)[:3]})"
        )

    cases = [EvalCase(b, tracts[b.geoid]) for b in briefings]
    logger.info(
        f"Loaded {len(cases)} eval cases for week {week_str} (rank_by={rank_by})"
    )
    return cases


def load_cases_jsonl(path: str | Path) -> list[EvalCase]:
    """Cases from a JSONL file of Briefing records that include their input.

    Accepts plain Briefing records (checkpoint files) or records with the
    Briefing under a "briefing" key (frozen eval samples).
    """
    cases = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        record = record.get("briefing", record)
        if not record.get("inputs"):
            raise ValueError(f"{path}: briefing {record['geoid']} has no inputs")
        briefing = Briefing(**record)
        cases.append(EvalCase(briefing, TractRisk(**record["inputs"])))
    return sorted(cases, key=lambda c: c.briefing.geoid)


def load_briefings(config: dict[str, Any], week_str: str) -> list[Briefing]:
    """Stored briefings for one week and city, as Briefing objects."""
    df = read_query(
        f"""
        SELECT *
        FROM {{dataset}}.{BRIEFINGS_TABLE}
        WHERE week_start_date = '{week_str}'
          AND city_fips = '{config["full_fips"]}'
        ORDER BY geoid
        """,
        config,
    )
    if df.empty:
        raise ValueError(f"No stored briefings for week {week_str}")
    return [_row_to_briefing(row) for _, row in df.iterrows()]


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------


def _row_to_briefing(row: pd.Series) -> Briefing:
    """Undo briefings_table._briefing_row: parse the JSON text columns back."""
    return Briefing(
        geoid=row["geoid"],
        urgency=row["urgency"],
        summary=row["summary"],
        key_risks=json.loads(row["key_risks_json"]),
        actions=json.loads(row["actions_json"]),
        data_gaps=json.loads(row["data_gaps_json"] or "[]"),
        sources=json.loads(row["sources_json"] or "[]"),
        retrieved_chunk_ids=json.loads(row["retrieved_chunk_ids_json"] or "[]"),
        model=row["model"],
        generated_at=str(row["generated_at"]),
    )


def _percentile_ranks(config: dict[str, Any], week_str: str) -> dict[str, int]:
    """Burden rank the pre-32d3f38 way: on burden_percentile, ties share the lowest rank."""
    df = read_query(
        f"""
        SELECT geoid, burden_percentile
        FROM {{dataset}}.compound_scores
        WHERE week_start_date = '{week_str}'
          AND city_fips = '{config["full_fips"]}'
        """,
        config,
    )
    ranks = df["burden_percentile"].rank(ascending=False, method="min").astype(int)
    return dict(zip(df["geoid"], ranks, strict=True))
