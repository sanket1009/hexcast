"""
briefing_runner.py
==================
Generate one week of briefings with three safeguards:

- Checkpoints: each briefing is saved to a JSONL file as soon as it finishes,
  so a crash or a stop does not waste model calls. A re-run skips tracts
  already in the checkpoint.
- Retry: a failed tract gets one more attempt. A tract that still fails is
  recorded and the run continues.
- No partial week writes: save_briefings uses REPLACE_PARTITION, which replaces
  every row for the week. commit_week refuses to write when tracts failed,
  unless the caller passes allow_partial.

Also builds a stratified sample for review and a markdown file with the
sample's summaries and checks.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, Protocol

from src.rag.briefing import Briefing, CallCapError, TractRisk
from src.rag.briefings_table import save_briefings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Indicator label used by risk_inputs.INDICATORS for healthcare access
ACCESS_LABEL = "Hospital beds per 1,000 residents, 30-minute drive"

# Sample size per urgency level: 25 tracts in total, about 25 model calls
SAMPLE_PER_URGENCY = {"CRITICAL": 13, "ELEVATED": 12}
MAX_DESERTS_IN_SAMPLE = 3


class Generator(Protocol):
    def generate(self, tract: TractRisk) -> Briefing: ...


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------


def select_sample(risks: list[TractRisk]) -> list[TractRisk]:
    """
    Pick tracts for review: up to 3 zero-access tracts per urgency level (the
    cases that test the access story), then evenly spaced tracts by geoid for
    the rest. Same input always gives the same sample.
    """
    chosen: list[TractRisk] = []
    for urgency, n in SAMPLE_PER_URGENCY.items():
        group = sorted(
            (r for r in risks if r.urgency == urgency), key=lambda r: r.geoid
        )
        deserts = [r for r in group if r.indicators.get(ACCESS_LABEL) == 0][
            :MAX_DESERTS_IN_SAMPLE
        ]
        picked = {r.geoid for r in deserts}
        rest = [r for r in group if r.geoid not in picked]

        need = min(n - len(deserts), len(rest))
        spaced: list[TractRisk] = []
        if need > 0:
            step = len(rest) / need
            spaced = [rest[int(i * step)] for i in range(need)]

        chosen.extend(deserts + spaced)
    return chosen


# ---------------------------------------------------------------------------
# Generation with checkpoints and retry
# ---------------------------------------------------------------------------


def run_week(
    risks: list[TractRisk],
    generator: Generator,
    checkpoint_path: Path,
    retries: int = 1,
) -> tuple[list[Briefing], dict[str, str]]:
    """
    Generate a briefing for each tract, skipping tracts already checkpointed.
    Returns (briefings, failures), where failures maps geoid to the last error.
    """
    done = _load_checkpoint(checkpoint_path)
    briefings = dict(done)
    failures: dict[str, str] = {}
    if done:
        logger.info(f"Checkpoint has {len(done)} briefings; skipping those tracts")

    for risk in risks:
        if risk.geoid in briefings:
            continue

        briefing, last_error = None, ""
        for attempt in range(retries + 1):
            try:
                briefing = generator.generate(risk)
                break
            except CallCapError:
                logger.error(
                    f"Model call cap reached at tract {risk.geoid}; "
                    f"{len(briefings) - len(done)} new briefings are checkpointed"
                )
                raise
            except (
                Exception
            ) as exc:  # BriefingError or API errors: record, retry, move on
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    f"Tract {risk.geoid} attempt {attempt + 1} failed: {last_error}"
                )

        if briefing is None:
            failures[risk.geoid] = last_error
            continue

        briefings[risk.geoid] = briefing
        _append_checkpoint(checkpoint_path, briefing)

    logger.info(
        f"Generated {len(briefings) - len(done)} new briefings; {len(failures)} failed"
    )
    return list(briefings.values()), failures


def commit_week(
    briefings: list[Briefing],
    failures: dict[str, str],
    week_start: date,
    config: dict[str, Any],
    allow_partial: bool = False,
):
    """Write the week to the warehouse. Refuses if tracts failed, unless allow_partial."""
    if failures and not allow_partial:
        raise RuntimeError(
            f"{len(failures)} tracts failed, so the week was not written. "
            "A partial write would replace the whole week. Fix the failures, "
            "or pass allow_partial to accept the gap."
        )
    return save_briefings(briefings, week_start, config)


# ---------------------------------------------------------------------------
# Checks and review file
# ---------------------------------------------------------------------------

# Phrases that mean the briefing is talking about how it was scored instead of
# the tract. Readers are health officials; they need conditions and actions.
SYSTEM_TALK = re.compile(
    r"\bpipeline\b|\burgency (level )?(is |was |has been )?set\b|\bset by the\b|\bscoring\b",
    re.IGNORECASE,
)


def _briefing_text(briefing: Briefing) -> str:
    parts = [briefing.summary, *briefing.key_risks, *briefing.data_gaps]
    parts += [a["action"] for a in briefing.actions]
    return "\n".join(parts)


def check_briefing(briefing: Briefing, risk: TractRisk) -> dict[str, bool]:
    return {
        "no talk about the scoring system": not SYSTEM_TALK.search(
            _briefing_text(briefing)
        ),
        "urgency matches burden class": briefing.urgency == risk.urgency,
        "summary written": bool(briefing.summary.strip()),
        "at least one grounded action": len(briefing.actions) > 0,
        "every action cites a source": all(a["citations"] for a in briefing.actions),
        "policy excerpts were retrieved": len(briefing.retrieved_chunk_ids) > 0,
    }


def write_review(
    path: Path,
    week_str: str,
    risks: list[TractRisk],
    briefings: list[Briefing],
    failures: dict[str, str],
) -> None:
    """Markdown file for reading the sample: summaries, actions, and checks per tract."""
    by_geoid = {r.geoid: r for r in risks}
    lines = [f"# Briefing sample, week of {week_str}", ""]

    passed = 0
    checks_by_geoid: dict[str, dict[str, bool]] = {}
    for b in briefings:
        checks_by_geoid[b.geoid] = check_briefing(b, by_geoid[b.geoid])
        passed += all(checks_by_geoid[b.geoid].values())
    lines.append(
        f"Tracts: {len(briefings)} written, {len(failures)} failed. "
        f"{passed} of {len(briefings)} passed every check."
    )
    lines.append("")

    for b in sorted(briefings, key=lambda x: x.geoid):
        risk = by_geoid[b.geoid]
        failed = [name for name, ok in checks_by_geoid[b.geoid].items() if not ok]
        lines += [
            f"## {b.geoid}: {b.urgency}, rank {risk.risk_rank} of {risk.tract_count}",
            f"Drivers: {', '.join(risk.drivers)}",
            f"Checks: {'all passed' if not failed else 'FAILED: ' + '; '.join(failed)}",
            "",
            b.summary,
            "",
            "Key risks:",
            *[f"- {k}" for k in b.key_risks],
            "",
            "Actions:",
            *[
                f"- {'(' + a['group'] + ') ' if a.get('group') else ''}"
                f"{a['action']} [{', '.join(a['citations'])}]"
                for a in b.actions
            ],
            "",
            f"Data gaps: {'; '.join(b.data_gaps) if b.data_gaps else 'none'}",
            "",
        ]

    if failures:
        lines += [
            "## Failed tracts",
            *[f"- {g}: {err}" for g, err in failures.items()],
            "",
        ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Checkpoint files
# ---------------------------------------------------------------------------


def _load_checkpoint(path: Path) -> dict[str, Briefing]:
    if not path.exists():
        return {}
    out: dict[str, Briefing] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            out[record["geoid"]] = Briefing(**record)
    return out


def _append_checkpoint(path: Path, briefing: Briefing) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(asdict(briefing)) + "\n")
