"""
faithfulness.py
===============
A2: does the briefing say only what the tract data says?

Checks, one scorer each, all free (no model calls):
    numbers_match_data   every number in the tract text is in the tract data
    urgency_matches      stored urgency equals the input, and the text agrees
    drivers_mentioned    share of the tract's drivers the summary/risks name
    drivers_accounted    each driver has an action or is listed as a data gap

Vulnerable groups are not checked yet: every input uses the default group
until the Census ACS module exists.
"""

import re

from src.evals.base import DRIVER_PATTERNS, Score, action_text, tract_text
from src.evals.dataset import EvalCase

# Numbers like 2,498 / 55.61 / 10 / -0.4. Not preceded by a letter or dot, so
# "PM2.5" is not read as the number 2.5.
NUMBER = re.compile(
    r"(?<![\w.])-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|(?<![\w.])-?\d+(?:\.\d+)?"
)

# The prompt says "(1 = highest burden)", so the model may repeat the 1.
PROMPT_CONSTANTS = {1.0}

URGENCY_WORDS = ("CRITICAL", "ELEVATED", "MONITOR")
# Urgency stated as a level: "rated CRITICAL", "critical urgency",
# "urgency level is elevated", "at elevated risk". Plain uses such as
# "elevated temperatures" or "elevated risk of heat illness" do not match.
URGENCY_CLAIM = re.compile(
    r"\b(?:rated|urgency(?: level)?(?: is| of)?|is at|at an?)\s+(critical|elevated|monitor)\b"
    r"|\b(critical|elevated|monitor)\s+(?:urgency|risk|level)\b(?!\s+(?:of|for)\b)",
    re.IGNORECASE,
)


def numbers_match_data(case: EvalCase) -> Score:
    """1.0 when every number in the tract text comes from the tract data."""
    b, t = case.briefing, case.tract
    text = tract_text(b).replace(t.geoid, "")

    allowed = {float(t.risk_rank), float(t.tract_count), *PROMPT_CONSTANTS}
    for label in t.indicators:  # numbers inside labels: "30-minute", "1,000"
        allowed |= {_to_float(n) for n in NUMBER.findall(label)}

    unmatched = [
        n
        for n in NUMBER.findall(text)
        if _to_float(n) not in allowed
        and not any(_same_when_rounded(n, v) for v in t.indicators.values())
    ]
    if unmatched:
        return Score(0.0, f"not in tract data: {', '.join(unmatched)}")
    return Score(1.0, "")


def urgency_matches(case: EvalCase) -> Score:
    """1.0 when the stored urgency is the input's and the text claims no other level."""
    b, t = case.briefing, case.tract
    if b.urgency != t.urgency:
        return Score(0.0, f"stored {b.urgency}, input {t.urgency}")

    claimed = {
        (m.group(1) or m.group(2)).upper()
        for m in URGENCY_CLAIM.finditer(tract_text(b))
    }
    wrong = sorted(claimed - {t.urgency})
    if wrong:
        return Score(0.0, f"text calls it {', '.join(wrong)}, input {t.urgency}")
    return Score(1.0, "")


def drivers_mentioned(case: EvalCase) -> Score:
    """Share of the tract's drivers named in the summary or key risks (recall)."""
    b, t = case.briefing, case.tract
    text = "\n".join([b.summary, *b.key_risks])
    missing = [d for d in t.drivers if not DRIVER_PATTERNS[d].search(text)]
    value = 1 - len(missing) / len(t.drivers)
    return Score(value, f"not mentioned: {', '.join(missing)}" if missing else "")


def drivers_accounted(case: EvalCase) -> Score:
    """1.0 when every driver has an action about it or is listed in data_gaps."""
    b, t = case.briefing, case.tract
    actions, gaps = action_text(b), "\n".join(b.data_gaps)
    dropped = [
        d
        for d in t.drivers
        if not DRIVER_PATTERNS[d].search(actions)
        and not DRIVER_PATTERNS[d].search(gaps)
    ]
    if dropped:
        return Score(0.0, f"no action and no data gap for: {', '.join(dropped)}")
    return Score(1.0, "")


SCORERS = {
    "numbers_match_data": numbers_match_data,
    "urgency_matches": urgency_matches,
    "drivers_mentioned": drivers_mentioned,
    "drivers_accounted": drivers_accounted,
}


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------


def _to_float(number: str) -> float:
    return float(number.replace(",", ""))


def _same_when_rounded(number: str, value: float) -> bool:
    """True when `value` rounded to the number's decimals equals it (55.61 -> "55.6")."""
    decimals = len(number.split(".")[1]) if "." in number else 0
    return round(value, decimals) == _to_float(number)
