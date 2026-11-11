"""
rules.py
========
A4: does the briefing follow the rules in SYSTEM_PROMPT (src/rag/briefing.py)?

One scorer per rule, all free (no model calls):
    no_source_named     tract text names no pollution or heat source
    no_system_talk      no mention of the pipeline, model, scoring or z-scores
    no_jargon           no technical terms a non-technical reader would stumble on
    reading_grade       Flesch-Kincaid grade of the whole briefing (target < 12)
    on_driver_actions   no action is about a hazard the tract does not have
                        (Phase 5.1: access plans carried extreme-heat actions)
    concrete_actions    share of actions that do not open with a vague verb
                        (encourage, tell, advise, promote, share, consider)

"Uses only the given numbers" is checked in faithfulness.py (numbers_match_data).
"""

import re

from src.evals.base import Score, action_text, tract_text
from src.evals.dataset import EvalCase

# Rule: "Do not name a source (such as traffic, freeways, or industry) for this
# tract". Checked on the summary and key risks, where the briefing makes claims
# about the tract. Actions and data gaps are left out: they describe what the
# excerpts cover, and excerpts may discuss sources in general terms.
SOURCE_TERMS = re.compile(
    r"\b(traffic|freeways?|highways?|busy roads?|roadways?|ports?|refiner(y|ies)|"
    r"industr(y|ies|ial)|factor(y|ies)|warehouses?|rail ?yards?|airports?|"
    r"oil (wells?|fields?)|diesel|trucks?|wildfires?|smoke|urban heat island)\b",
    re.IGNORECASE,
)

# Rule: "Never mention the pipeline, the model, the scoring, or how the
# urgency or rank was decided." Extends briefing_runner.SYSTEM_TALK.
SYSTEM_TERMS = re.compile(
    r"\b(pipeline|scoring|scored|z-?scores?|percentiles?|algorithm|"
    r"burden (index|score)|compound (index|score)|our (model|analysis)|"
    r"urgency (level )?(is |was |has been )?set|set by the|classified as)\b",
    re.IGNORECASE,
)

# Rule: "Write plainly for a non-technical reader." Terms that leak in from the
# data labels or the methods.
JARGON_TERMS = re.compile(
    r"\b(kriged|kriging|interpolat\w*|catchment|2SFCA|geoid|LST|"
    r"spatial(ly)?|covariates?|standard deviations?)\b",
    re.IGNORECASE,
)

READING_GRADE_TARGET = 12.0

# Phase 5.1 rules in PLAN_SYSTEM_PROMPT. Hazard words only: healthcare access
# has no hazard vocabulary, and general words such as "cool" or "filter" also
# appear in on-driver actions.
HAZARD_TERMS = {
    "heat": re.compile(
        r"\b(extreme heat|heat ?waves?|heat[- ]related|heat risks?|heat illness|"
        r"heat safety|cooling cent(er|re)s?|hot weather|during heat)\b",
        re.IGNORECASE,
    ),
    "air_quality": re.compile(
        r"\b(pm2\.5|particles?|particulate|air pollution|air quality|smoke|"
        r"air filt\w*|air purifiers?|air cleaners?)\b",
        re.IGNORECASE,
    ),
}
VAGUE_LEAD = re.compile(
    r"^\s*(encourage|tell|advise|promote|share|consider|raise awareness|remind|urge)\b",
    re.IGNORECASE,
)


def no_source_named(case: EvalCase) -> Score:
    b = case.briefing
    return _no_match(
        SOURCE_TERMS, "\n".join([b.summary, *b.key_risks]), "names a source"
    )


def no_system_talk(case: EvalCase) -> Score:
    return _no_match(SYSTEM_TERMS, _all_text(case), "talks about the system")


def no_jargon(case: EvalCase) -> Score:
    return _no_match(JARGON_TERMS, _all_text(case), "jargon")


def reading_grade(case: EvalCase) -> Score:
    """Flesch-Kincaid grade level. Not pass/fail: lower is easier to read."""
    grade = flesch_kincaid_grade(_all_text(case))
    note = (
        f"above grade {READING_GRADE_TARGET:g}" if grade >= READING_GRADE_TARGET else ""
    )
    return Score(round(grade, 1), note)


def on_driver_actions(case: EvalCase) -> Score:
    """1.0 when no action names a hazard outside the tract's drivers."""
    off = sorted(
        {
            hazard
            for a in case.briefing.actions
            for hazard, pattern in HAZARD_TERMS.items()
            if hazard not in case.tract.drivers and pattern.search(a["action"])
        }
    )
    return Score(0.0, f"actions about {', '.join(off)}") if off else Score(1.0, "")


def concrete_actions(case: EvalCase) -> Score:
    """Share of actions that do not open with a vague verb (1.0 when none do)."""
    actions = [a["action"] for a in case.briefing.actions]
    if not actions:
        return Score(1.0, "")
    vague = [a for a in actions if VAGUE_LEAD.match(a)]
    return Score(
        round(1 - len(vague) / len(actions), 3),
        f"{len(vague)} vague: {vague[0][:60]}" if vague else "",
    )


SCORERS = {
    "no_source_named": no_source_named,
    "no_system_talk": no_system_talk,
    "no_jargon": no_jargon,
    "reading_grade": reading_grade,
    "on_driver_actions": on_driver_actions,
    "concrete_actions": concrete_actions,
}


def flesch_kincaid_grade(text: str) -> float:
    """0.39 * words/sentence + 11.8 * syllables/word - 15.59."""
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    words = re.findall(r"[A-Za-z]+(?:'[a-z]+)?", text)
    if not sentences or not words:
        return 0.0
    syllables = sum(_syllables(w) for w in words)
    return 0.39 * len(words) / len(sentences) + 11.8 * syllables / len(words) - 15.59


# ---------------------------------------------------------------------------
# Private functions
# ---------------------------------------------------------------------------


def _all_text(case: EvalCase) -> str:
    return tract_text(case.briefing) + "\n" + action_text(case.briefing)


def _no_match(pattern: re.Pattern, text: str, label: str) -> Score:
    hits = sorted({m.group(0).lower() for m in pattern.finditer(text)})
    if hits:
        return Score(0.0, f"{label}: {', '.join(hits)}")
    return Score(1.0, "")


def _syllables(word: str) -> int:
    """Vowel groups, minus a silent final e. Close enough for a grade estimate."""
    word = word.lower()
    count = len(re.findall(r"[aeiouy]+", word))
    if word.endswith("e") and not word.endswith(("le", "ee")) and count > 1:
        count -= 1
    return max(count, 1)
