"""
base.py
=======
Shared pieces for every eval scorer.

A scorer is a plain function: EvalCase -> Score. It gets one briefing and its
tract input, and returns a number plus a short reason a person can read.
Keeping scorers as plain functions means they run the same in tests, in the
harness, and in CI, with no dependency on a particular eval tool.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass

from src.evals.dataset import EvalCase
from src.rag.briefing import Briefing


@dataclass(frozen=True)
class Score:
    value: float  # 1.0 = pass, 0.0 = fail; rates and grades use their own scale
    reason: str  # why, in one line; empty when there is nothing to report


Scorer = Callable[[EvalCase], Score]


def tract_text(b: Briefing) -> str:
    """The parts that describe this tract: summary, key risks, data gaps.

    Actions are left out on purpose: they come from the policy excerpts, which
    may quote their own numbers and name pollution sources in general terms.
    """
    return "\n".join([b.summary, *b.key_risks, *b.data_gaps])


def action_text(b: Briefing) -> str:
    return "\n".join(a["action"] for a in b.actions)


# Words that show a briefing talks about a risk driver. Matching is
# case-insensitive and on word boundaries.
DRIVER_TERMS = {
    "heat": r"heat|temperature|hot|cooling|cool",
    "air_quality": r"pm2\.5|particle|particulate|air quality|air pollution|pollution|air filtration|filters?",
    "healthcare_access": r"hospitals?|health ?care|emergency (room|department|care)|beds|medical|clinics?|EMTs?|ER",
}
DRIVER_PATTERNS = {
    d: re.compile(rf"\b({terms})\b", re.IGNORECASE) for d, terms in DRIVER_TERMS.items()
}
