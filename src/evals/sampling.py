"""
sampling.py
===========
Stratified samples of eval cases, the same every time for the same input.

Used for the A6 judge sample (60 briefings), the 20 the user labels by hand,
and the A7 perturbation tracts.

Strata are urgency x driver combination: the two things that decide which
action plan and which urgency wording a briefing gets. Every stratum gets at
least `floor` cases (or all it has), the rest are shared out in proportion to
stratum size (largest remainder). Within a stratum, cases are shuffled with a
fixed seed and then ordered so different vulnerable-group sets come first:
the group actions are the other part of a briefing that varies, so the
sample covers as many of them as it can.
"""

import random
from collections import defaultdict

from src.evals.dataset import EvalCase


def stratum(case: EvalCase) -> str:
    return f"{case.tract.urgency}|{'+'.join(sorted(case.tract.drivers))}"


def stratified_sample(
    cases: list[EvalCase], n: int, seed: int = 42, floor: int = 2
) -> list[EvalCase]:
    """n cases across urgency x driver strata, ordered by stratum then geoid."""
    by_stratum: dict[str, list[EvalCase]] = defaultdict(list)
    for c in sorted(cases, key=lambda c: c.briefing.geoid):
        by_stratum[stratum(c)].append(c)
    if n >= len(cases):
        return sorted(cases, key=lambda c: (stratum(c), c.briefing.geoid))

    quota = _allocate({k: len(v) for k, v in by_stratum.items()}, n, floor)
    picked = []
    for key in sorted(by_stratum):
        ordered = _group_diverse_order(by_stratum[key], random.Random(f"{seed}|{key}"))
        picked += sorted(ordered[: quota[key]], key=lambda c: c.briefing.geoid)
    return picked


def _allocate(sizes: dict[str, int], n: int, floor: int) -> dict[str, int]:
    """Floor per stratum first, then the rest by size (largest remainder)."""
    quota = {k: min(size, floor) for k, size in sizes.items()}
    if sum(quota.values()) > n:  # more strata than the floor allows: 1 each
        quota = {k: 1 for k in sizes}
        if len(sizes) > n:
            raise ValueError(f"{len(sizes)} strata do not fit in a sample of {n}")
    left = n - sum(quota.values())
    room = {k: sizes[k] - quota[k] for k in sizes}
    total_room = sum(room.values())
    if left <= 0 or total_room == 0:
        return quota

    shares = {k: left * room[k] / total_room for k in sizes}
    extra = {k: min(int(shares[k]), room[k]) for k in sizes}
    # Hand out what rounding left over, biggest remainders first
    order = sorted(sizes, key=lambda k: (-(shares[k] - int(shares[k])), k))
    remaining = left - sum(extra.values())
    while remaining > 0:
        progressed = False
        for k in order:
            if remaining and extra[k] < room[k]:
                extra[k] += 1
                remaining -= 1
                progressed = True
        if not progressed:
            break
    return {k: quota[k] + extra[k] for k in sizes}


def _group_diverse_order(cases: list[EvalCase], rng: random.Random) -> list[EvalCase]:
    """Shuffle, then take round-robin across vulnerable-group sets."""
    shuffled = list(cases)
    rng.shuffle(shuffled)
    seen: dict[frozenset[str], int] = defaultdict(int)
    keyed = []
    for i, c in enumerate(shuffled):
        groups = frozenset(c.tract.vulnerable_groups)
        keyed.append((seen[groups], i, c))  # nth case with this group set
        seen[groups] += 1
    return [c for _, _, c in sorted(keyed, key=lambda t: (t[0], t[1]))]
