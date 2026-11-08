"""Tests for picking the week a pipeline run processes."""

from datetime import date, datetime

import pytest

from src.utils.weeks import monday_of, week_to_process


def test_monday_of_snaps_back_to_monday():
    assert monday_of(date(2026, 6, 25)) == date(2026, 6, 22)   # Thursday
    assert monday_of(date(2026, 6, 22)) == date(2026, 6, 22)   # already Monday
    assert monday_of(datetime(2026, 6, 28, 23, 59)) == date(2026, 6, 22)  # Sunday night


def test_week_to_process_applies_the_lag():
    # Run on Monday Nov 9, 2026 with a 20-week lag -> week of June 22
    assert week_to_process(datetime(2026, 11, 9, 6, 0), lag_weeks=20) == date(2026, 6, 22)


def test_week_to_process_with_no_lag_is_this_week():
    assert week_to_process(date(2026, 11, 11), lag_weeks=0) == date(2026, 11, 9)


def test_negative_lag_is_rejected():
    with pytest.raises(ValueError):
        week_to_process(date(2026, 11, 9), lag_weeks=-1)
