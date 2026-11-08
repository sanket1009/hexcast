"""
weeks.py
========
Which week a pipeline run works on.

Every table is keyed by week_start_date, the Monday that starts the week.
A scheduled run cannot process its own week: EPA AQS data arrives months
late. So a run processes the Monday that is `lag_weeks` before its run
date. A manual run can name the week instead.

Usage:
    from src.utils.weeks import week_to_process, monday_of
    week = week_to_process(run_date, lag_weeks=20)
"""

from datetime import date, datetime, timedelta


def monday_of(day: date | datetime) -> date:
    """The Monday on or before `day`."""
    if isinstance(day, datetime):
        day = day.date()
    return day - timedelta(days=day.weekday())


def week_to_process(run_date: date | datetime, lag_weeks: int) -> date:
    """Monday of the week that is `lag_weeks` before `run_date`."""
    if lag_weeks < 0:
        raise ValueError("lag_weeks must be zero or more")
    if isinstance(run_date, datetime):
        run_date = run_date.date()
    return monday_of(run_date - timedelta(weeks=lag_weeks))
